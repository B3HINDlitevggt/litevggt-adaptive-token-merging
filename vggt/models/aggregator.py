# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

### 1. __init__        → 레이어/설정값 세팅
### 2. __build_patch_embed__ → DINOv2 인코더 생성
### 3. _encode_patches_chunked → DINOv2 청크 인코딩
### 4. forward         → 실제 추론 메인 함수
### 5. _process_frame_attention  → Frame Attention 청크 처리
### 6. _process_global_attention → merge + attention
### 7. slice_expand_and_flatten  → 특수 토큰 확장 유틸

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from merging.merge import compute_info_maps
from merging.complexity import reset_frame_counter

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD  = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.
    """
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                init_values=init_values, qk_norm=qk_norm, rope=self.rope
            )
            for _ in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                init_values=init_values, qk_norm=qk_norm, rope=self.rope
            )
            for _ in range(depth)
        ])

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.camera_token   = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant  = False
        self.num_special    = 5
        self.unmerge_layers = [4, 11, 17, 23]

        self.global_merging = True
        self.use_info       = True
        self.cal_layer      = [0, 6, 15, 20]   # 기본: 4번 재계산. run_experiment에서 덮어쓸 수 있음
        self.m_u            = None

        # ========================================
        # 모드별 플래그
        # use_dynamic_protect    : GA Token 비율 동적 결정
        # use_dynamic_grid       : Grid stride 동적 결정
        # use_sttm               : STTM 공간+시간 merge (기존 STTM)
        # use_quadtree_bipartite : Quadtree + Bipartite merge (★ 새 방식)
        # ========================================
        self.use_dynamic_protect    = False
        self.use_dynamic_grid       = False
        self.use_sttm               = False
        self.use_quadtree_bipartite = False    # ← 새 플래그

        # STTM 하이퍼파라미터
        self.sttm_spatial_thresh  = 0.8
        self.sttm_temporal_thresh = 0.6

        # Quadtree-Bipartite 하이퍼파라미터
        self.qt_spatial_thresh    = 0.8         # quadtree 영역 유사도 임계값
        self.qt_root_block_size   = 8           # 시작 블록 크기 (Level 1)
        self.qt_min_block_size    = 2           # 종료 블록 크기 (Level 3)

        self.verbose_protect      = True

        # DINOv2 / Frame Attention 청크 처리
        self.dino_chunk_size  = 64
        self.frame_chunk_size = 64

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg":  vit_giant2,
            }
            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def _encode_patches_chunked(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2 청크 단위 인코딩 → DINOv2 OOM 방지"""
        N = images.shape[0]
        chunk_size = self.dino_chunk_size
        if chunk_size is None or chunk_size >= N:
            tokens = self.patch_embed(images)
            if isinstance(tokens, dict):
                tokens = tokens["x_norm_patchtokens"]
            return tokens
        all_tokens = []
        for start in range(0, N, chunk_size):
            end   = min(start + chunk_size, N)
            t     = self.patch_embed(images[start:end])
            if isinstance(t, dict):
                t = t["x_norm_patchtokens"]
            all_tokens.append(t)
        return torch.cat(all_tokens, dim=0)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # 모드별 로그 헤더
        if self.use_dynamic_protect or self.use_dynamic_grid:
            reset_frame_counter()
            if self.verbose_protect:
                mode_str = []
                if self.use_dynamic_protect:
                    mode_str.append("Dynamic protect_ratio")
                if self.use_dynamic_grid:
                    mode_str.append("Dynamic grid stride")
                print(f"\n{'='*50}")
                print(f"{' + '.join(mode_str)} 모드 | 총 {B*S}개 프레임 | cal_layer={self.cal_layer}")
                print(f"{'='*50}")

        if self.use_sttm and self.verbose_protect:
            print(f"\n{'='*50}")
            print(f"STTM 모드 | 총 {B*S}개 프레임 "
                  f"| spatial_thresh={self.sttm_spatial_thresh} "
                  f"| temporal_thresh={self.sttm_temporal_thresh} "
                  f"| cal_layer={self.cal_layer}")
            print(f"{'='*50}")

        if self.use_quadtree_bipartite and self.verbose_protect:
            print(f"\n{'='*50}")
            print(f"Quadtree-Bipartite 모드 | 총 {B*S}개 프레임")
            print(f"  spatial_thresh = {self.qt_spatial_thresh}")
            print(f"  root_block     = {self.qt_root_block_size}×{self.qt_root_block_size}")
            print(f"  min_block      = {self.qt_min_block_size}×{self.qt_min_block_size}")
            print(f"  cal_layer      = {self.cal_layer}")
            print(f"{'='*50}")

        # Step 1: 이미지 정규화 + DINOv2 인코딩 (청크 처리)
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.to(torch.bfloat16)
        images = images.view(B * S, C_in, H, W)
        images_dino = images

        patch_tokens = self._encode_patches_chunked(images_dino)
        patch_tokens = patch_tokens.to(torch.bfloat16)

        N, P, C = patch_tokens.shape

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        # Step 2: GA Map 계산 (STTM 모드에서는 사용 안 함)
        info_map = None
        if self.global_merging and self.use_info and not self.use_sttm:
            info_map = compute_info_maps(images, patch_tokens)["info_map"]

        # Step 3: 특수 토큰 붙이기
        camera_token   = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        N, P, C = tokens.shape

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Step 4: Frame + Global Attention 24번 교대 반복
        frame_idx  = 0
        global_idx = 0
        output_list = []
        pos_to_use = pos

        # 새 cal_layer 적용 시 이전 m_u 캐시 초기화
        self.m_u = None

        for block_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos_to_use
                    )
                elif attn_type == "global":
                    tokens, global_idx = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos_to_use, info_map=info_map
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if block_idx in self.unmerge_layers:
                concat_inter = torch.cat([frame_intermediates, tokens], dim=-1).view(B, S, -1, 2*C)
                output_list.append(concat_inter.to(torch.bfloat16))
            else:
                output_list.append(None)

        del concat_inter
        del frame_intermediates
        if info_map is not None:
            del info_map

        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """Frame Attention 청크 처리 → Frame Attention MLP OOM 방지"""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        N_frames  = tokens.shape[0]
        chunk_size = self.frame_chunk_size

        if chunk_size is None or chunk_size >= N_frames:
            if self.training:
                tokens, _ = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, _ = self.frame_blocks[frame_idx](tokens, pos=pos)
        else:
            out_chunks = []
            for start in range(0, N_frames, chunk_size):
                end       = min(start + chunk_size, N_frames)
                tok_chunk = tokens[start:end]
                pos_chunk = pos[start:end] if pos is not None else None
                if self.training:
                    out_chunk, _ = checkpoint(
                        self.frame_blocks[frame_idx], tok_chunk, pos_chunk,
                        use_reentrant=self.use_reentrant
                    )
                else:
                    out_chunk, _ = self.frame_blocks[frame_idx](tok_chunk, pos=pos_chunk)
                out_chunks.append(out_chunk)
            tokens = torch.cat(out_chunks, dim=0)

        frame_idx += 1
        intermediates = tokens
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, info_map=None):
        """Global Attention + merge (모드별 분기)"""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        cal_merge = global_idx in self.cal_layer
        verbose   = self.verbose_protect and (global_idx == self.cal_layer[0])

        if self.training:
            tokens, m_u = checkpoint(
                self.global_blocks[global_idx], tokens, pos,
                global_merging=self.global_merging,
                info_map=info_map,
                cal_merge=cal_merge,
                m_u=self.m_u,
                use_reentrant=self.use_reentrant
            )
        else:
            tokens, m_u = self.global_blocks[global_idx](
                tokens, pos=pos,
                global_merging=self.global_merging,
                info_map=info_map,
                cal_merge=cal_merge,
                m_u=self.m_u,
                use_dynamic_protect=self.use_dynamic_protect,
                use_dynamic_grid=self.use_dynamic_grid,
                use_sttm=self.use_sttm,
                use_quadtree_bipartite=self.use_quadtree_bipartite,   # ← 추가
                sttm_spatial_thresh=self.sttm_spatial_thresh,
                sttm_temporal_thresh=self.sttm_temporal_thresh,
                qt_spatial_thresh=self.qt_spatial_thresh,             # ← 추가
                qt_root_block_size=self.qt_root_block_size,           # ← 추가
                qt_min_block_size=self.qt_min_block_size,             # ← 추가
                verbose=verbose,
            )

        self.m_u = m_u

        # STTM 실제 토큰 수 저장 (cal_layer에서 새로 계산된 경우)
        if self.use_sttm:
            block = self.global_blocks[global_idx - 1] if global_idx > 0 else None
            if block is not None and hasattr(block, '_sttm_tokens_after'):
                self.last_sttm_tokens_after = block._sttm_tokens_after

        global_idx += 1
        return tokens.view(B * S, P, C), global_idx


def slice_expand_and_flatten(token_tensor, B, S):
    query  = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined