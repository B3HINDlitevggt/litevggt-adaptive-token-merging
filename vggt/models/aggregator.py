# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

### 1. __init__        → 레이어/설정값 세팅
### 2. __build_patch_embed__ → DINOv2 인코더 생성
### 3. _encode_patches_chunked → [추가] DINOv2 청크 인코딩
### 4. forward         → 실제 추론 메인 함수
### 5. _process_frame_attention  → [수정] 청크 처리
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
        self.cal_layer      = [0, 6, 15, 20]
        self.m_u            = None

        self.use_info = True
        self.ga_edge_weight = 0.7
        self.ga_variance_weight = 0.3
        self.ga_depth_boundary_weight = 0.0
        self.ga_depth_map_is_boundary = False
        self.ga_interaction_weight = 0.0
        self.ga_interaction_mode = "sqrt"
        self.ga_laplacian_weight = 0.0
        self.ga_adaptive_weights = False
        self.ga_adaptive_protect_ratio = False
        self.ga_protect_base_ratio = 0.1
        self.ga_protect_complexity_lambda = 0.0
        self.ga_protect_min_ratio = 0.05
        self.ga_protect_max_ratio = 0.2
        self.ga_protect_nms = False
        self.ga_depth_protect_ratio = 0.0

        # ========================================
        # 동적 모드 플래그
        # use_dynamic_protect: Exp 2 - GA Token 비율 동적
        # use_dynamic_grid:    Exp 3 - Grid stride 동적
        # use_sttm:            Exp 5 - STTM 공간+시간 merge
        # ========================================
        self.use_dynamic_protect  = False
        self.use_dynamic_grid     = False
        self.use_sttm             = False
        self.sttm_spatial_thresh  = 0.8
        self.sttm_temporal_thresh = 0.6
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

    def forward(
        self,
        images: torch.Tensor,
        ga_depth: Optional[torch.Tensor] = None,
        return_info_maps: bool = False,
    ) -> Union[Tuple[List[torch.Tensor], int], Tuple[List[torch.Tensor], int, Dict[str, torch.Tensor]]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if self.use_dynamic_protect or self.use_dynamic_grid:
            reset_frame_counter()
            if self.verbose_protect:
                mode_str = []
                if self.use_dynamic_protect:
                    mode_str.append("Dynamic protect_ratio")
                if self.use_dynamic_grid:
                    mode_str.append("Dynamic grid stride")
                print(f"\n{'='*50}")
                print(f"{' + '.join(mode_str)} 모드 | 총 {B*S}개 프레임")
                print(f"{'='*50}")

        if self.use_sttm and self.verbose_protect:
            print(f"\n{'='*50}")
            print(f"STTM 모드 | 총 {B*S}개 프레임 "
                  f"| spatial_thresh={self.sttm_spatial_thresh} "
                  f"| temporal_thresh={self.sttm_temporal_thresh}")
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

        # Step 2: GA Map 계산 (STTM 모드에서는 사용 안 하지만 유지)
        info_map = None
        protect_ratio = self.ga_protect_base_ratio
        protect_aux_map = None
        info_maps = {}
        if self.global_merging and self.use_info and not self.use_sttm:
            if ga_depth is not None:
                if ga_depth.dim() == 4:
                    ga_depth = ga_depth.unsqueeze(2)
                if ga_depth.dim() != 5:
                    raise ValueError(
                        f"Expected ga_depth to have shape [B, S, H, W] or [B, S, 1, H, W], got {tuple(ga_depth.shape)}"
                    )
                ga_depth = ga_depth.view(B * S, ga_depth.shape[2], ga_depth.shape[3], ga_depth.shape[4])

            info_maps = compute_info_maps(
                images,
                patch_tokens,
                depth_map=ga_depth,
                depth_map_is_boundary=self.ga_depth_map_is_boundary,
                edge_weight=self.ga_edge_weight,
                variance_weight=self.ga_variance_weight,
                depth_boundary_weight=self.ga_depth_boundary_weight,
                interaction_weight=self.ga_interaction_weight,
                interaction_mode=self.ga_interaction_mode,
                laplacian_weight=self.ga_laplacian_weight,
                adaptive_weights=self.ga_adaptive_weights,
                adaptive_protect_ratio=self.ga_adaptive_protect_ratio,
                protect_base_ratio=self.ga_protect_base_ratio,
                protect_complexity_lambda=self.ga_protect_complexity_lambda,
                protect_min_ratio=self.ga_protect_min_ratio,
                protect_max_ratio=self.ga_protect_max_ratio,
            )
            if self.ga_depth_protect_ratio > 0.0:
                info_map = info_maps["base_info_map"]
                protect_aux_map = info_maps["depth_info_map"]
            else:
                info_map = info_maps["info_map"]
            protect_ratio = info_maps["protect_ratio"]

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

        for block_idx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos_to_use
                    )
                elif attn_type == "global":
                    tokens, global_idx = self._process_global_attention(
                        tokens,
                        B,
                        S,
                        P,
                        C,
                        global_idx,
                        pos=pos_to_use,
                        info_map=info_map,
                        protect_ratio=protect_ratio,
                        protect_nms=self.ga_protect_nms,
                        protect_aux_map=protect_aux_map,
                        protect_aux_ratio=self.ga_depth_protect_ratio,
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

        if return_info_maps:
            return output_list, self.patch_start_idx, info_maps

        return output_list, self.patch_start_idx

    def set_ga_metric_weights(
        self,
        edge_weight: float = 0.7,
        variance_weight: float = 0.3,
        depth_boundary_weight: float = 0.0,
        depth_map_is_boundary: bool = False,
        interaction_weight: float = 0.0,
        interaction_mode: str = "sqrt",
        laplacian_weight: float = 0.0,
        adaptive_weights: bool = False,
        adaptive_protect_ratio: bool = False,
        protect_base_ratio: float = 0.1,
        protect_complexity_lambda: float = 0.0,
        protect_min_ratio: float = 0.05,
        protect_max_ratio: float = 0.2,
        protect_nms: bool = False,
        depth_protect_ratio: float = 0.0,
    ):
        self.ga_edge_weight = edge_weight
        self.ga_variance_weight = variance_weight
        self.ga_depth_boundary_weight = depth_boundary_weight
        self.ga_depth_map_is_boundary = depth_map_is_boundary
        self.ga_interaction_weight = interaction_weight
        self.ga_interaction_mode = interaction_mode
        self.ga_laplacian_weight = laplacian_weight
        self.ga_adaptive_weights = adaptive_weights
        self.ga_adaptive_protect_ratio = adaptive_protect_ratio
        self.ga_protect_base_ratio = protect_base_ratio
        self.ga_protect_complexity_lambda = protect_complexity_lambda
        self.ga_protect_min_ratio = protect_min_ratio
        self.ga_protect_max_ratio = protect_max_ratio
        self.ga_protect_nms = protect_nms
        self.ga_depth_protect_ratio = depth_protect_ratio

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

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        info_map=None,
        protect_ratio=0.1,
        protect_nms=False,
        protect_aux_map=None,
        protect_aux_ratio=0.0,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        cal_merge = global_idx in self.cal_layer
        verbose   = self.verbose_protect and (global_idx == self.cal_layer[0])

        if self.training:
            tokens, m_u = checkpoint(
                self.global_blocks[global_idx],
                tokens,
                pos,
                self.global_merging,
                info_map,
                protect_ratio,
                protect_nms,
                protect_aux_map,
                protect_aux_ratio,
                cal_merge,
                self.m_u,
                use_reentrant=self.use_reentrant,
            )
        else:
            tokens, m_u = self.global_blocks[global_idx](
                tokens,
                pos=pos,
                global_merging=self.global_merging,
                info_map=info_map,
                protect_ratio=protect_ratio,
                protect_nms=protect_nms,
                protect_aux_map=protect_aux_map,
                protect_aux_ratio=protect_aux_ratio,
                cal_merge=cal_merge,
                m_u=self.m_u,
                use_dynamic_protect=self.use_dynamic_protect,
                use_dynamic_grid=self.use_dynamic_grid,
                use_sttm=self.use_sttm,
                sttm_spatial_thresh=self.sttm_spatial_thresh,
                sttm_temporal_thresh=self.sttm_temporal_thresh,
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
