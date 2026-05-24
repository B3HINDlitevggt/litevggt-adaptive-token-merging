# 05.25.2026
# 준혁님이 올려주신 aggregator.py  + Sobel Based Frame Selection 
# Aggregator variant with two-stage frame selection and info-map global token merging.
#
# Frame selection :
#   Info-map deduplication (_select_diverse_frames): GPU-side, reuses info_map already
#      computed for token merging — drops low-information frames before global attention.
# 바꿀 수 있는 Parameter : 
#   use_info_dedup: bool = True,   # master switch
#   min_dedup_frames: int = 2,     # always keep at least this many frames
#   max_drop_ratio: float = 0.50,  # never drop more than this fraction of frames
#   score_floor: float = 0.11,      # hard quality gate: drop frames below this score

import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from merging.merge import (
    compute_info_maps
)
from merging.complexity import reset_frame_counter

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
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
        # --- Info-map based frame selection ---
        use_info_dedup: bool = True,   # master switch
        min_dedup_frames: int = 2,     # always keep at least this many frames
        max_drop_ratio: float = 0.50,  # never drop more than this fraction of frames
        score_floor: float = 0.11,      # hard quality gate: drop frames below this score
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope
                )
                for _ in range(depth)
            ]
        )
        

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False
        self.num_special = 5
        self.unmerge_layers = [4, 11, 17, 23]  

        self.global_merging = True

        self.use_info = True
        # self.cal_layer = list(range(24))
        self.cal_layer = [0, 6, 15, 20]
        self.m_u = None

        self.use_info_dedup = use_info_dedup
        self.min_dedup_frames = min_dedup_frames
        self.max_drop_ratio = max_drop_ratio
        self.score_floor = score_floor

        self.use_dynamic_protect = False
        self.use_dynamic_grid = False
        self.use_sttm = False
        self.sttm_spatial_thresh = 0.8
        self.sttm_temporal_thresh = 0.6
        self.verbose_protect = True
        self.dino_chunk_size = 64
        self.frame_chunk_size = 64

        self.last_sel_idx = None
        self.last_sttm_tokens_after = None


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
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
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

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)


    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
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
                print(f"{' + '.join(mode_str)} mode | total {B*S} frames")
                print(f"{'='*50}")

        if self.use_sttm and self.verbose_protect:
            print(f"\n{'='*50}")
            print(
                f"STTM mode | total {B*S} frames | spatial_thresh={self.sttm_spatial_thresh} | "
                f"temporal_thresh={self.sttm_temporal_thresh}"
            )
            print(f"{'='*50}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        images = images.to(torch.bfloat16)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        images_dino = images
        patch_tokens = self._encode_patches_chunked(images_dino)

        N, P, C = patch_tokens.shape
        patch_tokens= patch_tokens.to(torch.bfloat16)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
        
        info_map = None
        if (self.global_merging and self.use_info and not self.use_sttm) or self.use_info_dedup:
            info_map = compute_info_maps(images, patch_tokens)["info_map"]

        # Frame selection: keep the most informative frames using info_map scores.
        # info_map is already computed above — zero extra cost.
        sel_idx = None
        S_prime = S
        if self.use_info_dedup and S > 1 and info_map is not None:
            sel_idx = self._select_diverse_frames(info_map, B, S, images.device)
            S_prime = sel_idx.shape[0]
            if S_prime == S:
                sel_idx = None
        self.last_sel_idx = sel_idx

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        N, P, C = tokens.shape

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        info_map_sel = info_map
        if sel_idx is not None and info_map is not None:
            Hp, Wp = info_map.shape[-2:]
            info_map_sel = (info_map.view(B, S, 1, Hp, Wp)[:, sel_idx]
                            .contiguous().view(B * S_prime, 1, Hp, Wp))

        frame_idx = 0
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
                    if sel_idx is not None:
                        tokens_sel = (tokens.view(B, S, P, C)[:, sel_idx]
                                      .contiguous().view(B * S_prime, P, C))
                        pos_sel = (pos_to_use.view(B, S, P, 2)[:, sel_idx]
                                   .contiguous().view(B * S_prime, P, 2)) if pos_to_use is not None else None

                        tokens_sel, global_idx = self._process_global_attention(
                            tokens_sel, B, S_prime, P, C, global_idx,
                            pos=pos_sel, info_map=info_map_sel
                        )

                        tokens_2d = tokens.view(B, S, P, C)
                        tokens_2d[:, sel_idx] = tokens_sel.view(B, S_prime, P, C)
                        tokens = tokens_2d.view(B * S, P, C)
                    else:
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
        del info_map

        return output_list, self.patch_start_idx

    def _encode_patches_chunked(self, images: torch.Tensor) -> torch.Tensor:
        """Encode patches in chunks to reduce DINOv2 peak memory."""
        n_images = images.shape[0]
        chunk_size = self.dino_chunk_size
        if chunk_size is None or chunk_size >= n_images:
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            return patch_tokens

        all_tokens = []
        for start in range(0, n_images, chunk_size):
            end = min(start + chunk_size, n_images)
            token_chunk = self.patch_embed(images[start:end])
            if isinstance(token_chunk, dict):
                token_chunk = token_chunk["x_norm_patchtokens"]
            all_tokens.append(token_chunk)
        return torch.cat(all_tokens, dim=0)

    # @classmethod
    # def pre_select_frames_cpu(
    #     cls,
    #     images_rgb: List[np.ndarray],
    #     n_keep: int,
    #     score_floor: float = 0.10,
    #     **kwargs,
    # ) -> List[int]:
    #     """Select n_keep high-edge-content frames using CPU Sobel before GPU loading."""
    #     S = len(images_rgb)
    #     if S <= n_keep:
    #         return list(range(S))

    #     scores = []
    #     for img in images_rgb:
    #         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    #         sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    #         sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    #         scores.append(np.sqrt(sx**2 + sy**2).mean())

    #     scores = np.array(scores)
    #     max_s = scores.max() if scores.max() > 0 else 1.0
    #     above = np.where(scores / max_s >= score_floor)[0]

    #     if len(above) >= n_keep:
    #         top = above[np.argsort(scores[above])[::-1][:n_keep]]
    #     else:
    #         top = np.argsort(scores)[::-1][:n_keep]

    #     selected = sorted(top.tolist())
    #     print(f"[Sobel CPU pre-select] {S} → {len(selected)} frames")
    #     return selected

    def _select_diverse_frames(
        self,
        info_map: torch.Tensor,
        B: int,
        S: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Keep the frames with the highest average information score.

        info_map is [B*S, 1, Hp, Wp] where each value in [0,1] represents
        patch informativeness (edge strength + feature variance). Frames with
        low mean scores are blurry or featureless and contribute little to
        reconstruction — they are dropped first.
        """
        import math
        min_keep = max(math.ceil(S * (1.0 - self.max_drop_ratio)), self.min_dedup_frames)

        # Mean info score per frame, averaged across batch
        # info_map: [B*S, 1, Hp, Wp] → [B, S] → [S]
        scores = info_map.float().view(B, S, -1).mean(dim=-1).mean(dim=0)  # [S]

        # Primary gate: keep all frames with score >= score_floor
        above = (scores >= self.score_floor).nonzero(as_tuple=True)[0]

        # Safety: if score_floor drops too many, add back the highest-scoring dropped frames
        if above.shape[0] < min_keep:
            below = (scores < self.score_floor).nonzero(as_tuple=True)[0]
            extra = below[scores[below].argsort(descending=True)[: min_keep - above.shape[0]]]
            selected_idx = torch.cat([above, extra])
        else:
            selected_idx = above

        if 0 not in selected_idx.tolist():
            selected_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=selected_idx.device), selected_idx])
        selected, _ = selected_idx.sort()

        selected_set = set(selected.tolist())
        frame_scores = {
            i: {"score": round(scores[i].item(), 4), "kept": i in selected_set}
            for i in range(S)
        }
        logger.info(
            f"[infomap dedup] total={S}, min_keep={min_keep}, kept={len(selected)}, "
            f"removed={S - len(selected)}, score_floor={self.score_floor}"
        )
        logger.info(f"[infomap dedup] kept indices: {selected.tolist()}")
        logger.info(f"[infomap dedup] frame scores: {frame_scores}")

        return selected

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        n_frames = tokens.shape[0]
        chunk_size = self.frame_chunk_size

        if chunk_size is None or chunk_size >= n_frames:
            if self.training:
                tokens, _ = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens, _ = self.frame_blocks[frame_idx](tokens, pos=pos)
        else:
            out_chunks = []
            for start in range(0, n_frames, chunk_size):
                end = min(start + chunk_size, n_frames)
                tok_chunk = tokens[start:end]
                pos_chunk = pos[start:end] if pos is not None else None
                if self.training:
                    out_chunk, _ = checkpoint(
                        self.frame_blocks[frame_idx], tok_chunk, pos_chunk, use_reentrant=self.use_reentrant
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
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        if global_idx in self.cal_layer:
            cal_merge = True
        else:
            cal_merge = False

        verbose = self.verbose_protect and (global_idx == self.cal_layer[0])

        if self.training:
            tokens, m_u = checkpoint(
                self.global_blocks[global_idx],
                tokens,
                pos,
                global_merging=self.global_merging,
                info_map=info_map,
                cal_merge=cal_merge,
                m_u=self.m_u,
                use_dynamic_protect=self.use_dynamic_protect,
                use_dynamic_grid=self.use_dynamic_grid,
                use_sttm=self.use_sttm,
                sttm_spatial_thresh=self.sttm_spatial_thresh,
                sttm_temporal_thresh=self.sttm_temporal_thresh,
                verbose=verbose,
                use_reentrant=self.use_reentrant,
            )
        else:
            tokens, m_u = self.global_blocks[global_idx](
                tokens,
                pos=pos,
                global_merging=self.global_merging,
                info_map=info_map,
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

        if self.use_sttm:
            block = self.global_blocks[global_idx]
            if hasattr(block, "_sttm_tokens_after"):
                self.last_sttm_tokens_after = block._sttm_tokens_after

        global_idx += 1

        return tokens.view(B * S, P, C), global_idx
    


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
