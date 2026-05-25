# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from merging.merge import (
    token_merge_bipartite2d, token_merge_bipartite2d_multi_batch
)
from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

XFORMERS_AVAILABLE = False


class LayerNormMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(dim))
        self.layer_norm_bias = nn.Parameter(torch.zeros(dim))
        self.fc1_weight = nn.Parameter(torch.empty(hidden_dim, dim))
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight = nn.Parameter(torch.empty(dim, hidden_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.fc1_weight, a=0.01)
        nn.init.kaiming_uniform_(self.fc2_weight, a=0.01)

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-1:], self.layer_norm_weight, self.layer_norm_bias)
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = F.gelu(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x


# 디버깅용 전역 카운터
_debug_layer_count = {"count": 0}

def reset_debug_counter():
    _debug_layer_count["count"] = 0


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        patch_width: int = 31,
        patch_height: int = 25,
        rope=None
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.patch_width = patch_width
        self.patch_height = patch_height

        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm,
            fused_attn=fused_attn, rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.normMlp = LayerNormMLP(dim, mlp_hidden_dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
        self.debug = False
        self.debug_layer = None

    def forward(
        self,
        x,
        pos=None,
        global_merging=False,
        info_map=None,
        protect_ratio=0.1,
        protect_nms=False,
        protect_aux_map=None,
        protect_aux_ratio=0.0,
        cal_merge=False,
        m_u=None,
        use_dynamic_protect: bool = False,  # Exp 2: GA Token 비율 동적
        use_dynamic_grid: bool = False,     # Exp 3: Grid stride 동적
        use_sttm: bool = False,             # Exp 5: STTM merge
        sttm_spatial_thresh: float = 0.8,
        sttm_temporal_thresh: float = 0.6,
        verbose: bool = True,
    ):
        x_norm = self.norm1(x)

        if global_merging and cal_merge:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)
            merge_ratio = 0.9
            r = int(x_norm.shape[1] * merge_ratio)

            # 디버깅 출력 (기존 유지)
            layer_idx = _debug_layer_count["count"]
            _debug_layer_count["count"] += 1
            should_debug = self.debug and (
                self.debug_layer is None or layer_idx == self.debug_layer
            )
            if should_debug:
                B, N, C = x_norm.shape
                tokens_per_img = self.patch_width * self.patch_height + 5
                num_imgs = N // tokens_per_img
                protect_ratio = 0.1
                ga_per_frame  = max(1, int(self.patch_width * self.patch_height * protect_ratio))
                dst_per_frame = (self.patch_width // 2) * (self.patch_height // 2)
                src_per_frame = self.patch_width * self.patch_height - ga_per_frame - dst_per_frame
                special = 5
                print(f"\n{'='*60}")
                print(f"[DEBUG] Global Attention Layer {layer_idx} | cal_merge=True")
                print(f"{'='*60}")
                print(f"  총 프레임 수:            {num_imgs}장")
                print(f"  전체 토큰 수 (merge 전): {N:,}개")
                print(f"  프레임당 전체 토큰:      {tokens_per_img}개")
                print(f"    ├── 특수 토큰:         {special}개 (항상 보호)")
                print(f"    ├── GA Token:          {ga_per_frame}개 ({protect_ratio*100:.0f}%, 보호)")
                print(f"    ├── Dst Token:         {dst_per_frame}개 (살아남음)")
                print(f"    └── Src Token:         {src_per_frame}개 (제거됨)")
                print(f"  merge 후 예상 토큰:      {(special + ga_per_frame + dst_per_frame) * num_imgs:,}개")
                print(f"  압축률:                  {(1 - (special + ga_per_frame + dst_per_frame) / tokens_per_img)*100:.1f}%")
                print(f"{'='*60}")

            with torch.no_grad():
                # -----------------------------------------------
                # STTM merge (Exp 5)
                # use_sttm=True → STTM 공간+시간 merge
                # -----------------------------------------------
                if use_sttm:
                    from merging.sttm_merge import token_merge_sttm
                    tokens_per_img = self.patch_width * self.patch_height + 5
                    num_imgs = x_norm.shape[1] // tokens_per_img
                    # cal_merge=True → STTM 새로 계산 + tyxyx_tlbr 캐시 저장
                    # cal_merge=False → 캐시된 tyxyx_tlbr 재사용
                    cached_tlbr = getattr(self, '_sttm_cached_tlbr', None)
                    m, u, new_tlbr, tokens_after_sttm = token_merge_sttm(
                        x_norm,
                        w=self.patch_width,
                        h=self.patch_height,
                        r=r,
                        spatial_thresh=sttm_spatial_thresh,
                        temporal_thresh=sttm_temporal_thresh,
                        tokens_per_img=tokens_per_img,
                        num_imgs=num_imgs,
                        verbose=verbose,
                        cached_tlbr=cached_tlbr,
                    )
                    # 새로 계산된 경우 캐시 업데이트
                    if new_tlbr is not None:
                        self._sttm_cached_tlbr = new_tlbr
                        self._sttm_tokens_after = tokens_after_sttm
                else:
                    # -----------------------------------------------
                    # 기존 방식: GA/Dst/Src merge
                    # -----------------------------------------------
                    # Step 1: protect_ratio 결정
                    if use_dynamic_protect and info_map is not None:
                        protect_ratio = None

                    # Step 2: grid stride 결정
                    if use_dynamic_grid and info_map is not None:
                        from merging.complexity import get_dynamic_grid_stride
                        stride = get_dynamic_grid_stride(
                            info_map[:, 0], min_stride=2, max_stride=4, verbose=verbose,
                        )
                        sx, sy = stride, stride
                    else:
                        sx, sy = 2, 2

                    m, u = token_merge_bipartite2d(
                        x_norm,
                        self.patch_width,
                        self.patch_height,
                        sx, sy,
                        r,
                        False,
                        generator,
                        enable_protection=True,
                        info_map=info_map,
                        use_dynamic_protect=use_dynamic_protect,
                        protect_ratio=protect_ratio if protect_ratio is not None else 0.1,
                        protect_nms=protect_nms,
                        protect_aux_map=protect_aux_map,
                        protect_aux_ratio=protect_aux_ratio,
                        verbose=verbose,
                    )
            m_u = (m, u, getattr(m, "b_idx", None))

        att_out = self.attn(x_norm, pos=pos, global_merging=global_merging, m_u=m_u)
        x = x + self.ls1(att_out)
        x = x + self.ls2(self.normMlp(x))

        if info_map is not None:
            del info_map

        return x, m_u


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    if pos is not None:
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))
    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        assert isinstance(self.attn, MemEffAttention)
        if self.training and self.sample_drop_ratio > 0.0:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)
            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.normMlp(x)
            x_list = drop_add_residual_stochastic_depth_list(
                x_list, residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list, residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.normMlp(x))
            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
