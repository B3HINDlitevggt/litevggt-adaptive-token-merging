# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


"""
aggregator.py
    ↓
DINOv2 인코딩 (토큰 추출)
    ↓
GA Map 계산 (info_map)
    ↓
24번 반복:
    Frame Attention
    ↓
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Global Attention ← 지금 보고 있는 부분
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓
        block.py
            GA Map 기반으로 GA/Dst/Src Token 분류
            merge index 계산 → m_u
        ↓
        attention.py
            m_a() → merge (Src를 Dst에 흡수, 토큰 압축)
            attention 수행
            u_a() → unmerge (토큰 복원)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
output_list 반환
    ↓
camera_head / depth_head
"""

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
# =====================================================================
### 1. LayerNormMLP
# FP8 Transformer Engine 없이 동작하는 대체 MLP 레이어
# LayerNorm + Linear + GELU + Linear 구조
# =====================================================================

class LayerNormMLP(nn.Module):
    """te.LayerNormMLP 대체 — key 이름을 te_dict.pt 체크포인트에 맞춤"""
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

# =====================================================================
### 2. Block: Frame/Global Attention 레이어 하나
# aggregator.py의 frame_blocks, global_blocks 각각의 레이어가 이 클래스
# Frame Attention: global_merging=False로 호출
# Global Attention: global_merging=True로 호출 → 여기서 merge 발생
# =====================================================================
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
        # Attention 레이어 (attention.py) → 실제 merge/unmerge도 여기서 실행
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.normMlp = LayerNormMLP(dim, mlp_hidden_dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path


    # =====================================================================
    ### 3. forward: Block의 핵심 실행 함수
    # aggregator.py에서 global_blocks[i](tokens, ...) 호출 시 여기 실행
    #
    # 핵심 흐름:
    #   1. 정규화
    #   2. merge index 계산 (cal_merge==True일 때만)
    #   3. attention.py로 넘김 (실제 merge/unmerge + attention 실행)
    #   4. MLP
    # =====================================================================
    def forward(
        self,
        x,
        pos=None,
        global_merging=False,
        info_map=None,
        cal_merge=False,
        m_u=None,
        use_dynamic_protect=False,  # ← 동적 protect_ratio 사용 여부
        verbose=True,               # ← 프레임별 출력 여부
    ):
            # -----------------------------------------------
        ##### Step 1: 정규화
        # -----------------------------------------------
        x_norm = self.norm1(x)


        ##### Step 2: merge index 계산
        # global_merging=True이고 cal_merge=True일 때만 실행
        # cal_layer [0,6,15,20]에서만 해당 → 나머지는 m_u 캐시 재사용
        #
        # 여기서 결정되는 것:
        #   - GA Token (상위 10%): 보호, merge 대상 제외
        #   - Dst Token (2×2 그리드당 1개): 살아남아서 Src 흡수
        #   - Src Token (나머지): Dst에 흡수되어 사라짐
        #   → m, u = (merge함수, unmerge함수) 로 저장
        # -----------------------------------------------
        if global_merging and cal_merge:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)
            merge_ratio = 0.9
            r = int(x_norm.shape[1] * merge_ratio)

            with torch.no_grad():
                m, u = token_merge_bipartite2d(
                    x_norm,
                    self.patch_width,
                    self.patch_height,
                    2,
                    2,
                    r,
                    False,
                    generator,
                    enable_protection=True,
                    info_map=info_map,
                    use_dynamic_protect=use_dynamic_protect,  # ← 전달
                    verbose=verbose,                           # ← 전달
                )

            m_u = (m, u)
        # -----------------------------------------------
        ##### Step 3: Attention 실행
        # attention.py로 넘어가서:
        #   → merge 실행 (m_u의 merge함수 호출, 토큰 압축)
        #   → attention 수행 (압축된 토큰으로)
        #   → unmerge 실행 (m_u의 unmerge함수 호출, 토큰 복원)
        # -----------------------------------------------
        att_out = self.attn(x_norm, pos=pos, global_merging=global_merging, m_u=m_u)

        # -----------------------------------------------
        ##### Step 4: Residual 연결 + MLP
        # attention 결과를 원본에 더하고 MLP 통과
        # -----------------------------------------------
        x = x + self.ls1(att_out)

        x = x + self.ls2(self.normMlp(x))

        if info_map is not None:
            del info_map

        return x, m_u


# =====================================================================
### 4. 나머지 함수들 (학습 시 사용, 추론에서는 거의 안 쓰임)
# drop_add_residual_stochastic_depth: 학습 중 일부 샘플 드롭
# NestedTensorBlock: xFormers 사용 시 중첩 텐서 처리
# =====================================================================

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
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
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