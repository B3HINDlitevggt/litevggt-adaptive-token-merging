# merging/sttm_bipartite_merge.py
"""
프레임별 자유 Quadtree + ToMe-style Bipartite Token Merging

핵심 아이디어:
  - 각 프레임이 독립적으로 quadtree 분할 (944개 동의 같은 거 없음)
  - 모든 프레임의 노드들을 평탄화해서 전역 ID로 관리
  - 각 노드 안에서 info_map 최소 = Dst (원본 유지)
  - 나머지 패치 → Src 후보 → bipartite matching으로 Dst에 흡수
  - merge/unmerge는 scatter_reduce / gather로 벡터화 (Python loop 최소화)

장점:
  - 프레임마다 자유로운 압축 (단순 영역 잘 합침, 복잡 영역 보존)
  - 첫 프레임이 복잡해도 다른 프레임에 영향 없음
  - GPU 친화적 (scatter/gather 기반)
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Callable, Optional, Union

from merging.complexity import get_dynamic_protect_ratio_single


# 노드 크기 통계 (전역, 로그용)
_node_size_stats = {"counter": {2: 0, 4: 0, 8: 0}, "total_nodes": 0, "total_patches": 0}


def reset_node_stats():
    global _node_size_stats
    _node_size_stats = {"counter": {2: 0, 4: 0, 8: 0}, "total_nodes": 0, "total_patches": 0}


def get_node_stats():
    return dict(_node_size_stats)


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, num_src, C = a.shape
    original_dtype = a.dtype
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(x, extra_tensors=None, extra_tensors_2=None):
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    return x


# =====================================================================
# 프레임별 독립 Quadtree 영역 결정
# 출력: 각 패치가 어느 "전역 노드 ID"에 속하는지의 매핑
# =====================================================================
def build_per_frame_quadtree(
    patch_features: torch.Tensor,   # [N_img, C, H, W]
    spatial_thresh: float = 0.8,
    root_block_size: int = 8,
    min_block_size: int = 2,
) -> Tuple[torch.Tensor, int]:
    """
    각 프레임이 독립적으로 quadtree로 영역 분할.
    
    Returns:
        patch_to_node: [N_img, H, W] long tensor.
            patch_to_node[f, y, x] = 그 패치가 속한 전역 노드 ID (0부터 시작)
        total_nodes: 전체 노드 수
    """
    device = patch_features.device
    N_img, C, H, W = patch_features.shape

    # 각 패치의 전역 노드 ID. -1로 초기화
    patch_to_node = torch.full((N_img, H, W), -1, device=device, dtype=torch.long)
    
    # 다음에 할당할 전역 노드 ID (전체 누적 카운터)
    next_node_id = 0

    # ====================================================
    # 프레임 단위 루프 (프레임별 독립 quadtree)
    # 각 프레임 안의 처리는 벡터화
    # ====================================================
    for frame_idx in range(N_img):
        frame_features = patch_features[frame_idx]  # [C, H, W]
        
        # 이 프레임의 처리할 블록들
        pending_blocks = []
        for y0 in range(0, H, root_block_size):
            for x0 in range(0, W, root_block_size):
                pending_blocks.append((y0, x0, root_block_size))

        while len(pending_blocks) > 0:
            y0, x0, size = pending_blocks.pop()
            y1 = min(y0 + size, H)
            x1 = min(x0 + size, W)
            actual_h = y1 - y0
            actual_w = x1 - x0

            # 종료 조건: 최소 크기 도달 또는 분할 불가
            if size <= min_block_size or actual_h < 2 or actual_w < 2:
                patch_to_node[frame_idx, y0:y1, x0:x1] = next_node_id
                next_node_id += 1
                _update_stats(size, actual_h * actual_w, n_img=1)
                continue

            # 이 프레임의 이 영역이 균일한지 판정
            region = frame_features[:, y0:y1, x0:x1]  # [C, h, w]
            is_homo = _check_homogeneous_single(region, spatial_thresh)

            if is_homo:
                # 합침: 모든 패치에 같은 노드 ID 부여
                patch_to_node[frame_idx, y0:y1, x0:x1] = next_node_id
                next_node_id += 1
                _update_stats(size, actual_h * actual_w, n_img=1)
            else:
                # 쪼갬: 4개 자식 블록을 큐에 추가
                mid_y = y0 + actual_h // 2
                mid_x = x0 + actual_w // 2
                half = size // 2
                pending_blocks.append((y0, x0, half))
                pending_blocks.append((y0, mid_x, half))
                pending_blocks.append((mid_y, x0, half))
                pending_blocks.append((mid_y, mid_x, half))

    return patch_to_node, next_node_id


def _check_homogeneous_single(region: torch.Tensor, thresh: float) -> bool:
    """
    한 프레임 한 영역이 균일한지 판정 (자식 4개 평균 vs 부모 평균 cosine sim).
    Returns: bool
    """
    C, H, W = region.shape
    if H < 2 or W < 2:
        return True

    parent_mean = region.mean(dim=(1, 2))  # [C]
    h_mid, w_mid = H // 2, W // 2
    q1 = region[:, :h_mid, :w_mid].mean(dim=(1, 2))
    q2 = region[:, :h_mid, w_mid:].mean(dim=(1, 2))
    q3 = region[:, h_mid:, :w_mid].mean(dim=(1, 2))
    q4 = region[:, h_mid:, w_mid:].mean(dim=(1, 2))

    parent_norm = F.normalize(parent_mean.float(), dim=-1)
    sims = []
    for q in (q1, q2, q3, q4):
        q_norm = F.normalize(q.float(), dim=-1)
        sims.append((parent_norm * q_norm).sum().item())

    return all(s >= thresh for s in sims)


def _update_stats(size, num_patches, n_img=1):
    global _node_size_stats
    if size in _node_size_stats["counter"]:
        _node_size_stats["counter"][size] += n_img
    _node_size_stats["total_nodes"] += n_img
    _node_size_stats["total_patches"] += n_img * num_patches


# =====================================================================
# Main 함수: 프레임별 자유 Quadtree + ToMe Bipartite
# =====================================================================
def token_merge_quadtree_bipartite(
    metric: torch.Tensor,           # [B, N, C] - Global Attention 입력
    patch_features: torch.Tensor,   # [N_img, C, H, W] - quadtree 판정용
    w: int, h: int,                 # patch grid 크기
    r: int,                         # 목표 제거 토큰 수 (참고만, 실제로는 노드 구조가 결정)
    spatial_thresh: float = 0.8,
    root_block_size: int = 8,
    min_block_size: int = 2,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,                  # [N_img, 1, H, W]
    use_dynamic_protect: bool = False,
    protect_ratio: float = 0.1,
    verbose: bool = True,
) -> Tuple[Callable, Callable]:
    """
    프레임별 자유 Quadtree + ToMe Bipartite token merging.
    
    동작:
      1. 각 프레임이 독립적으로 quadtree 영역 분할
      2. 각 노드 안에서 info_map 최소 위치 = Dst
      3. 나머지 패치 = Src 후보
      4. Bipartite matching으로 Src를 Dst에 흡수
      5. GA(보호) 토큰은 따로 보존
    """
    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    device = metric.device
    gather = torch.gather
    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N

    with torch.no_grad():
        # ============================================================
        # Step 1: GA 선정 (기존 ToMe와 동일)
        # ============================================================
        if enable_protection:
            if info_map is not None:
                info = info_map[:, 0].to(device)
                if use_dynamic_protect:
                    protect_ratio = get_dynamic_protect_ratio_single(
                        info, min_ratio=0.05, max_ratio=0.20, verbose=verbose,
                    )
                k = max(1, int(info.shape[-2] * info.shape[-1] * protect_ratio))
                topk_idx = info.flatten(1).topk(k, dim=1).indices  # [num_imgs, k]
                offsets = torch.arange(num_imgs, device=device) * tokens_per_img + 5
                protected_indices = (topk_idx + offsets[:, None]).flatten()
                num_protected = protected_indices.numel()
            else:
                num_protected = int(N * 0.1)
                step = max(1, N // num_protected)
                protected_indices = torch.arange(0, N, step, device=device)[:num_protected]
        else:
            protected_indices = None
            num_protected = 0

        # ============================================================
        # Step 2: 프레임별 독립 Quadtree
        # ============================================================
        reset_node_stats()
        patch_to_node, total_nodes = build_per_frame_quadtree(
            patch_features,
            spatial_thresh=spatial_thresh,
            root_block_size=root_block_size,
            min_block_size=min_block_size,
        )
        # patch_to_node: [num_imgs, H, W] - 각 패치의 전역 노드 ID
        # total_nodes: 모든 프레임의 노드 수 합계

        if verbose:
            stats = get_node_stats()
            cnt = stats["counter"]
            total = stats["total_nodes"]
            if total > 0:
                pct_8 = cnt.get(8, 0) / total * 100
                pct_4 = cnt.get(4, 0) / total * 100
                pct_2 = cnt.get(2, 0) / total * 100
                avg = stats["total_patches"] / max(total, 1)
                print(f"\n[Quadtree-PerFrame] 노드 분포 (전체 {total:,}개): "
                      f"8×8={cnt.get(8,0):,}({pct_8:.1f}%) | "
                      f"4×4={cnt.get(4,0):,}({pct_4:.1f}%) | "
                      f"2×2={cnt.get(2,0):,}({pct_2:.1f}%) | "
                      f"평균 패치/노드: {avg:.2f}")
                print(f"  → 프레임당 평균 노드 수: {total/num_imgs:.1f}개")

        # ============================================================
        # Step 3: 각 노드에서 Dst 선정 (info_map 최소 위치)
        # ============================================================
        # 전체 토큰 시퀀스에서의 패치 인덱스 만들기
        # 패치 i의 전체 시퀀스 인덱스: frame_idx * tokens_per_img + 5 + (y * w + x)
        
        # patch_to_node를 평탄화: [num_imgs * H * W]
        flat_node_ids = patch_to_node.flatten()  # 각 패치의 노드 ID
        
        # 전체 시퀀스에서 패치의 인덱스 계산
        # frame f의 (y, x) 패치는 시퀀스 인덱스 f*tokens_per_img + 5 + y*w + x
        frame_offsets = torch.arange(num_imgs, device=device) * tokens_per_img + 5  # [num_imgs]
        patch_local_idx = torch.arange(h * w, device=device)  # [H*W]
        patch_seq_idx = (frame_offsets[:, None] + patch_local_idx[None, :]).flatten()  # [num_imgs * H*W]
        
        # info_map 평탄화 (Dst 선정용)
        if info_map is not None:
            score_flat = info_map[:, 0].flatten().float()  # [num_imgs * H * W]
        else:
            score_flat = torch.rand(num_imgs * h * w, device=device, generator=generator)

        # 각 노드별 최소 score 위치를 Dst로 선정
        # scatter_reduce로 노드별 최소값과 그 위치 찾기
        # 트릭: score에 노드 ID를 곱한 큰 수를 더해서 argmin 효과
        
        # 방법: 각 노드별로 (score, patch_seq_idx)를 가지고
        # 같은 노드 ID인 것 중에서 score 최소인 patch_seq_idx를 찾음
        
        dst_seq_indices = _find_dst_per_node(
            flat_node_ids, score_flat, patch_seq_idx, total_nodes
        )  # [total_nodes] - 각 노드의 Dst가 될 전체 시퀀스 인덱스

        # ============================================================
        # Step 4: Dst/Src 마킹
        # ============================================================
        idx_buffer_seq = torch.zeros(N, device=device, dtype=torch.int64)
        
        # 첫 프레임은 통째로 Dst (ToMe 컨벤션 유지)
        idx_buffer_seq[:tokens_per_img] = -1

        # 다른 프레임의 특수 토큰 5개도 Dst
        if num_imgs > 1:
            cls_indices = (
                torch.arange(1, num_imgs, device=device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=device)
            idx_buffer_seq[cls_indices.flatten()] = -1

            # 첫 프레임 제외한 각 노드의 Dst 위치를 -1로 마킹
            # dst_seq_indices에서 첫 프레임(0~tokens_per_img-1)에 속한 건 제외
            mask = dst_seq_indices >= tokens_per_img
            non_first_dst = dst_seq_indices[mask]
            idx_buffer_seq[non_first_dst] = -1

        # ============================================================
        # Step 5: ToMe와 동일한 bipartite matching
        # ============================================================
        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())
        a_idx = rand_idx[:, num_dst_orig:, :]  # Src 후보
        b_idx = rand_idx[:, :num_dst_orig, :]  # Dst

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        def split(x):
            C_ = x.shape[-1]
            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C_))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C_))
                protected = gather(x, dim=1, index=protected_idx.expand(B, num_protected_actual, C_))
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C_))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C_))
                return src, dst

        metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, _ = split(metric_norm)
        else:
            a, b = split(metric_norm)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]
            valid_count = valid_edges.shape[0]
            r_actual = min(r, valid_count)
            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r

        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual

        after_merge = N - r_actual
        per_frame_after = after_merge // num_imgs
        if verbose:
            print(f"[Token-QT-PerFrame] merge 전: {N:,}개 ({tokens_per_img}개/프레임) | "
                  f"GA: {num_protected:,} | Dst: {num_dst:,} | Src제거: {r_actual:,} | "
                  f"merge 후: {after_merge:,}개 ({per_frame_after}개/프레임) | "
                  f"압축률: {r_actual/N*100:.1f}%")

    def merge(x, mode="mean", extra_tensors=None, extra_tensors_2=None):
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))
            dst_e = dst_e.scatter_reduce(-2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode)
            if enable_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)
            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))
            dst_e2 = dst_e2.scatter_reduce(-2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode)
            if enable_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        if enable_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        return main_result

    def unmerge(x):
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len:unm_len + dst_len, :]
        if enable_protection:
            protected = x[..., unm_len + dst_len:unm_len + dst_len + num_protected_actual, :]
        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, src_len, c),
            src=src,
        )
        if enable_protection:
            out.scatter_(dim=-2, index=protected_idx.expand(B, num_protected_actual, c), src=protected)
        return out

    return merge, unmerge


def _find_dst_per_node(
    node_ids: torch.Tensor,    # [num_patches] - 각 패치의 노드 ID
    scores: torch.Tensor,      # [num_patches] - 각 패치의 info_map score
    seq_indices: torch.Tensor, # [num_patches] - 각 패치의 전체 시퀀스 인덱스
    num_nodes: int,
) -> torch.Tensor:
    """
    각 노드별로 score 최소 패치의 seq_idx를 반환.
    벡터화된 구현.
    
    Returns:
        dst_seq_indices: [num_nodes] - 각 노드의 Dst 패치의 시퀀스 인덱스
    """
    device = node_ids.device
    
    # 각 노드의 최소 score 찾기 (scatter_reduce)
    INF = torch.tensor(float('inf'), device=device, dtype=scores.dtype)
    min_scores_per_node = torch.full((num_nodes,), float('inf'), device=device, dtype=scores.dtype)
    min_scores_per_node = min_scores_per_node.scatter_reduce(
        0, node_ids, scores, reduce="amin", include_self=True
    )

    # 각 패치가 자기 노드의 최소값과 일치하는지 마스크
    # min_scores_per_node[node_ids[i]] == scores[i]면 그 패치가 노드의 최소
    is_node_min = scores == min_scores_per_node[node_ids]
    
    # 같은 노드에 여러 패치가 최소값을 가질 수 있음 (tie). 첫 번째만 선택.
    # seq_indices 작은 것 우선 → 작은 seq_idx에 INF 큰 값을 부여하면 안 됨, 작은 값 부여
    # 트릭: 마스크 안 맞으면 INF, 맞으면 seq_idx → 노드별 min(seq_idx)이 첫 dst
    
    tiebreaker = torch.where(
        is_node_min,
        seq_indices.float(),
        torch.full_like(seq_indices, fill_value=2**30, dtype=torch.float)
    )
    
    dst_seq_indices = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
    dst_seq_indices_float = torch.full((num_nodes,), float(2**30), device=device, dtype=torch.float)
    dst_seq_indices_float = dst_seq_indices_float.scatter_reduce(
        0, node_ids, tiebreaker, reduce="amin", include_self=True
    )
    dst_seq_indices = dst_seq_indices_float.long()
    
    return dst_seq_indices