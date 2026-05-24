"""
STTM (Spatio-Temporal Token Merging) for LiteVGGT

기존 LiteVGGT의 token_merge_bipartite2d를 대체하는 STTM 기반 merge 함수.

동작 방식:
  1. 공간 merge: quadtree_build_video()로 복잡도 기반 adaptive merge
     - 단순 영역 → 큰 토큰 (많이 압축)
     - 복잡 영역 → 작은 토큰 (적게 압축)
  2. 시간 merge: cross_frame_node_merging_fast()로 인접 프레임 중복 제거
     - 인접 프레임 간 유사도 높은 토큰 → 하나로 합침

B 방식: 기존 merge → attention → unmerge 구조 유지
        Dst 선택 방식만 STTM으로 교체
"""

import torch
import torch.nn.functional as F
import math
import einops
from typing import Tuple, Callable, Optional, Union


# =====================================================================
# quadtree_spatial_merger.py 에서 필요한 함수들 (그대로 가져옴)
# =====================================================================
_stride = 2

def avgpool_to_even_side_feature(video_feature):
    T, C, ori_h, ori_w = video_feature.shape
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    device = video_feature.device
    even_height_flag, even_width_flag = ori_h % _stride == 0, ori_w % _stride == 0
    if even_height_flag and even_width_flag:
        return F.avg_pool2d(video_feature, (_stride, _stride))
    else:
        video_feature_next_level = torch.empty((T, C, new_h, new_w), device=device, dtype=video_feature.dtype)
        if not even_height_flag and even_width_flag:
            video_feature_next_level[:, :, 0, :] = video_feature[:, :, 0, :].reshape(T, C, new_w, _stride).mean(dim=-1)
            video_feature_next_level[:, :, 1:, :] = F.avg_pool2d(video_feature[:, :, 1:, :], (_stride, _stride))
        elif even_height_flag and not even_width_flag:
            video_feature_next_level[:, :, :, 0] = video_feature[:, :, :, 0].reshape(T, C, new_h, _stride).mean(dim=-1)
            video_feature_next_level[:, :, :, 1:] = F.avg_pool2d(video_feature[:, :, :, 1:], (_stride, _stride))
        else:
            video_feature_next_level[:, :, 0, 0] = video_feature[:, :, 0, 0]
            video_feature_next_level[:, :, 0, 1:] = video_feature[:, :, 0, 1:].reshape(T, C, new_w-1, _stride).mean(dim=-1)
            video_feature_next_level[:, :, 1:, 0] = video_feature[:, :, 1:, 0].reshape(T, C, new_h-1, _stride).mean(dim=-1)
            video_feature_next_level[:, :, 1:, 1:] = F.avg_pool2d(video_feature[:, :, 1:, 1:], (_stride, _stride))
        return video_feature_next_level


def pool_to_even_side_index_video(video_feature, prev_tyxyx_tlbr):
    T, ori_h, ori_w, _ = video_feature.shape
    device = video_feature.device
    new_h, new_w = math.ceil(ori_h/2), math.ceil(ori_w/2)
    even_height_flag, even_width_flag = ori_h % _stride == 0, ori_w % _stride == 0

    if even_height_flag and even_width_flag:
        grid_t = torch.arange(0, T, device=device, dtype=torch.int32)
        grid_y = torch.arange(0, new_h, device=device, dtype=torch.int32) * 2
        grid_x = torch.arange(0, new_w, device=device, dtype=torch.int32) * 2
        gt, gy, gx = torch.meshgrid(grid_t, grid_y, grid_x, indexing='ij')
        top_left = torch.stack([gt, gy, gx], dim=-1).unsqueeze(-2)
        offsets = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1]], device=device, dtype=torch.int32).view(1,1,1,4,3)
        child_tyx_coords = top_left + offsets
        child_tyxyx_tlbr = torch.zeros((T, new_h, new_w, 5), device=device, dtype=torch.int32)
        child_tyxyx_tlbr[:, :, :, 0]   = prev_tyxyx_tlbr[:, 0::2, 0::2, 0]
        child_tyxyx_tlbr[:, :, :, 1:3] = prev_tyxyx_tlbr[:, 0::2, 0::2, 1:3]
        child_tyxyx_tlbr[:, :, :, 3:5] = prev_tyxyx_tlbr[:, 1::2, 1::2, 3:5]
        child_valid_mask = torch.ones((T, new_h, new_w, 4), device=device, dtype=torch.bool)
    else:
        # odd size 처리 (원본 quadtree_spatial_merger.py 그대로)
        child_yx_coords  = torch.zeros((new_h, new_w, 4, 2), device=device, dtype=torch.int32)
        child_valid_mask = torch.zeros((new_h, new_w, 4), device=device, dtype=torch.bool)
        child_tyxyx_tlbr = torch.zeros((T, new_h, new_w, 5), device=device, dtype=torch.int32)

        if not even_height_flag and even_width_flag:
            h_start = 1 + 2 * torch.arange(0, new_h-1, device=device, dtype=torch.int32)
            w_start = 2 * torch.arange(0, new_w, device=device, dtype=torch.int32)
            # (a) h=0: 1x2 block
            child_yx_coords[0, :, [0,1], 1] = torch.stack([w_start, w_start+1], dim=-1)
            child_valid_mask[0, :, [0,1]] = True
            child_tyxyx_tlbr[:, 0, :, 0]   = prev_tyxyx_tlbr[:, 0, 0::2, 0]
            child_tyxyx_tlbr[:, 0, :, 1:3] = prev_tyxyx_tlbr[:, 0, 0::2, 1:3]
            child_tyxyx_tlbr[:, 0, :, 3:5] = prev_tyxyx_tlbr[:, 0, 1::2, 3:5]
            # (b) h>=1: 2x2 block
            child_yx_coords[1:, :, [0,1,2,3], 0] = torch.stack([*([h_start]*2), *([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w, 1)
            child_yx_coords[1:, :, [0,1,2,3], 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h-1, 1, 1)
            child_valid_mask[1:, :, [0,1,2,3]] = True
            child_tyxyx_tlbr[:, 1:, :, 0]   = prev_tyxyx_tlbr[:, 1::2, 0::2, 0]
            child_tyxyx_tlbr[:, 1:, :, 1:3] = prev_tyxyx_tlbr[:, 1::2, 0::2, 1:3]
            child_tyxyx_tlbr[:, 1:, :, 3:5] = prev_tyxyx_tlbr[:, 2::2, 1::2, 3:5]

        elif even_height_flag and not even_width_flag:
            h_start = 2 * torch.arange(0, new_h, device=device, dtype=torch.int32)
            w_start = 1 + 2 * torch.arange(0, new_w-1, device=device, dtype=torch.int32)
            # (a) w=0: 2x1 block
            child_yx_coords[:, 0, [0,2], 0] = torch.stack([h_start, h_start+1], dim=-1)
            child_valid_mask[:, 0, [0,2]] = True
            child_tyxyx_tlbr[:, :, 0, 0]   = prev_tyxyx_tlbr[:, 0::2, 0, 0]
            child_tyxyx_tlbr[:, :, 0, 1:3] = prev_tyxyx_tlbr[:, 0::2, 0, 1:3]
            child_tyxyx_tlbr[:, :, 0, 3:5] = prev_tyxyx_tlbr[:, 1::2, 0, 3:5]
            # (b) w>=1: 2x2 block
            child_yx_coords[:, 1:, [0,1,2,3], 0] = torch.stack([*([h_start]*2), *([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
            child_yx_coords[:, 1:, [0,1,2,3], 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h, 1, 1)
            child_valid_mask[:, 1:, [0,1,2,3]] = True
            child_tyxyx_tlbr[:, :, 1:, 0]   = prev_tyxyx_tlbr[:, 0::2, 1::2, 0]
            child_tyxyx_tlbr[:, :, 1:, 1:3] = prev_tyxyx_tlbr[:, 0::2, 1::2, 1:3]
            child_tyxyx_tlbr[:, :, 1:, 3:5] = prev_tyxyx_tlbr[:, 1::2, 2::2, 3:5]

        else:  # not even_height_flag and not even_width_flag
            h_start = 1 + 2 * torch.arange(0, new_h-1, device=device, dtype=torch.int32)
            w_start = 1 + 2 * torch.arange(0, new_w-1, device=device, dtype=torch.int32)
            # (a) top-left 1x1
            child_valid_mask[0, 0, 0] = True
            child_tyxyx_tlbr[:, 0, 0] = prev_tyxyx_tlbr[:, 0, 0]
            # (b) top-right 1x2
            child_yx_coords[0, 1:, [0,1], 1] = torch.stack([w_start, w_start+1], dim=-1)
            child_valid_mask[0, 1:, [0,1]] = True
            child_tyxyx_tlbr[:, 0, 1:, 0]   = prev_tyxyx_tlbr[:, 0, 1::2, 0]
            child_tyxyx_tlbr[:, 0, 1:, 1:3] = prev_tyxyx_tlbr[:, 0, 1::2, 1:3]
            child_tyxyx_tlbr[:, 0, 1:, 3:5] = prev_tyxyx_tlbr[:, 0, 2::2, 3:5]
            # (c) bottom-left 2x1
            child_yx_coords[1:, 0, [0,2], 0] = torch.stack([h_start, h_start+1], dim=-1)
            child_valid_mask[1:, 0, [0,2]] = True
            child_tyxyx_tlbr[:, 1:, 0, 0]   = prev_tyxyx_tlbr[:, 1::2, 0, 0]
            child_tyxyx_tlbr[:, 1:, 0, 1:3] = prev_tyxyx_tlbr[:, 1::2, 0, 1:3]
            child_tyxyx_tlbr[:, 1:, 0, 3:5] = prev_tyxyx_tlbr[:, 2::2, 0, 3:5]
            # (d) bottom-right 2x2
            child_yx_coords[1:, 1:, [0,1,2,3], 0] = torch.stack([*([h_start]*2), *([h_start+1]*2)], dim=-1).unsqueeze(1).repeat(1, new_w-1, 1)
            child_yx_coords[1:, 1:, [0,1,2,3], 1] = torch.stack([w_start, w_start+1], dim=-1).repeat(1, 2).unsqueeze(0).repeat(new_h-1, 1, 1)
            child_valid_mask[1:, 1:, [0,1,2,3]] = True
            child_tyxyx_tlbr[:, 1:, 1:, 0]   = prev_tyxyx_tlbr[:, 1::2, 1::2, 0]
            child_tyxyx_tlbr[:, 1:, 1:, 1:3] = prev_tyxyx_tlbr[:, 1::2, 1::2, 1:3]
            child_tyxyx_tlbr[:, 1:, 1:, 3:5] = prev_tyxyx_tlbr[:, 2::2, 2::2, 3:5]

        # temporal 축 확장
        t_idx = torch.arange(0, T, device=device, dtype=torch.int32)
        t_coords = t_idx.reshape(-1,1,1,1,1).repeat(1, new_h, new_w, 4, 1)
        _child_yx_coords = child_yx_coords.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        child_tyx_coords = torch.cat([t_coords, _child_yx_coords], dim=-1)
        child_valid_mask = child_valid_mask.unsqueeze(0).repeat(T, 1, 1, 1)

    return child_tyx_coords, child_valid_mask, child_tyxyx_tlbr


# =====================================================================
# quadtree_temporal_merger.py 에서 필요한 함수들 (그대로 가져옴)
# =====================================================================
def get_cross_frame_node_pairs_fast(quadtree_tyxyx_tlbr):
    device = quadtree_tyxyx_tlbr.device
    N_node = quadtree_tyxyx_tlbr.size(0)
    new_frame_idx_list = (torch.nonzero(quadtree_tyxyx_tlbr[0:-1, 0] != quadtree_tyxyx_tlbr[1:, 0]).squeeze(1)+1).tolist()
    new_frame_idx_list = [0, *new_frame_idx_list, len(quadtree_tyxyx_tlbr)]
    new_frame_idx = torch.tensor(new_frame_idx_list, device=device, dtype=torch.int32)
    num_nodes_per_frame = new_frame_idx[1:] - new_frame_idx[:-1]
    max_num_node = num_nodes_per_frame.max().item()
    N_frame = len(new_frame_idx_list) - 1
    node_idx = torch.arange(N_node, device=device, dtype=torch.int32)
    frame_ids = torch.bucketize(node_idx, new_frame_idx[1:-1], out_int32=True, right=True)
    frame_starts = new_frame_idx[frame_ids]
    local_idx = node_idx - frame_starts
    src2tgt_index = local_idx + frame_ids * max_num_node
    tlbr_padded = torch.zeros(N_frame*max_num_node, 4, device=device, dtype=torch.int32)
    valid_mask_padded = torch.zeros(N_frame*max_num_node, device=device, dtype=torch.int32)
    tlbr_padded.index_add_(0, src2tgt_index, quadtree_tyxyx_tlbr[:, 1:])
    valid_mask_padded.index_add_(0, src2tgt_index, torch.ones(N_node, device=device, dtype=torch.int32))
    tlbr_padded = tlbr_padded.reshape(N_frame, max_num_node, 4)
    valid_mask_padded = (valid_mask_padded > 0).reshape(N_frame, max_num_node)
    cur_nodes_tlbr, nxt_nodes_tlbr = tlbr_padded[:-1], tlbr_padded[1:]
    cur_valid_mask, nxt_valid_mask = valid_mask_padded[:-1], valid_mask_padded[1:]
    diff = cur_nodes_tlbr.unsqueeze(2) - nxt_nodes_tlbr.unsqueeze(1)
    cur_contain_nxt = ((diff[..., :2] <= 0).all(dim=-1)) & ((diff[..., 2:] >= 0).all(dim=-1))
    nxt_contain_cur = ((diff[..., :2] >= 0).all(dim=-1)) & ((diff[..., 2:] <= 0).all(dim=-1))
    valid_node_mask = cur_valid_mask.unsqueeze(2) & nxt_valid_mask.unsqueeze(1)
    pair_mask = (cur_contain_nxt | nxt_contain_cur) & valid_node_mask
    pair_indices = torch.nonzero(pair_mask)
    b_idx, cur_idx, nxt_idx = pair_indices.T
    cur_idx_offset = new_frame_idx[:-2]
    nxt_idx_offset = new_frame_idx[1:-1]
    cur_idx = cur_idx + cur_idx_offset[b_idx]
    nxt_idx = nxt_idx + nxt_idx_offset[b_idx]
    return torch.stack([cur_idx, nxt_idx], dim=1)


def filter_cross_frame_node_pairs(quadtree_features, pair_idxs, temporal_thresh):
    feat_f32 = quadtree_features.float()
    feat_norm = feat_f32 / (feat_f32.norm(dim=-1, keepdim=True) + 1e-8)
    pair_sim = (feat_norm[pair_idxs[:, 0]] * feat_norm[pair_idxs[:, 1]]).sum(dim=-1)
    return pair_idxs[pair_sim >= temporal_thresh]


def get_merge_dst_idx_safe(pair_idxs_for_merging, N_node):
    device = pair_idxs_for_merging.device
    final_representative = torch.arange(N_node, device=device, dtype=torch.int32)
    while True:
        dst_idx = pair_idxs_for_merging[:, 0]
        src_idx = pair_idxs_for_merging[:, 1]
        dst_rep = final_representative[dst_idx]
        src_rep = final_representative[src_idx]
        min_rep = torch.minimum(src_rep, dst_rep)
        final_representative.scatter_reduce_(0, dst_idx, min_rep, reduce="amin")
        final_representative.scatter_reduce_(0, src_idx, min_rep, reduce="amin")
        final_representative = final_representative[final_representative]
        if torch.equal(final_representative, final_representative[final_representative]):
            break
    return final_representative


# =====================================================================
# STTM merge 함수 (B 방식: merge/unmerge 구조 유지)
# =====================================================================
def token_merge_sttm(
    metric: torch.Tensor,      # [B, N, C] 전체 토큰
    w: int,                    # 패치 그리드 가로
    h: int,                    # 패치 그리드 세로
    r: int,                    # 제거할 토큰 수 (현재 미사용, STTM이 자동 결정)
    spatial_thresh: float = 0.8,   # 공간 merge 유사도 임계값
    temporal_thresh: float = 0.6,  # 시간 merge 유사도 임계값
    root_level: int = 0,           # quadtree 시작 레벨
    tokens_per_img: int = None,    # 프레임당 토큰 수 (특수토큰 포함)
    num_imgs: int = None,          # 프레임 수
    verbose: bool = True,
    cached_tlbr=None,              # 캐시된 tyxyx_tlbr (None이면 새로 계산)
):
    """
    STTM 기반 token merge/unmerge 함수.
    기존 token_merge_bipartite2d와 동일한 인터페이스 (merge, unmerge 반환).

    Args:
        metric: [B, N, C] - Global Attention 입력 토큰
        w, h: 패치 그리드 크기
        spatial_thresh: 공간 merge 임계값 (높을수록 덜 merge)
        temporal_thresh: 시간 merge 임계값 (높을수록 덜 merge, -1이면 시간 merge 비활성화)
        tokens_per_img: 프레임당 토큰 수 (특수토큰 5개 포함)
        num_imgs: 프레임 수
    """
    B, N, C = metric.shape
    device = metric.device

    if tokens_per_img is None:
        tokens_per_img = w * h + 5
    if num_imgs is None:
        num_imgs = N // tokens_per_img

    # -----------------------------------------------
    # Step 1: 특수 토큰(5개) 분리
    # -----------------------------------------------
    special_tokens = metric[:, :5 * num_imgs, :]     # 특수 토큰 [B, 5*T, C]
    patch_tokens   = metric[:, 5 * num_imgs:, :]     # 패치 토큰 [B, T*H*W, C]

    # [B, T*H*W, C] → [T, C, H, W] (STTM 입력 형식)
    patch_tokens_thwc = patch_tokens[0].view(num_imgs, h, w, C)       # [T, H, W, C]
    patch_tokens_tchw = patch_tokens_thwc.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

    # -----------------------------------------------
    # Step 2: STTM 공간 + 시간 merge 실행
    # -----------------------------------------------
    with torch.no_grad():
        if cached_tlbr is not None:
            # -----------------------------------------------
            # 캐시된 tyxyx_tlbr 재사용 (cal_merge=False인 레이어)
            # feature로 top-left 토큰만 대표로 사용
            # -----------------------------------------------
            tyxyx_tlbr = cached_tlbr
            t_c = tyxyx_tlbr[:, 0].long()
            y_c = tyxyx_tlbr[:, 1].long()
            x_c = tyxyx_tlbr[:, 2].long()
            flat_idx = t_c * (h * w) + y_c * w + x_c
            sttm_features = patch_tokens_tchw.permute(0,2,3,1).contiguous().view(-1, C)[flat_idx]
            new_tlbr = None   # 캐시 업데이트 불필요
        else:
            # -----------------------------------------------
            # STTM 새로 계산 (cal_merge=True인 레이어)
            # -----------------------------------------------
            sttm_features, num_patches_per_node, tyxyx_tlbr = _sttm_build(
                patch_tokens_tchw,
                spatial_thresh=spatial_thresh,
                temporal_thresh=temporal_thresh,
                root_level=root_level,
            )
            new_tlbr = tyxyx_tlbr   # 캐시 저장용

    N_merged = sttm_features.shape[0]

    if verbose:
        print(f"\n[STTM] merge 전: {N}개 | 패치: {num_imgs*h*w}개 | "
              f"merge 후: {N_merged + 5*num_imgs}개 | "
              f"압축률: {(1 - N_merged/(num_imgs*h*w))*100:.1f}%"
              f"{'' if cached_tlbr is None else ' (캐시 재사용)'}")

    # -----------------------------------------------
    # Step 3: merge 함수 정의
    # (특수토큰 유지 + STTM merge된 패치토큰)
    # -----------------------------------------------
    # 미리 계산: 각 노드의 좌표
    t_idx_pre = tyxyx_tlbr[:, 0].long()
    y_tl_pre  = tyxyx_tlbr[:, 1].long()
    x_tl_pre  = tyxyx_tlbr[:, 2].long()
    y_br_pre  = tyxyx_tlbr[:, 3].long()
    x_br_pre  = tyxyx_tlbr[:, 4].long()

    def _merge_single(x: torch.Tensor):
        special = x[:, :5 * num_imgs, :]
        patches = x[:, 5 * num_imgs:, :]
        C_x = patches.shape[-1]
        patches_thwc = patches[0].view(num_imgs, h, w, C_x)
        merged_patches = torch.zeros(N_merged, C_x, device=device, dtype=x.dtype)
        for i in range(N_merged):
            t  = t_idx_pre[i].item()
            yt = y_tl_pre[i].item()
            yb = y_br_pre[i].item()
            xt = x_tl_pre[i].item()
            xb = x_br_pre[i].item()
            region = patches_thwc[t, yt:yb, xt:xb, :]
            merged_patches[i] = region.mean(dim=(0, 1))
        return torch.cat([special, merged_patches.unsqueeze(0)], dim=1)

    def merge(x: torch.Tensor, mode: str = "mean", extra_tensors=None, extra_tensors_2=None):
        # q, k, v 모두 동일한 STTM merge 적용
        merged = _merge_single(x)
        if extra_tensors is not None and extra_tensors_2 is not None:
            return merged, _merge_single(extra_tensors), _merge_single(extra_tensors_2)
        elif extra_tensors is not None:
            return merged, _merge_single(extra_tensors)
        return merged

    # -----------------------------------------------
    # Step 4: unmerge 함수 정의
    # (각 노드의 값을 원래 위치로 복원)
    # -----------------------------------------------
    def unmerge(x: torch.Tensor):
        B_x      = x.shape[0]
        special  = x[:, :5 * num_imgs, :]   # [B, 5T, C_out]
        merged_p = x[:, 5 * num_imgs:, :]   # [B, N_merged, C_out]
        C_out    = merged_p.shape[-1]

        restored = torch.zeros(B_x, num_imgs * h * w, C_out, device=device, dtype=x.dtype)

        for i in range(N_merged):
            t  = t_idx_pre[i].item()
            yt = y_tl_pre[i].item()
            yb = y_br_pre[i].item()
            xt = x_tl_pre[i].item()
            xb = x_br_pre[i].item()
            for ry in range(yt, yb):
                for rx in range(xt, xb):
                    flat_idx = t * h * w + ry * w + rx
                    restored[:, flat_idx, :] = merged_p[:, i, :]

        out = torch.cat([special, restored], dim=1)
        return out

    return merge, unmerge, new_tlbr, N_merged + 5 * num_imgs


def _sttm_build(
    video_feature: torch.Tensor,   # [T, C, H, W]
    spatial_thresh: float = 0.8,
    temporal_thresh: float = 0.6,
    root_level: int = 0,
):
    """
    quadtree_build_video의 핵심 로직.
    공간 merge → 시간 merge 순서로 실행.
    """
    device = video_feature.device
    T, C, H, W = video_feature.shape

    # 다단계 피라미드 생성
    size_per_level = [(H, W)]
    h_tmp, w_tmp = H, W
    while h_tmp != 2 and w_tmp != 2:
        w_tmp = math.ceil(w_tmp / 2)
        h_tmp = math.ceil(h_tmp / 2)
        size_per_level.insert(0, (h_tmp, w_tmp))

    video_features_per_level = [video_feature]
    _feat = video_feature
    while video_features_per_level[0].size(-1) != size_per_level[root_level][1]:
        _feat = avgpool_to_even_side_feature(_feat)
        video_features_per_level.insert(0, _feat)

    n_level = len(video_features_per_level)
    for i in range(n_level):
        video_features_per_level[i] = video_features_per_level[i].permute(0, 2, 3, 1)  # [T, H, W, C]

    # node 메타데이터 초기화
    T_f, H_f, W_f, _ = video_features_per_level[-1].shape
    grid_t = torch.arange(T_f, device=device, dtype=torch.int32)
    grid_y = torch.arange(H_f, device=device, dtype=torch.int32)
    grid_x = torch.arange(W_f, device=device, dtype=torch.int32)
    gt, gy, gx = torch.meshgrid(grid_t, grid_y, grid_x, indexing='ij')
    node_tyxyx_tlbr_nxt = torch.stack([gt, gy, gx, gy+1, gx+1], dim=-1)  # [T, H, W, 5]

    node_tyxyx_tlbr_per_level = [node_tyxyx_tlbr_nxt]
    child_tyx_coords_per_level, child_valid_mask_per_level = [], []
    for i in range(n_level - 1):
        tgt_lvl = n_level - 1 - i
        child_tyx, child_mask, node_tyxyx_tlbr_nxt = pool_to_even_side_index_video(
            video_features_per_level[tgt_lvl], node_tyxyx_tlbr_nxt
        )
        node_tyxyx_tlbr_per_level.insert(0, node_tyxyx_tlbr_nxt)
        child_tyx_coords_per_level.insert(0, child_tyx)
        child_valid_mask_per_level.insert(0, child_mask)

    # quadtree 반복 빌드
    T0, H0, W0, _ = video_features_per_level[0].shape
    grid_t0 = torch.arange(T0, device=device, dtype=torch.int32)
    grid_y0 = torch.arange(H0, device=device, dtype=torch.int32)
    grid_x0 = torch.arange(W0, device=device, dtype=torch.int32)
    gt0, gy0, gx0 = torch.meshgrid(grid_t0, grid_y0, grid_x0, indexing='ij')
    parent_tyx_coords_3d = torch.stack([gt0, gy0, gx0], dim=-1).flatten(0, 2)

    quadtree_features_list, quadtree_tyxyx_tlbr_list = [], []
    for curr_lvl in range(n_level):
        parent_tyx_coords_3d = _quadtree_iteration(
            parent_tyx_coords_3d,
            video_features_per_level, node_tyxyx_tlbr_per_level,
            child_tyx_coords_per_level, child_valid_mask_per_level,
            quadtree_features_list, quadtree_tyxyx_tlbr_list,
            curr_lvl, n_level, spatial_thresh,
        )

    quadtree_features = torch.cat(quadtree_features_list)      # [N_merged, C]
    quadtree_tyxyx_tlbr = torch.cat(quadtree_tyxyx_tlbr_list)  # [N_merged, 5]

    # 정렬
    tyx_offsets = torch.tensor([H_f*W_f, W_f, 1], device=device, dtype=torch.int32)
    quadtree_1d_index = (quadtree_tyxyx_tlbr[:, :3] * tyx_offsets.unsqueeze(0)).sum(dim=-1)
    sorted_idx = torch.argsort(quadtree_1d_index)
    quadtree_features = quadtree_features[sorted_idx]
    quadtree_tyxyx_tlbr = quadtree_tyxyx_tlbr[sorted_idx]

    # 노드당 패치 수
    node_h = quadtree_tyxyx_tlbr[:, 3] - quadtree_tyxyx_tlbr[:, 1]
    node_w = quadtree_tyxyx_tlbr[:, 4] - quadtree_tyxyx_tlbr[:, 2]
    num_patches_per_node = node_h * node_w

    # 시간 merge
    if temporal_thresh > 0 and T > 1:
        pair_idxs = get_cross_frame_node_pairs_fast(quadtree_tyxyx_tlbr)
        if len(pair_idxs) > 0:
            pair_idxs_filtered = filter_cross_frame_node_pairs(quadtree_features, pair_idxs, temporal_thresh)
            if len(pair_idxs_filtered) > 0:
                final_rep = get_merge_dst_idx_safe(pair_idxs_filtered, quadtree_features.shape[0])
                # 대표 토큰만 선택
                survived_mask = final_rep == torch.arange(len(final_rep), device=device, dtype=torch.int32)
                quadtree_features = quadtree_features[survived_mask]
                quadtree_tyxyx_tlbr = quadtree_tyxyx_tlbr[survived_mask]
                num_patches_per_node = num_patches_per_node[survived_mask]

    return quadtree_features, num_patches_per_node, quadtree_tyxyx_tlbr


def _quadtree_iteration(
    parent_tyx_coords_3d,
    video_features_per_level, node_tyxyx_tlbr_per_level,
    child_tyx_coords_per_level, child_valid_mask_per_level,
    quadtree_features_list, quadtree_tyxyx_tlbr_list,
    curr_lvl, n_level, threshold,
):
    if curr_lvl == n_level - 1:
        last_features = video_features_per_level[curr_lvl]
        last_tlbr = node_tyxyx_tlbr_per_level[curr_lvl]
        p_t, p_y, p_x = parent_tyx_coords_3d.T
        quadtree_features_list.append(last_features[p_t, p_y, p_x])
        quadtree_tyxyx_tlbr_list.append(last_tlbr[p_t, p_y, p_x])
        return None

    parent_features = video_features_per_level[curr_lvl]
    parent_tlbr = node_tyxyx_tlbr_per_level[curr_lvl]
    p_t, p_y, p_x = parent_tyx_coords_3d.T
    tgt_parent_features = parent_features[p_t, p_y, p_x]
    tgt_parent_tlbr = parent_tlbr[p_t, p_y, p_x]

    child_features = video_features_per_level[curr_lvl + 1]
    child_tyx_coords = child_tyx_coords_per_level[curr_lvl]
    child_valid_mask = child_valid_mask_per_level[curr_lvl]
    tgt_child_tyx_coords = child_tyx_coords[p_t, p_y, p_x]
    tgt_child_valid_mask = child_valid_mask[p_t, p_y, p_x]
    tgt_child_tyx_coords_3d = tgt_child_tyx_coords.flatten(0, 1)
    c_t, c_y, c_x = tgt_child_tyx_coords_3d.T
    tgt_child_features = einops.rearrange(child_features[c_t, c_y, c_x], "(n s) c -> n s c", s=4)

    sim = F.cosine_similarity(
        tgt_parent_features.unsqueeze(1).float(),
        tgt_child_features.float(), dim=-1
    )
    stop_mask = (sim >= threshold).all(dim=-1)
    split_node_mask = torch.logical_and((~stop_mask).unsqueeze(1), tgt_child_valid_mask).flatten(0, 1)
    parent_tyx_coords_3d = tgt_child_tyx_coords_3d[split_node_mask]

    quadtree_features_list.append(tgt_parent_features[stop_mask])
    quadtree_tyxyx_tlbr_list.append(tgt_parent_tlbr[stop_mask])

    return parent_tyx_coords_3d