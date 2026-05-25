# merging/sttm_bipartite_merge.py
"""
Debug + chunked version: Per-frame Quadtree + ToMe-style global bipartite merging.

Fixes:
  - Boundary-safe quadtree split.
  - Invalid dst/protected index filtering.
  - Automatic debug log saving.
  - Chunked q/k/v merge to avoid OOM during gather/scatter.

Env:
  QT_DEBUG=1
  QT_DEBUG_LOG=./logs/qt_debug.log
  QT_MERGE_CHUNK=32768
"""

import os
import time
import traceback
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from merging.complexity import get_dynamic_protect_ratio_single


_node_size_stats = {
    "counter": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 8: 0, 16: 0},
    "total_nodes": 0,
    "total_patches": 0,
}
_log_path = None


def _debug_enabled(verbose=True):
    return verbose or os.environ.get("QT_DEBUG", "0") == "1"


def _get_log_path():
    global _log_path
    if _log_path is not None:
        return _log_path

    env_path = os.environ.get("QT_DEBUG_LOG")
    if env_path:
        _log_path = env_path
    else:
        os.makedirs("debug_logs", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        _log_path = os.path.join(
            "debug_logs",
            f"qt_bipartite_debug_{stamp}_pid{os.getpid()}.log",
        )

    log_dir = os.path.dirname(_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    return _log_path


def _log(msg="", force=True):
    if not force:
        return
    text = str(msg)
    print(text, flush=True)
    with open(_get_log_path(), "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()


def _section(title, force=True):
    _log("\n" + "=" * 90, force)
    _log(title, force)
    _log("=" * 90, force)


def _cuda_log(tag: str, device=None, force=True):
    if not force or not torch.cuda.is_available():
        return

    if device is None:
        device = torch.cuda.current_device()

    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass

    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**3

    try:
        free, total = torch.cuda.mem_get_info(device)
        free = free / 1024**3
        total = total / 1024**3
        _log(
            f"[CUDA][{tag}] alloc={alloc:.3f}GB | reserved={reserved:.3f}GB | "
            f"max_alloc={max_alloc:.3f}GB | free={free:.3f}GB / total={total:.3f}GB",
            force,
        )
    except Exception:
        _log(
            f"[CUDA][{tag}] alloc={alloc:.3f}GB | reserved={reserved:.3f}GB | "
            f"max_alloc={max_alloc:.3f}GB",
            force,
        )


def _tensor_log(name: str, x: torch.Tensor, force=True):
    if not force:
        return
    mb = x.numel() * x.element_size() / 1024**2
    _log(
        f"[Tensor] {name}: shape={tuple(x.shape)} | dtype={x.dtype} | approx={mb:.2f}MB",
        force,
    )


def reset_node_stats():
    global _node_size_stats
    _node_size_stats = {
        "counter": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 8: 0, 16: 0},
        "total_nodes": 0,
        "total_patches": 0,
    }


def get_node_stats():
    return {
        "counter": dict(_node_size_stats["counter"]),
        "total_nodes": _node_size_stats["total_nodes"],
        "total_patches": _node_size_stats["total_patches"],
    }


def _update_stats(size, num_patches, n_img=1):
    size = int(size)
    if size not in _node_size_stats["counter"]:
        _node_size_stats["counter"][size] = 0
    _node_size_stats["counter"][size] += n_img
    _node_size_stats["total_nodes"] += n_img
    _node_size_stats["total_patches"] += n_img * int(num_patches)


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    chunk_size: int,
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
        node_max[:, i:end_i] = chunk_max_bf16.to(original_dtype)
        node_idx[:, i:end_i] = chunk_idx

    return node_max, node_idx


def do_nothing(x, extra_tensors=None, extra_tensors_2=None):
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    if extra_tensors is not None:
        return x, extra_tensors
    return x


def _check_homogeneous_single(region: torch.Tensor, thresh: float) -> bool:
    _, H, W = region.shape
    if H < 2 or W < 2:
        return True

    parent = F.normalize(region.mean(dim=(1, 2)).float(), dim=0)

    h_mid = H // 2
    w_mid = W // 2

    children = (
        region[:, :h_mid, :w_mid],
        region[:, :h_mid, w_mid:],
        region[:, h_mid:, :w_mid],
        region[:, h_mid:, w_mid:],
    )

    for child in children:
        if child.numel() == 0:
            continue
        child_mean = F.normalize(child.mean(dim=(1, 2)).float(), dim=0)
        if torch.dot(parent, child_mean).item() < thresh:
            return False

    return True


def _split_block(y0: int, x0: int, y1: int, x1: int):
    h = y1 - y0
    w = x1 - x0
    if h < 2 or w < 2:
        return []

    y_mid = y0 + h // 2
    x_mid = x0 + w // 2

    children = [
        (y0, x0, y_mid, x_mid),
        (y0, x_mid, y_mid, x1),
        (y_mid, x0, y1, x_mid),
        (y_mid, x_mid, y1, x1),
    ]

    return [
        (cy0, cx0, cy1, cx1)
        for cy0, cx0, cy1, cx1 in children
        if cy1 > cy0 and cx1 > cx0
    ]


def _node_size_from_bounds(y0: int, x0: int, y1: int, x1: int) -> int:
    return max(y1 - y0, x1 - x0)


def build_per_frame_quadtree(
    patch_features: torch.Tensor,
    spatial_thresh: float = 0.8,
    root_block_size: int = 8,
    min_block_size: int = 2,
) -> Tuple[torch.Tensor, int]:
    device = patch_features.device
    num_imgs, _, H, W = patch_features.shape

    patch_to_node = torch.full((num_imgs, H, W), -1, device=device, dtype=torch.long)
    next_node_id = 0

    for frame_idx in range(num_imgs):
        frame_features = patch_features[frame_idx]
        pending_blocks = []

        for y0 in range(0, H, root_block_size):
            for x0 in range(0, W, root_block_size):
                y1 = min(y0 + root_block_size, H)
                x1 = min(x0 + root_block_size, W)
                pending_blocks.append((y0, x0, y1, x1))

        while pending_blocks:
            y0, x0, y1, x1 = pending_blocks.pop()

            actual_h = y1 - y0
            actual_w = x1 - x0
            size = _node_size_from_bounds(y0, x0, y1, x1)

            if actual_h <= 0 or actual_w <= 0:
                continue

            if size <= min_block_size or actual_h < 2 or actual_w < 2:
                patch_to_node[frame_idx, y0:y1, x0:x1] = next_node_id
                _update_stats(size, actual_h * actual_w)
                next_node_id += 1
                continue

            region = frame_features[:, y0:y1, x0:x1]

            if _check_homogeneous_single(region, spatial_thresh):
                patch_to_node[frame_idx, y0:y1, x0:x1] = next_node_id
                _update_stats(size, actual_h * actual_w)
                next_node_id += 1
            else:
                children = _split_block(y0, x0, y1, x1)
                if not children:
                    patch_to_node[frame_idx, y0:y1, x0:x1] = next_node_id
                    _update_stats(size, actual_h * actual_w)
                    next_node_id += 1
                else:
                    pending_blocks.extend(children)

    return patch_to_node, next_node_id


def _find_dst_per_node(node_ids, scores, seq_indices, num_nodes):
    device = node_ids.device
    invalid_value = 2**30

    min_scores = torch.full((num_nodes,), float("inf"), device=device, dtype=scores.dtype)
    min_scores = min_scores.scatter_reduce(
        0,
        node_ids,
        scores,
        reduce="amin",
        include_self=True,
    )

    is_min = scores == min_scores[node_ids]

    tiebreaker = torch.where(
        is_min,
        seq_indices.float(),
        torch.full_like(seq_indices, fill_value=invalid_value, dtype=torch.float),
    )

    dst_float = torch.full((num_nodes,), float(invalid_value), device=device, dtype=torch.float)
    dst_float = dst_float.scatter_reduce(
        0,
        node_ids,
        tiebreaker,
        reduce="amin",
        include_self=True,
    )

    return dst_float.long()


def _validate_quadtree_mapping(patch_to_node, total_nodes, debug=False):
    flat = patch_to_node.flatten()

    invalid_neg = int((flat < 0).sum().item())
    invalid_hi = int((flat >= total_nodes).sum().item())

    counts = torch.zeros(total_nodes, device=flat.device, dtype=torch.long)
    valid = (flat >= 0) & (flat < total_nodes)

    if valid.any():
        counts.scatter_add_(0, flat[valid], torch.ones_like(flat[valid], dtype=torch.long))

    empty_nodes = int((counts == 0).sum().item())

    if debug:
        _log(
            f"[Validate] patch_to_node invalid_neg={invalid_neg} | "
            f"invalid_hi={invalid_hi} | empty_nodes={empty_nodes}"
        )

    return counts, invalid_neg, invalid_hi, empty_nodes


def token_merge_quadtree_bipartite(
    metric: torch.Tensor,
    patch_features: torch.Tensor,
    w: int,
    h: int,
    r: int,
    spatial_thresh: float = 0.8,
    root_block_size: int = 8,
    min_block_size: int = 2,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,
    use_dynamic_protect: bool = False,
    protect_ratio: float = 0.1,
    verbose: bool = True,
) -> Tuple[Callable, Callable]:
    B, N, _ = metric.shape
    device = metric.device
    debug = _debug_enabled(verbose)
    merge_chunk_size = int(os.environ.get("QT_MERGE_CHUNK", "32768"))

    if debug:
        _section("[QT-Bipartite DEBUG] enter token_merge_quadtree_bipartite")
        _log(f"[Log file] {_get_log_path()}")
        _log(f"metric={tuple(metric.shape)} dtype={metric.dtype}")
        _log(f"patch_features={tuple(patch_features.shape)} dtype={patch_features.dtype}")
        _log(
            f"w={w}, h={h}, r={r}, spatial_thresh={spatial_thresh}, "
            f"root_block={root_block_size}, min_block={min_block_size}, "
            f"merge_chunk={merge_chunk_size}"
        )
        _cuda_log("00_enter", device=device)
        _tensor_log("metric", metric)
        _tensor_log("patch_features", patch_features)

    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img

    assert tokens_per_img * num_imgs == N, "Token count does not match frame layout."

    try:
        with torch.no_grad():
            if debug:
                _log(f"[Layout] tokens_per_img={tokens_per_img}, num_imgs={num_imgs}, total_tokens={N}")

            if enable_protection:
                if info_map is not None:
                    info = info_map[:, 0].to(device)

                    if use_dynamic_protect:
                        protect_ratio = get_dynamic_protect_ratio_single(
                            info,
                            min_ratio=0.05,
                            max_ratio=0.20,
                            verbose=verbose,
                        )

                    num_patches = info.shape[-2] * info.shape[-1]
                    info_flat = info.flatten(1)
                    offsets = torch.arange(num_imgs, device=device) * tokens_per_img + 5

                    if torch.is_tensor(protect_ratio):
                        protect_ratios = protect_ratio.to(device).flatten().to(torch.float32)
                        if protect_ratios.numel() == 1:
                            protect_ratios = protect_ratios.expand(num_imgs)
                        elif protect_ratios.numel() != num_imgs:
                            repeats = (num_imgs + protect_ratios.numel() - 1) // protect_ratios.numel()
                            protect_ratios = protect_ratios.repeat(repeats)[:num_imgs]

                        protected_chunks = []
                        for img_idx in range(num_imgs):
                            ratio_i = float(protect_ratios[img_idx].clamp(0.0, 1.0).item())
                            k_i = max(1, int(num_patches * ratio_i))
                            topk_i = info_flat[img_idx].topk(k_i, dim=0).indices
                            protected_chunks.append(topk_i + offsets[img_idx])
                        protected_indices = torch.cat(protected_chunks, dim=0)
                    else:
                        ratio = max(0.0, min(float(protect_ratio), 1.0))
                        k = max(1, int(num_patches * ratio))
                        topk_idx = info_flat.topk(k, dim=1).indices
                        protected_indices = (topk_idx + offsets[:, None]).flatten()
                else:
                    num_protected = int(N * 0.1)
                    step = max(1, N // num_protected)
                    protected_indices = torch.arange(0, N, step, device=device)[:num_protected]

                valid_protected = (protected_indices >= 0) & (protected_indices < N)
                protected_indices = protected_indices[valid_protected]
                num_protected = protected_indices.numel()
            else:
                protected_indices = None
                num_protected = 0

            if debug:
                _log(f"[Step 1] protection | enable={enable_protection} | protected={num_protected}")
                if info_map is not None:
                    _tensor_log("info_map", info_map)
                _cuda_log("01_after_protection", device=device)

            reset_node_stats()
            patch_to_node, total_nodes = build_per_frame_quadtree(
                patch_features,
                spatial_thresh=spatial_thresh,
                root_block_size=root_block_size,
                min_block_size=min_block_size,
            )

            counts, invalid_neg, invalid_hi, empty_nodes = _validate_quadtree_mapping(
                patch_to_node,
                total_nodes,
                debug=debug,
            )

            if invalid_neg > 0 or invalid_hi > 0:
                raise RuntimeError(
                    f"Invalid patch_to_node mapping: invalid_neg={invalid_neg}, invalid_hi={invalid_hi}"
                )

            if debug:
                stats = get_node_stats()
                avg = stats["total_patches"] / max(stats["total_nodes"], 1)
                _log(
                    f"[Step 2] quadtree | total_nodes={total_nodes:,} | "
                    f"node_stats={stats['counter']} | avg_patch_per_node={avg:.2f}"
                )
                _tensor_log("patch_to_node", patch_to_node)
                _cuda_log("02_after_quadtree", device=device)

            flat_node_ids = patch_to_node.flatten()

            frame_offsets = torch.arange(num_imgs, device=device) * tokens_per_img + 5
            patch_local_idx = torch.arange(h * w, device=device)
            patch_seq_idx = (frame_offsets[:, None] + patch_local_idx[None, :]).flatten()

            if info_map is not None:
                score_flat = info_map[:, 0].flatten().float()
            else:
                score_flat = torch.rand(num_imgs * h * w, device=device, generator=generator)

            dst_seq_indices = _find_dst_per_node(
                flat_node_ids,
                score_flat,
                patch_seq_idx,
                total_nodes,
            )

            valid_dst_all = (dst_seq_indices >= 0) & (dst_seq_indices < N)
            invalid_dst_count = int((~valid_dst_all).sum().item())

            if debug:
                _log(
                    f"[Step 3] dst selection | dst_seq_indices={tuple(dst_seq_indices.shape)} | "
                    f"invalid_dst={invalid_dst_count}"
                )
                if invalid_dst_count > 0:
                    bad = dst_seq_indices[~valid_dst_all][:10].detach().cpu().tolist()
                    _log(f"[WARN] first invalid dst values: {bad}")
                _cuda_log("03_after_dst_selection", device=device)

            idx_buffer_seq = torch.zeros(N, device=device, dtype=torch.int64)

            idx_buffer_seq[:tokens_per_img] = -1

            if num_imgs > 1:
                cls_indices = torch.arange(1, num_imgs, device=device) * tokens_per_img
                cls_indices = cls_indices[:, None] + torch.arange(5, device=device)
                idx_buffer_seq[cls_indices.flatten()] = -1

                valid_non_first = (dst_seq_indices >= tokens_per_img) & (dst_seq_indices < N)
                non_first_dst = dst_seq_indices[valid_non_first]

                if debug:
                    skipped = int((~valid_dst_all).sum().item())
                    first_frame_dst = int(
                        ((dst_seq_indices >= 0) & (dst_seq_indices < tokens_per_img)).sum().item()
                    )
                    _log(
                        f"[Step 3b] apply dst | non_first_dst={non_first_dst.numel():,} | "
                        f"first_frame_dst={first_frame_dst:,} | skipped_invalid={skipped:,}"
                    )

                if non_first_dst.numel() > 0:
                    idx_buffer_seq[non_first_dst] = -1

            rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
            num_dst_orig = int((idx_buffer_seq == -1).sum().item())

            a_idx = rand_idx[:, num_dst_orig:, :]
            b_idx = rand_idx[:, :num_dst_orig, :]

            if enable_protection:
                protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
                num_protected_actual = protected_idx.shape[1]
            else:
                protected_idx = None
                num_protected_actual = 0

            num_src = a_idx.shape[1]
            num_dst = b_idx.shape[1]

            if debug:
                _log(
                    f"[Step 4] split index | num_src={num_src:,} | "
                    f"num_dst={num_dst:,} | protected_actual={num_protected_actual:,}"
                )
                _log(
                    f"[Cost] full_similarity_pairs={num_src * num_dst:,} | "
                    f"full_bf16_matrix={(num_src * num_dst * 2) / 1024**3:.3f}GB"
                )
                _cuda_log("04_after_index_split", device=device)

            def split(x):
                C = x.shape[-1]

                if enable_protection:
                    src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                    dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                    protected = gather(
                        x,
                        dim=1,
                        index=protected_idx.expand(B, num_protected_actual, C),
                    )
                    return src, dst, protected

                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

            metric_norm = metric / (metric.norm(dim=-1, keepdim=True) + 1e-8)

            if enable_protection:
                a, b, _ = split(metric_norm)
            else:
                a, b = split(metric_norm)

            if debug:
                _log(f"[Step 5] gather src/dst | src={tuple(a.shape)} | dst={tuple(b.shape)}")
                _tensor_log("a_src", a)
                _tensor_log("b_dst", b)
                _cuda_log("05_after_gather_src_dst", device=device)

            r = min(a.shape[1], r)
            num_src_actual = a.shape[1]
            chunk_size = min(5000, num_src_actual)

            b_transposed = b.transpose(-1, -2)

            if debug:
                sim_chunk_gb = (B * chunk_size * num_dst * 2) / 1024**3
                _log(
                    f"[Step 6] before similarity | chunk_size={chunk_size:,} | "
                    f"sim_chunk_bf16_est={sim_chunk_gb:.3f}GB"
                )
                _tensor_log("b_transposed", b_transposed)
                _cuda_log("06_before_fast_similarity", device=device)

            try:
                node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
            except torch.cuda.OutOfMemoryError:
                _section("[OOM CAUGHT] during fast_similarity_chunks")
                _log(
                    f"num_src_actual={num_src_actual:,}, num_dst={num_dst:,}, "
                    f"chunk_size={chunk_size:,}, C={a.shape[-1]}"
                )
                _log(
                    f"estimated_sim_chunk_bf16={(B * chunk_size * num_dst * 2) / 1024**3:.3f}GB"
                )
                _cuda_log("OOM_fast_similarity", device=device)
                raise

            if debug:
                _log("[Step 6] similarity done")
                _tensor_log("node_max", node_max)
                _tensor_log("node_idx", node_idx)
                _cuda_log("07_after_fast_similarity", device=device)

            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            if debug:
                _log(f"[Step 7] edge sort | edge_idx={tuple(edge_idx.shape)}")
                _cuda_log("08_after_edge_sort", device=device)

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

            after_merge = N - r_actual
            per_frame_after = after_merge // num_imgs

            if debug:
                _log(
                    f"[Step 8] merge plan | before={N:,} | after={after_merge:,} | "
                    f"per_frame_after={per_frame_after:,} | removed={r_actual:,} | "
                    f"compression={r_actual / N * 100:.2f}%"
                )
                _cuda_log("09_after_merge_plan", device=device)
                _log("[QT-Bipartite DEBUG] finished index calculation")

    except Exception as e:
        _section("[QT-Bipartite DEBUG] exception")
        _log(f"{type(e).__name__}: {e}")
        _log(traceback.format_exc())
        _cuda_log("exception_exit", device=device)
        raise

    def _gather_abs(x, abs_idx):
        n = x.shape[0]
        C = x.shape[-1]
        return gather(x, dim=1, index=abs_idx.expand(n, abs_idx.shape[1], C))

    def _merge_single_tensor_chunked(x, mode="mean"):
        n, _, C = x.shape

        dst = gather(x, dim=1, index=b_idx.expand(n, num_dst, C))

        src_len = src_idx.shape[1]

        if mode == "mean":
            dst_acc = dst.float()
            dst_count = torch.ones(n, num_dst, 1, device=x.device, dtype=torch.float32)

            for start in range(0, src_len, merge_chunk_size):
                end = min(start + merge_chunk_size, src_len)

                src_idx_chunk = src_idx[:, start:end, :]
                dst_idx_chunk = dst_idx[:, start:end, :]

                abs_src_idx = gather(
                    a_idx.expand(n, a_idx.shape[1], 1),
                    dim=1,
                    index=src_idx_chunk.expand(n, src_idx_chunk.shape[1], 1),
                )

                src_chunk = _gather_abs(x, abs_src_idx).float()
                dst_target = dst_idx_chunk.expand(n, dst_idx_chunk.shape[1], 1)

                dst_acc.scatter_add_(
                    1,
                    dst_target.expand(n, dst_target.shape[1], C),
                    src_chunk,
                )

                dst_count.scatter_add_(
                    1,
                    dst_target,
                    torch.ones(n, dst_target.shape[1], 1, device=x.device, dtype=torch.float32),
                )

                del abs_src_idx, src_chunk, dst_target

            dst = (dst_acc / dst_count.clamp_min(1.0)).to(x.dtype)

        elif mode == "sum":
            dst = dst.clone()

            for start in range(0, src_len, merge_chunk_size):
                end = min(start + merge_chunk_size, src_len)

                src_idx_chunk = src_idx[:, start:end, :]
                dst_idx_chunk = dst_idx[:, start:end, :]

                abs_src_idx = gather(
                    a_idx.expand(n, a_idx.shape[1], 1),
                    dim=1,
                    index=src_idx_chunk.expand(n, src_idx_chunk.shape[1], 1),
                )

                src_chunk = _gather_abs(x, abs_src_idx)
                dst_target = dst_idx_chunk.expand(n, dst_idx_chunk.shape[1], 1)

                dst.scatter_add_(
                    1,
                    dst_target.expand(n, dst_target.shape[1], C),
                    src_chunk,
                )

                del abs_src_idx, src_chunk, dst_target

        else:
            dst = dst.clone()

            for start in range(0, src_len, merge_chunk_size):
                end = min(start + merge_chunk_size, src_len)

                src_idx_chunk = src_idx[:, start:end, :]
                dst_idx_chunk = dst_idx[:, start:end, :]

                abs_src_idx = gather(
                    a_idx.expand(n, a_idx.shape[1], 1),
                    dim=1,
                    index=src_idx_chunk.expand(n, src_idx_chunk.shape[1], 1),
                )

                src_chunk = _gather_abs(x, abs_src_idx)
                dst_target = dst_idx_chunk.expand(n, dst_idx_chunk.shape[1], 1)

                dst.scatter_reduce_(
                    1,
                    dst_target.expand(n, dst_target.shape[1], C),
                    src_chunk,
                    reduce=mode,
                    include_self=True,
                )

                del abs_src_idx, src_chunk, dst_target

        unm_len = unm_idx.shape[1]
        abs_unm_idx = gather(
            a_idx.expand(n, a_idx.shape[1], 1),
            dim=1,
            index=unm_idx.expand(n, unm_len, 1),
        )
        unm = _gather_abs(x, abs_unm_idx)

        if enable_protection:
            protected = gather(
                x,
                dim=1,
                index=protected_idx.expand(n, num_protected_actual, C),
            )
            return torch.cat([unm, dst, protected], dim=1)

        return torch.cat([unm, dst], dim=1)

    def merge(x, mode="mean", extra_tensors=None, extra_tensors_2=None):
        if debug:
            _log(
                f"[Chunked merge] mode={mode} | chunk={merge_chunk_size:,} | "
                f"src_remove={src_idx.shape[1]:,} | unm={unm_idx.shape[1]:,} | dst={num_dst:,}"
            )
            _cuda_log("merge_enter", device=x.device)

        main_result = _merge_single_tensor_chunked(x, mode=mode)

        if extra_tensors is not None and extra_tensors_2 is not None:
            extra_1 = _merge_single_tensor_chunked(extra_tensors, mode=mode)
            extra_2 = _merge_single_tensor_chunked(extra_tensors_2, mode=mode)

            if debug:
                _cuda_log("merge_exit_qkv", device=x.device)

            return main_result, extra_1, extra_2

        if extra_tensors is not None:
            extra_1 = _merge_single_tensor_chunked(extra_tensors, mode=mode)

            if debug:
                _cuda_log("merge_exit_qk", device=x.device)

            return main_result, extra_1

        if debug:
            _cuda_log("merge_exit_q", device=x.device)

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
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1),
                dim=1,
                index=unm_idx,
            ).expand(B, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1),
                dim=1,
                index=src_idx,
            ).expand(B, src_len, c),
            src=src,
        )

        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge
