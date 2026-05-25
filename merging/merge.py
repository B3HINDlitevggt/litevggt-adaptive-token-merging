import torch
from typing import Tuple, Callable, Optional, Union
import torch.nn.functional as F
from merging.complexity import get_dynamic_protect_ratio_single

@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_bipartite2d_multi_batch(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,  # info_map [N,1,Hp,Wp]
    protect_ratio=0.1,
    protect_nms: bool = False,
    protect_aux_map=None,
    protect_aux_ratio: float = 0.0,
):

    B, N, C = metric.shape
    per_batch_ops = []

    if info_map is not None:
        img_per_batch = info_map.shape[0] // B 
        info_map = info_map.view(B, img_per_batch, *info_map.shape[1:]) 
    if protect_aux_map is not None:
        img_per_batch = protect_aux_map.shape[0] // B
        protect_aux_map = protect_aux_map.view(B, img_per_batch, *protect_aux_map.shape[1:])

    for b in range(B):
        metric_b = metric[b:b+1]  # shape [1, N, C]
        info_map_b = info_map[b:b+1].squeeze(0) if info_map is not None else None
        protect_aux_map_b = protect_aux_map[b:b+1].squeeze(0) if protect_aux_map is not None else None

        merge_b, unmerge_b = token_merge_bipartite2d(
            metric_b, w, h, sx, sy, r,
            no_rand=no_rand,
            generator=generator,
            enable_protection=enable_protection,
            info_map=info_map_b,
            protect_ratio=protect_ratio,
            protect_nms=protect_nms,
            protect_aux_map=protect_aux_map_b,
            protect_aux_ratio=protect_aux_ratio,
        )
        per_batch_ops.append((merge_b, unmerge_b))

    def merge(x: torch.Tensor, mode: str = "mean", extra_tensors=None, extra_tensors_2=None):
        results_main, results_extra1, results_extra2 = [], [], []
        for b in range(B):
            m_b, _ = per_batch_ops[b]
            out = m_b(
                x[b:b+1],
                mode=mode,
                extra_tensors=None if extra_tensors is None else extra_tensors[b:b+1],
                extra_tensors_2=None if extra_tensors_2 is None else extra_tensors_2[b:b+1],
            )
            if isinstance(out, tuple):
                results_main.append(out[0])
                if len(out) > 1:
                    results_extra1.append(out[1])
                if len(out) > 2:
                    results_extra2.append(out[2])
            else:
                results_main.append(out)

        main = torch.cat(results_main, dim=0)
        if results_extra1 and results_extra2:
            return main, torch.cat(results_extra1, dim=0), torch.cat(results_extra2, dim=0)
        elif results_extra1:
            return main, torch.cat(results_extra1, dim=0)
        else:
            return main

    def unmerge(x: torch.Tensor):
        results = []
        for b in range(B):
            _, u_b = per_batch_ops[b]
            results.append(u_b(x[b:b+1]))
        return torch.cat(results, dim=0)

    return merge, unmerge


def compute_info_maps(
    images_normed: torch.Tensor,   # [N, 3, H, W]  
    patch_tokens: torch.Tensor,    # [N, P, C] 
    var_win: int = 3,
    proj_dim: int = 32,
    depth_map: Optional[torch.Tensor] = None,  # [N, H, W] or [N, 1, H, W]
    depth_map_is_boundary: bool = False,
    edge_weight: float = 0.7,
    variance_weight: float = 0.3,
    depth_boundary_weight: float = 0.0,
    interaction_weight: float = 0.0,
    interaction_mode: str = "sqrt",
    laplacian_weight: float = 0.0,
    adaptive_weights: bool = False,
    adaptive_protect_ratio: bool = False,
    protect_base_ratio: float = 0.1,
    protect_complexity_lambda: float = 0.0,
    protect_min_ratio: float = 0.05,
    protect_max_ratio: float = 0.2,
):

    images_normed = images_normed.to(torch.float32)
    patch_tokens = patch_tokens.to(torch.float32)
    device = patch_tokens.device
    N, P, C = patch_tokens.shape
    H = images_normed.shape[-2]
    W = images_normed.shape[-1]
    patch_size = 14
    Hp = H // patch_size
    Wp = W // patch_size
    Hc, Wc = Hp * patch_size, Wp * patch_size
    assert P == Hp * Wp, f"P={P} vs Hp*Wp={Hp*Wp}"

    tok = patch_tokens.view(N, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # [N,C,Hp,Wp]

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        Pmat = torch.empty(C, proj_dim, device=device)
        torch.nn.init.orthogonal_(Pmat)
    X = torch.einsum('nchw,cd->ndhw', tok, Pmat)   # [N,d,Hp,Wp]

    pad = var_win // 2 
    mu = F.avg_pool2d(X, kernel_size=var_win, stride=1, padding=pad, count_include_pad=False)       # [N,d,Hp,Wp]
    m2 = F.avg_pool2d(X*X, kernel_size=var_win, stride=1, padding=pad, count_include_pad=False)     # [N,d,Hp,Wp]
    var_map = (m2 - mu*mu).clamp_min(0.0).sum(dim=1, keepdim=True)         # [N,1,Hp,Wp]

    x = images_normed[:, :, :Hc, :Wc]           # [N,3,Hc,Wc]
    gray = (0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3])
    # Sobel
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
    ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx*gx + gy*gy + 1e-12)    # [N,1,Hc,Wc]
    grad_map_tok = F.adaptive_avg_pool2d(grad, (Hp, Wp))  # [N,1,Hp,Wp]

    klap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=device).view(1, 1, 3, 3)
    lap = F.conv2d(gray, klap, padding=1).abs()
    lap_map_tok = F.adaptive_avg_pool2d(lap, (Hp, Wp))

    def norm01(t):
        tmin = t.amin(dim=(-2,-1), keepdim=True)
        tmax = t.amax(dim=(-2,-1), keepdim=True)
        return (t - tmin) / (tmax - tmin + 1e-8)

    var_n  = norm01(var_map)
    grad_n = norm01(grad_map_tok)
    lap_n = norm01(lap_map_tok)
    interaction_product = (grad_n * var_n).clamp_min(0.0)
    if interaction_mode == "product":
        interaction_n = interaction_product
    elif interaction_mode == "sqrt":
        interaction_n = torch.sqrt(interaction_product + 1e-8)
    else:
        raise ValueError(f"Unknown GA interaction mode: {interaction_mode}")
    grad_mean = grad_n.mean(dim=(-2, -1), keepdim=True)
    var_mean = var_n.mean(dim=(-2, -1), keepdim=True)

    depth_boundary_map = torch.zeros_like(grad_map_tok)
    if depth_map is not None:
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        depth_map = depth_map.to(torch.float32)[..., :Hc, :Wc]

        if depth_map_is_boundary:
            depth_boundary_map = F.adaptive_avg_pool2d(depth_map.abs(), (Hp, Wp))
        else:
            dx = F.pad(depth_map[..., :, 1:] - depth_map[..., :, :-1], (0, 1, 0, 0))
            dy = F.pad(depth_map[..., 1:, :] - depth_map[..., :-1, :], (0, 0, 0, 1))
            depth_grad = torch.sqrt(dx * dx + dy * dy + 1e-12)
            depth_boundary_map = F.adaptive_avg_pool2d(depth_grad, (Hp, Wp))

    depth_n = norm01(depth_boundary_map)
    if adaptive_weights:
        denom = grad_mean + var_mean + 1e-8
        edge_w = (var_mean / denom).to(grad_n.dtype)
        variance_w = (grad_mean / denom).to(var_n.dtype)
        depth_w = torch.full_like(edge_w, depth_boundary_weight)
        interaction_w = torch.full_like(edge_w, interaction_weight)
        laplacian_w = torch.full_like(edge_w, laplacian_weight)
        total_weight = edge_w + variance_w + depth_w + interaction_w + laplacian_w
    else:
        total_weight = edge_weight + variance_weight + depth_boundary_weight + interaction_weight + laplacian_weight
        if total_weight <= 0:
            raise ValueError("At least one GA metric weight must be positive.")
        edge_w = edge_weight
        variance_w = variance_weight
        depth_w = depth_boundary_weight
        interaction_w = interaction_weight
        laplacian_w = laplacian_weight

    info = (
        variance_w * var_n
        + edge_w * grad_n
        + depth_w * depth_n
        + interaction_w * interaction_n
        + laplacian_w * lap_n
    ) / total_weight
    info_n = norm01(info)
    gamma = 1.4  
    info_n = info_n ** gamma
    info_up = F.interpolate(info_n, size=(Hc, Wc), mode='bilinear', align_corners=False)  # [N,1,Hc,Wc]

    complexity = (grad_mean + var_mean).flatten()
    if adaptive_protect_ratio:
        protect_ratio = protect_base_ratio + protect_complexity_lambda * complexity
        protect_ratio = protect_ratio.clamp(protect_min_ratio, protect_max_ratio)
    else:
        protect_ratio = torch.full_like(complexity, protect_base_ratio)

    base_total_weight = edge_weight + variance_weight
    if base_total_weight > 0:
        base_info = (variance_weight * var_n + edge_weight * grad_n) / base_total_weight
        base_info_n = norm01(base_info) ** gamma
    else:
        base_info_n = info_n

    return {
        "var_map": var_map,               # [N,1,Hp,Wp]
        "grad_map_tok": grad_map_tok,     # [N,1,Hp,Wp]
        "laplacian_map": lap_map_tok,     # [N,1,Hp,Wp]
        "depth_boundary_map": depth_boundary_map,  # [N,1,Hp,Wp]
        "depth_info_map": depth_n.to(torch.bfloat16),  # [N,1,Hp,Wp]
        "base_info_map": base_info_n.to(torch.bfloat16),  # [N,1,Hp,Wp]
        "interaction_map": interaction_n,  # [N,1,Hp,Wp]
        "complexity": complexity,          # [N]
        "protect_ratio": protect_ratio,    # [N]
        "info_map": info_n.to(torch.bfloat16),               # [N,1,Hp,Wp]
        "info_up": info_up,               # [N,1,Hc,Wc]
        "Hp": Hp, "Wp": Wp, "Hc": Hc, "Wc": Wc
    }

def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,
    use_dynamic_protect: bool = False,  # ← 동적 protect_ratio 사용 여부
    protect_ratio=0.1,
    protect_nms: bool = False,
    protect_aux_map=None,
    protect_aux_ratio: float = 0.0,
    verbose: bool = True,               # ← 프레임별 출력 여부
) -> Tuple[Callable, Callable]:
    """
    Divide tokens into source (src) and destination (dst) groups, and merge r tokens from src to dst.
    dst tokens are selected by randomly choosing one token from each (sx, sy) region.
    Optionally protect the top 10% of tokens from merging based on importance scores.

    Args:
     - metric [B, N, C]: Tensor for similarity computation
     - w: Image width in tokens
     - h: Image height in tokens
     - sx: dst stride in x dimension
     - sy: dst stride in y dimension
     - r: Number of tokens to remove through merging
     - no_rand: If True, disable randomness
     - generator: Random number generator
     - enable_protection: If True, enable importance protection feature
     - use_dynamic_protect: If True, use dynamic protect_ratio based on frame complexity
     - protect_ratio: protect_ratio value (used when use_dynamic_protect=False)
     - verbose: If True, print per-frame protect_ratio

    Returns:
     - (merge, unmerge): Two functions for merging tokens and restoring pre-merge state
    """

    B, N, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather

    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    with torch.no_grad():
        if enable_protection:
            if info_map is not None:
                info = info_map[:, 0].to(metric.device)  # [num_imgs, Hp, Wp]
                aux_info = protect_aux_map[:, 0].to(metric.device) if protect_aux_map is not None else None
                if use_dynamic_protect:
                    protect_ratio = get_dynamic_protect_ratio_single(
                        info,
                        min_ratio=0.05,
                        max_ratio=0.20,
                        verbose=verbose,
                    )
                if torch.is_tensor(protect_ratio):
                    protect_ratios = protect_ratio.to(metric.device).flatten().to(torch.float32)
                    if protect_ratios.numel() == 1:
                        protect_ratios = protect_ratios.expand(num_imgs)
                    elif protect_ratios.numel() != num_imgs:
                        repeats = (num_imgs + protect_ratios.numel() - 1) // protect_ratios.numel()
                        protect_ratios = protect_ratios.repeat(repeats)[:num_imgs]
                else:
                    protect_ratios = torch.full((num_imgs,), float(protect_ratio), device=metric.device)

                topk_per_img = []
                num_patch_tokens = info.shape[-2] * info.shape[-1]
                for img_idx in range(num_imgs):
                    ratio_i = float(protect_ratios[img_idx].clamp(0.0, 1.0).item())
                    aux_ratio_i = max(0.0, min(float(protect_aux_ratio), ratio_i))
                    base_ratio_i = ratio_i - aux_ratio_i if aux_info is not None else ratio_i
                    k_i = max(1, int(num_patch_tokens * base_ratio_i))
                    info_i = info[img_idx:img_idx + 1].unsqueeze(1)
                    if protect_nms:
                        local_max = info_i == F.max_pool2d(info_i, kernel_size=3, stride=1, padding=1)
                        candidate = info_i.masked_fill(~local_max, torch.finfo(info_i.dtype).min)
                        valid_count = int(local_max.sum().item())
                        if valid_count >= k_i:
                            topk_idx = candidate.flatten().topk(k_i, dim=0).indices
                        else:
                            topk_idx = info_i.flatten().topk(k_i, dim=0).indices
                    else:
                        topk_idx = info_i.flatten().topk(k_i, dim=0).indices
                    if aux_info is not None and aux_ratio_i > 0.0:
                        k_aux = max(1, int(num_patch_tokens * aux_ratio_i))
                        aux_i = aux_info[img_idx].flatten().clone()
                        aux_i[topk_idx] = torch.finfo(aux_i.dtype).min
                        aux_topk_idx = aux_i.topk(k_aux, dim=0).indices
                        topk_idx = torch.cat([topk_idx, aux_topk_idx], dim=0)
                    topk_per_img.append(topk_idx.unique() + img_idx * tokens_per_img + 5)

                protected_indices = torch.cat(topk_per_img, dim=0)
                num_protected = protected_indices.numel()

            else:
                ratio = float(protect_ratio) if not torch.is_tensor(protect_ratio) else float(protect_ratio.flatten()[0])
                ratio = max(0.0, min(1.0, ratio))
                num_protected = max(1, int(N * ratio))
                step = max(1, N // num_protected)
                protected_indices = torch.arange(0, N, step, device=metric.device)[
                    :num_protected
                ]
        else:
            protected_indices = None
            num_protected = 0

        # Global idx_buffer_seq of length N; -1 indicates dst, 0 indicates src
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx

        # Mark first image entirely as dst
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        if num_imgs > 1:
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1

            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1

                if info_map is not None:
                    info_map_other_imgs = info_map[1:, 0]  # [num_imgs-1, Hp, Wp]
                    Hp, Wp = info_map_other_imgs.shape[-2:]
                    valid_h = (Hp // sy) * sy
                    valid_w = (Wp // sx) * sx
                    info_valid = info_map_other_imgs[:, :valid_h, :valid_w]

                    all_rand_idx = (
                        info_valid
                        .view(total_other_imgs, valid_h // sy, sy, valid_w // sx, sx)
                        .reshape(total_other_imgs, valid_h // sy, valid_w // sx, sy * sx)
                        .argmin(dim=-1)
                    )
                else:
                    all_rand_idx = torch.randint(
                        sy * sx,
                        size=(total_other_imgs, hsy, wsx),
                        device=metric.device,
                        generator=generator,
                    )

                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )

                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )

                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        a_idx_orig = rand_idx[:, num_dst_orig:, :]
        b_idx_orig = rand_idx[:, :num_dst_orig, :]
        a_idx = a_idx_orig
        b_idx = b_idx_orig

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        def split(x):
            C = x.shape[-1]
            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)

        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

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

        # ========================================
        # 0427 토큰 수 로그 추가
        # ========================================
        after_merge = N - r_actual  # merge 후 살아남는 토큰 수
        per_frame_before = tokens_per_img          # merge 전 프레임당 토큰
        per_frame_after = after_merge // num_imgs  # merge 후 프레임당 토큰

        print(f"\n[Token] merge 전: {N:,}개 ({per_frame_before}개/프레임) | "
              f"GA: {num_protected:,} | Dst: {num_dst:,} | Src제거: {r_actual:,} | "
              f"merge 후: {after_merge:,}개 ({per_frame_after}개/프레임) | "
              f"압축률: {r_actual/N*100:.1f}%")

    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
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
            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
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
            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
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
        else:
            return main_result

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
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

    merge.b_idx = b_idx
    return merge, unmerge
