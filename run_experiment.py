"""
LiteVGGT Dynamic Token Merging 실험 스크립트

[모드 선택]
  baseline           : 원본 (고정 10%, stride 2×2)
  dynamic_protect    : GA Token 비율만 동적 결정
  dynamic_grid       : Grid stride만 동적 결정
  dynamic_all        : 둘 다 동시 적용
  sttm               : STTM 공간+시간 merge (기존)
  quadtree_bipartite : Quadtree + Bipartite merge (★ 새 방식)

[cal_layer 옵션] (몇 번 merge를 재계산할지)
  --cal_layer_mode 4  : 기본값, [0, 6, 15, 20] (4번)
  --cal_layer_mode 3  : [0, 8, 16]
  --cal_layer_mode 2  : [0, 12]
  --cal_layer_mode 1  : [0]

[실행 예시]
  CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py \
      --img_dir ./SampleSet/test2 \
      --output_dir ./output/qt_bipartite \
      --gt_path "$GT_PATH" \
      --mode quadtree_bipartite \
      --baseline_dir ./output/baseline \
      --qt_spatial_thresh 0.8 \
      --cal_layer_mode 4
"""

import torch
import os
import numpy as np
import argparse
import time
import cv2
from datetime import datetime
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.eval_utils import load_image_file_crop
from vggt.utils.geometry import unproject_depth_map_to_point_map
from merging.complexity import set_log_path, flush_log, reset_frame_counter


# =====================================================================
# 로거
# =====================================================================
class Logger:
    def __init__(self, output_dir, mode_label):
        os.makedirs(output_dir, exist_ok=True)
        self.lines = []
        self.path = os.path.join(output_dir, "experiment_log.txt")
        self.mode_label = mode_label

    def add(self, text=""):
        self.lines.append(text)

    def header(self, img_dir, num_frames, H, W, ckpt_path, cal_layer):
        ph, pw = H // 14, W // 14
        tpf = ph * pw + 5
        self.add("=" * 60)
        self.add("LiteVGGT Dynamic Token Merging Experiment")
        self.add(f"실행 시각 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add("=" * 60)
        self.add()
        self.add("[실험 환경]")
        self.add(f"  모드             : {self.mode_label}")
        self.add(f"  체크포인트       : {ckpt_path}")
        self.add(f"  데이터셋         : {img_dir}")
        self.add(f"  프레임 수        : {num_frames}장")
        self.add(f"  이미지 해상도    : {H} × {W}")
        self.add(f"  패치 그리드      : {ph} × {pw}")
        self.add(f"  토큰 수/프레임   : {tpf}개  (패치 {ph*pw} + 특수 5)")
        self.add(f"  전체 토큰 수     : {tpf * num_frames:,}개")
        self.add(f"  cal_layer        : {cal_layer}  ({len(cal_layer)}번 재계산)")
        self.add()

    def merge_result(self, tokens_before, ga, dst, src_removed, tokens_after,
                     protect_ratio, stride, avg_complexity, num_frames):
        comp = src_removed / tokens_before * 100
        attn = (1 - (tokens_after / tokens_before) ** 2) * 100
        self.add("[Token Merging 결과]")
        self.add(f"  평균 복잡도              : {avg_complexity:.4f}")
        self.add(f"  protect_ratio            : {protect_ratio:.4f}  ({protect_ratio*100:.1f}%)")
        self.add(f"  grid stride              : {stride}×{stride}")
        self.add()
        self.add(f"  merge 전 토큰            : {tokens_before:,}개  ({tokens_before//num_frames}개/프레임)")
        self.add(f"  GA Token (보호)          : {ga:,}개  ({ga//num_frames}개/프레임)")
        self.add(f"  Dst Token (생존)         : {dst:,}개  ({dst//num_frames}개/프레임)")
        self.add(f"  Src Token (제거)         : {src_removed:,}개  ({src_removed//num_frames}개/프레임)")
        self.add(f"  merge 후 토큰            : {tokens_after:,}개  ({tokens_after//num_frames}개/프레임)")
        self.add()
        self.add(f"  압축률                   : {comp:.1f}%")
        self.add(f"  Attention 연산량 감소    : {attn:.1f}%")
        self.add()

    def qt_result(self, tokens_before, tokens_after, num_frames,
                  spatial_thresh, root_size, min_size, node_stats=None):
        comp = (1 - tokens_after / tokens_before) * 100
        attn = (1 - (tokens_after / tokens_before) ** 2) * 100
        self.add("[Quadtree-Bipartite Token Merging 결과]")
        self.add(f"  spatial_thresh           : {spatial_thresh}")
        self.add(f"  root_block_size          : {root_size}×{root_size}")
        self.add(f"  min_block_size           : {min_size}×{min_size}")
        self.add()
        if node_stats is not None and node_stats.get("total_nodes", 0) > 0:
            cnt = node_stats["counter"]
            total = node_stats["total_nodes"]
            self.add(f"  [노드 분포] (마지막 cal_layer 기준)")
            self.add(f"    8×8 노드 : {cnt.get(8,0):,}개  ({cnt.get(8,0)/total*100:.1f}%)")
            self.add(f"    4×4 노드 : {cnt.get(4,0):,}개  ({cnt.get(4,0)/total*100:.1f}%)")
            self.add(f"    2×2 노드 : {cnt.get(2,0):,}개  ({cnt.get(2,0)/total*100:.1f}%)")
            self.add(f"    전체 노드: {total:,}개")
            self.add(f"    평균 패치/노드: {node_stats['total_patches']/total:.2f}")
            self.add()
        self.add(f"  merge 전 토큰            : {tokens_before:,}개  ({tokens_before//num_frames}개/프레임)")
        self.add(f"  merge 후 토큰            : {tokens_after:,}개  ({tokens_after//num_frames}개/프레임)")
        self.add()
        self.add(f"  압축률                   : {comp:.1f}%")
        self.add(f"  Attention 연산량 감소    : {attn:.1f}%")
        self.add()

    def sttm_result(self, tokens_before, tokens_after, num_frames,
                    spatial_thresh, temporal_thresh):
        comp = (1 - tokens_after / tokens_before) * 100
        attn = (1 - (tokens_after / tokens_before) ** 2) * 100
        self.add("[STTM Token Merging 결과]")
        self.add(f"  spatial_thresh           : {spatial_thresh}")
        self.add(f"  temporal_thresh          : {temporal_thresh}")
        self.add()
        self.add(f"  merge 전 토큰            : {tokens_before:,}개  ({tokens_before//num_frames}개/프레임)")
        self.add(f"  merge 후 토큰 (추정)     : {tokens_after:,}개  ({tokens_after//num_frames}개/프레임)")
        self.add()
        self.add(f"  압축률 (추정)            : {comp:.1f}%")
        self.add(f"  Attention 연산량 감소    : {attn:.1f}%")
        self.add()

    def cd_result(self, cd, gt_path):
        self.add("[품질 평가 - Chamfer Distance]")
        self.add(f"  GT   : {gt_path}")
        self.add(f"  CD   : {cd:.6f}")
        self.add()

    def comparison(self, baseline_after, our_after, baseline_cd=None, our_cd=None):
        diff     = baseline_after - our_after
        diff_pct = diff / baseline_after * 100
        attn     = (1 - (our_after / baseline_after) ** 2) * 100
        self.add("[Baseline 대비 비교]")
        self.add(f"  Baseline  merge 후  : {baseline_after:,}개")
        self.add(f"  제안 방법 merge 후  : {our_after:,}개")
        self.add(f"  추가 감소           : {diff:,}개  ({diff_pct:.1f}%)")
        self.add(f"  Attention 추가 감소 : {attn:.1f}%")
        if baseline_cd is not None and our_cd is not None:
            cd_diff = our_cd - baseline_cd
            self.add()
            self.add(f"  Baseline CD         : {baseline_cd:.6f}")
            self.add(f"  제안 방법 CD        : {our_cd:.6f}")
            self.add(f"  CD 차이             : {cd_diff:+.6f}  ({'품질 거의 동일' if abs(cd_diff) < 1.0 else '품질 차이 있음'})")
        self.add()

    def save(self):
        self.add("=" * 60)
        self.add("실험 완료")
        self.add("=" * 60)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n✅ 실험 로그 저장: {self.path}")


# =====================================================================
# 유틸 (기존 그대로)
# =====================================================================
def save_ply(points, colors, save_dir, name, max_points=15000000):
    os.makedirs(save_dir, exist_ok=True)
    valid = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points, colors = points[valid], colors[valid].astype(np.uint8)
    if len(points) == 0:
        print("⚠️  No valid points.")
        return
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]
    path = os.path.join(save_dir, name)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    print(f"✅ PLY saved: {path}  ({len(points)} points)")


def compute_cd(pred_points, gt_path, num_sample=10000):
    try:
        from plyfile import PlyData
        data = PlyData.read(gt_path)
        gt = np.stack([np.array(data['vertex']['x']),
                       np.array(data['vertex']['y']),
                       np.array(data['vertex']['z'])], axis=1)
        ps = pred_points[np.random.choice(len(pred_points), min(num_sample, len(pred_points)), replace=False)]
        gs = gt[np.random.choice(len(gt), min(num_sample, len(gt)), replace=False)]
        pt = torch.from_numpy(ps).float().cuda()
        gt = torch.from_numpy(gs).float().cuda()
        d1 = torch.cdist(pt, gt).min(dim=1).values.mean()
        d2 = torch.cdist(gt, pt).min(dim=1).values.mean()
        return ((d1 + d2) / 2).item()
    except Exception as e:
        print(f"CD 계산 실패: {e}")
        return None


def load_baseline_cd(baseline_dir):
    cd_file = os.path.join(baseline_dir, "cd_result.txt")
    if os.path.exists(cd_file):
        try:
            with open(cd_file) as f:
                cd = float(f.read().strip())
            print(f"  Baseline CD 로드: {cd:.6f}")
            return cd
        except Exception as e:
            print(f"  Baseline CD 로드 실패: {e}")
    return None


def save_baseline_cd(output_dir, cd):
    cd_file = os.path.join(output_dir, "cd_result.txt")
    with open(cd_file, "w") as f:
        f.write(f"{cd:.6f}\n")
    print(f"✅ Baseline CD 저장: {cd_file}")


def get_cal_layer(mode):
    """cal_layer_mode에 따라 어느 레이어에서 merge 재계산할지 결정."""
    presets = {
        1: [0],
        2: [0, 12],
        3: [0, 8, 16],
        4: [0, 6, 15, 20],
    }
    return presets.get(mode, [0, 6, 15, 20])


def select_frames_by_sobel(image_paths, n_keep):
    if n_keep is None or len(image_paths) <= n_keep:
        selected = list(range(len(image_paths)))
        return image_paths, selected, []

    scores = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            scores.append(0.0)
            continue
        image = image.astype(np.float32) / 255.0
        sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        scores.append(float(np.sqrt(sx * sx + sy * sy).mean()))

    top_idx = np.argsort(np.asarray(scores))[::-1][:n_keep]
    selected = sorted(int(i) for i in top_idx)
    return [image_paths[i] for i in selected], selected, scores


def find_depth_file(depth_root, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(depth_root, f"{stem}.npy"),
        os.path.join(depth_root, f"{stem}.npz"),
        os.path.join(depth_root, f"{stem}.png"),
        os.path.join(depth_root, f"{stem}.jpg"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    for root, _, files in os.walk(depth_root):
        for ext in (".npy", ".npz", ".png", ".jpg", ".jpeg"):
            filename = f"{stem}{ext}"
            if filename in files:
                return os.path.join(root, filename)
    return None


def load_depth_file(path):
    if path.endswith(".npy"):
        depth = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def build_ga_depth_tensor(ga_depth_dir, image_paths, target_hw):
    if ga_depth_dir is None:
        return None

    target_h, target_w = target_hw
    depth_tensors = []
    for image_path in image_paths:
        depth_path = find_depth_file(ga_depth_dir, image_path)
        if depth_path is None:
            raise FileNotFoundError(
                f"No GA depth file found for image '{os.path.basename(image_path)}' under {ga_depth_dir}"
            )
        depth = load_depth_file(depth_path)
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_tensors.append(torch.from_numpy(depth.astype(np.float32)))

    stacked = torch.stack(depth_tensors, dim=0)
    print(
        f"✅ GA depth loaded: count={len(depth_tensors)}, shape={tuple(stacked.shape)}, "
        f"min={stacked.min().item():.4f}, max={stacked.max().item():.4f}, mean={stacked.mean().item():.4f}"
    )
    return stacked


# =====================================================================
# 메인
# =====================================================================
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",            type=str,   default="./te_dict.pt")
    parser.add_argument("--device",               type=str,   default="cuda:0")
    parser.add_argument("--img_dir",              type=str,   required=True)
    parser.add_argument("--keep_ratio",           type=float, default=0.42)
    parser.add_argument("--output_dir",           type=str,   required=True)
    parser.add_argument("--gt_path",              type=str,   default=None)
    parser.add_argument("--baseline_dir",         type=str,   default=None)
    parser.add_argument("--mode",                 type=str,   default="baseline",
                        choices=["baseline", "dynamic_protect", "dynamic_grid",
                                 "dynamic_all", "sttm", "quadtree_bipartite"])
    parser.add_argument("--sttm_spatial_thresh",  type=float, default=0.8)
    parser.add_argument("--sttm_temporal_thresh", type=float, default=0.6)
    parser.add_argument("--qt_spatial_thresh",    type=float, default=0.8,
                        help="Quadtree spatial merge 임계값")
    parser.add_argument("--qt_root_block_size",   type=int,   default=8,
                        help="Quadtree 시작 블록 크기 (8=Level 1)")
    parser.add_argument("--qt_min_block_size",    type=int,   default=2,
                        help="Quadtree 종료 블록 크기 (2=Level 3)")
    parser.add_argument("--cal_layer_mode",       type=int,   default=4,
                        choices=[1, 2, 3, 4],
                        help="몇 번 merge를 재계산할지 (1/2/3/4)")
    parser.add_argument("--max_frames",           type=int,   default=None)
    parser.add_argument("--frame_selection",      type=str,   default="none",
                        choices=["none", "sobel"],
                        help="Input frame selection method before LiteVGGT inference")
    parser.add_argument("--ga_depth_dir",         type=str,   default=None)
    parser.add_argument("--ga_edge_weight",       type=float, default=0.7)
    parser.add_argument("--ga_variance_weight",   type=float, default=0.3)
    parser.add_argument("--ga_depth_boundary_weight", type=float, default=0.0)
    parser.add_argument("--ga_depth_map_is_boundary", action="store_true")
    parser.add_argument("--ga_protect_base_ratio", type=float, default=0.1)
    parser.add_argument("--ga_depth_protect_ratio", type=float, default=0.0)
    parser.add_argument("--use_adaptive_cache",   action="store_true")
    parser.add_argument("--no_verbose",           action="store_true")
    return parser


def main(args):
    device  = args.device
    dtype   = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    verbose = not args.no_verbose

    cal_layer = get_cal_layer(args.cal_layer_mode)

    mode_labels = {
        "baseline":           "Baseline  (fixed 10%, stride 2×2)",
        "dynamic_protect":    "Dynamic Protect Ratio",
        "dynamic_grid":       "Dynamic Grid Stride",
        "dynamic_all":        "Dynamic ALL  (protect + grid)",
        "sttm":               f"STTM  (spatial={args.sttm_spatial_thresh}, temporal={args.sttm_temporal_thresh})",
        "quadtree_bipartite": f"Quadtree-Bipartite (qt_thresh={args.qt_spatial_thresh}, root={args.qt_root_block_size}, min={args.qt_min_block_size})",
    }
    mode_label = mode_labels[args.mode] + f" | cal_layer={cal_layer}"
    print(f"\n{'='*55}\n모드: {mode_label}\n{'='*55}")

    logger = Logger(args.output_dir, mode_label)

    # 모델 로드
    model = VGGT().to(device)
    ckpt  = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(torch.bfloat16)
    model.eval()
    model.set_ga_metric_weights(
        edge_weight=args.ga_edge_weight,
        variance_weight=args.ga_variance_weight,
        depth_boundary_weight=args.ga_depth_boundary_weight,
        depth_map_is_boundary=args.ga_depth_map_is_boundary,
        protect_base_ratio=args.ga_protect_base_ratio,
        depth_protect_ratio=args.ga_depth_protect_ratio,
    )

    # 모드별 플래그 설정
    model.aggregator.use_dynamic_protect    = args.mode in ("dynamic_protect", "dynamic_all")
    model.aggregator.use_dynamic_grid       = args.mode in ("dynamic_grid",    "dynamic_all")
    model.aggregator.use_sttm               = args.mode == "sttm"
    model.aggregator.use_quadtree_bipartite = args.mode == "quadtree_bipartite"

    model.aggregator.sttm_spatial_thresh    = args.sttm_spatial_thresh
    model.aggregator.sttm_temporal_thresh   = args.sttm_temporal_thresh
    model.aggregator.qt_spatial_thresh      = args.qt_spatial_thresh
    model.aggregator.qt_root_block_size     = args.qt_root_block_size
    model.aggregator.qt_min_block_size      = args.qt_min_block_size

    model.aggregator.cal_layer              = cal_layer
    model.aggregator.verbose_protect        = verbose

    reset_frame_counter()
    set_log_path(os.path.join(args.output_dir, "frame_log.csv"))

    # 이미지 로드
    all_images = sorted([
        os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    original_frame_count = len(all_images)
    selected_indices = list(range(original_frame_count))
    sobel_scores = []
    if args.frame_selection == "sobel":
        all_images, selected_indices, sobel_scores = select_frames_by_sobel(all_images, args.max_frames)
        print(f"✅ Sobel frame selection: {original_frame_count} -> {len(all_images)} frames")
        print(f"   selected indices: {selected_indices}")
    elif args.max_frames is not None:
        all_images = all_images[:args.max_frames]
        selected_indices = selected_indices[:args.max_frames]

    images = []
    for p in all_images:
        img = load_image_file_crop(p)
        images.append(torch.from_numpy(np.transpose(img, (2, 0, 1))))

    N         = len(images)
    N_aligned = (N // 8) * 8
    if N_aligned == 0:
        print(f"❌ Error: aligned 프레임 수가 0입니다. N={N}")
        return

    images    = torch.stack(images[:N_aligned], dim=0).to(device)
    H, W      = images.shape[-2], images.shape[-1]
    print(f"✅ 데이터 로드 완료 - images shape: {images.shape}")

    logger.header(args.img_dir, N_aligned, H, W, args.ckpt_path, cal_layer)
    logger.add("[Pipeline Options]")
    logger.add(f"  frame_selection          : {args.frame_selection}")
    logger.add(f"  original_frame_count     : {original_frame_count}")
    logger.add(f"  selected_frame_count     : {N_aligned}")
    logger.add(f"  selected_indices         : {selected_indices[:N_aligned]}")
    if sobel_scores:
        kept_scores = [sobel_scores[i] for i in selected_indices[:N_aligned]]
        logger.add(f"  selected_sobel_mean      : {float(np.mean(kept_scores)):.6f}")
    logger.add(f"  token_merging_mode       : {args.mode}")
    logger.add(f"  adaptive_cache           : {args.use_adaptive_cache}")
    logger.add(f"  ga_depth_dir             : {args.ga_depth_dir}")
    logger.add(f"  ga_weights               : edge={args.ga_edge_weight}, variance={args.ga_variance_weight}, depth={args.ga_depth_boundary_weight}")
    logger.add(f"  ga_protect               : base={args.ga_protect_base_ratio}, depth_split={args.ga_depth_protect_ratio}")
    logger.add()

    patch_width  = W // 14
    patch_height = H // 14
    model.update_patch_dimensions(patch_width, patch_height)
    images = images[None]
    ga_depth = build_ga_depth_tensor(args.ga_depth_dir, all_images[:N_aligned], images.shape[-2:])
    if ga_depth is not None:
        ga_depth = ga_depth.unsqueeze(0).to(device)

    # 추론
    t_start = time.time()
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = model.aggregator(
            images,
            ga_depth=ga_depth,
            use_adaptive_cache=args.use_adaptive_cache,
        )

        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, patch_start_idx)

        points_3d = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), w2c_pre.squeeze(0), intrinsic.squeeze(0)
        )
        points = points_3d.reshape(-1, 3)
        colors = np.clip(
            images[0].permute(0,2,3,1).reshape(-1,3).cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

        conf_flat = depth_conf.reshape(-1).cpu().numpy()
        keep_idx  = np.argsort(conf_flat)[::-1][:int(len(conf_flat) * args.keep_ratio)]
        points    = points[keep_idx]
        colors    = colors[keep_idx]

        save_ply(points, colors, args.output_dir, "recon.ply")

    t_elapsed = time.time() - t_start
    print(f"✅ 추론 시간: {t_elapsed:.1f}초 ({t_elapsed/60:.1f}분)")

    # 토큰 수치 계산 및 로그
    tokens_before = (patch_height * patch_width + 5) * N_aligned

    logger.add(f"[실행 시간]")
    logger.add(f"  추론 시간 : {t_elapsed:.1f}초  ({t_elapsed/60:.1f}분)")
    logger.add()

    if args.mode == "quadtree_bipartite":
        # 노드 통계 가져오기
        from merging.sttm_bipartite_merge import get_node_stats
        node_stats = get_node_stats()

        # 노드 분포로 추정 토큰 수 계산
        # 노드 1개당 1 Dst + (노드 평균 패치 - 1) Src 후보
        # 실제 압축률은 merge_ratio에 따라 달라지지만 보통 노드당 1개로 압축됨
        if node_stats.get("total_nodes", 0) > 0:
            nodes_per_frame = node_stats["total_nodes"] // N_aligned
            ga_per_frame = max(1, int(patch_height * patch_width * 0.1))
            tokens_after = (nodes_per_frame + ga_per_frame + 5) * N_aligned
        else:
            tokens_after = int(tokens_before * 0.35)
            print("⚠️  노드 통계 못 가져옴, 추정값 사용")

        logger.qt_result(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            num_frames=N_aligned,
            spatial_thresh=args.qt_spatial_thresh,
            root_size=args.qt_root_block_size,
            min_size=args.qt_min_block_size,
            node_stats=node_stats,
        )

    elif args.mode == "sttm":
        tokens_after = getattr(model.aggregator, 'last_sttm_tokens_after', None)
        if tokens_after is None:
            tokens_after = int(tokens_before * 0.35)
        logger.sttm_result(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            num_frames=N_aligned,
            spatial_thresh=args.sttm_spatial_thresh,
            temporal_thresh=args.sttm_temporal_thresh,
        )
    else:
        protect_ratio  = 0.1
        stride         = 2
        avg_complexity = 0.0
        if args.mode in ("dynamic_protect", "dynamic_all"):
            protect_ratio = 0.07
        if args.mode in ("dynamic_grid", "dynamic_all"):
            stride = 4
        ga_per_frame  = max(1, int(patch_height * patch_width * protect_ratio))
        dst_per_frame = (patch_height // stride) * (patch_width // stride)
        src_per_frame = patch_height * patch_width - ga_per_frame - dst_per_frame
        tokens_after  = (ga_per_frame + dst_per_frame + 5) * N_aligned
        src_removed   = src_per_frame * N_aligned
        logger.merge_result(
            tokens_before=tokens_before,
            ga=ga_per_frame * N_aligned,
            dst=dst_per_frame * N_aligned,
            src_removed=src_removed,
            tokens_after=tokens_after,
            protect_ratio=protect_ratio,
            stride=stride,
            avg_complexity=avg_complexity,
            num_frames=N_aligned,
        )

    # CD 계산
    cd = None
    if args.gt_path:
        cd = compute_cd(points, args.gt_path)
        if cd is not None:
            print(f"✅ Chamfer Distance: {cd:.6f}")
            logger.cd_result(cd, args.gt_path)

    if args.mode == "baseline" and cd is not None:
        save_baseline_cd(args.output_dir, cd)

    if args.mode != "baseline":
        baseline_ga    = max(1, int(patch_height * patch_width * 0.1))
        baseline_dst   = (patch_height // 2) * (patch_width // 2)
        baseline_after = (baseline_ga + baseline_dst + 5) * N_aligned

        baseline_cd = None
        if args.baseline_dir:
            baseline_cd = load_baseline_cd(args.baseline_dir)
        else:
            guessed = args.output_dir.replace(args.mode, "baseline")
            if guessed != args.output_dir:
                baseline_cd = load_baseline_cd(guessed)

        logger.comparison(
            baseline_after=baseline_after,
            our_after=tokens_after,
            baseline_cd=baseline_cd,
            our_cd=cd,
        )

    flush_log()
    logger.save()
    print("done!")


if __name__ == "__main__":
    parser = get_args_parser()
    args   = parser.parse_args()
    main(args)



"""
# Quadtree-Bipartite, 4번 재계산 (기본)
CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py \
    --img_dir ./SampleSet/test2 \
    --output_dir ./output/qt_4 \
    --gt_path "$GT_PATH" \
    --mode quadtree_bipartite \
    --baseline_dir ./output/baseline \
    --qt_spatial_thresh 0.8 \
    --cal_layer_mode 4

# 1번만 재계산
CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py \
    --img_dir ./SampleSet/test2 \
    --output_dir ./output/qt_1 \
    --gt_path "$GT_PATH" \
    --mode quadtree_bipartite \
    --baseline_dir ./output/baseline \
    --cal_layer_mode 1

# baseline에서도 cal_layer 줄여보기
CUDA_VISIBLE_DEVICES=0 python3 run_experiment.py \
    --img_dir ./SampleSet/test2 \
    --output_dir ./output/baseline_cal1 \
    --gt_path "$GT_PATH" \
    --mode baseline \
    --cal_layer_mode 1
"""
