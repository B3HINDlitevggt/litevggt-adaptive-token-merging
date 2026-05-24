#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from data import DTUDataset  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Save qualitative DTU prediction panels.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dtu_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/dtu_qualitative")
    parser.add_argument("--scene", default=None, help="Optional scene name, e.g. scan1")
    parser.add_argument("--frames", default="0,8,16,24,32,40", help="Comma-separated frame indices to save")
    parser.add_argument("--max_frames", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ga_depth_dir", default=None)
    parser.add_argument("--ga_edge_weight", type=float, default=0.7)
    parser.add_argument("--ga_variance_weight", type=float, default=0.3)
    parser.add_argument("--ga_depth_boundary_weight", type=float, default=0.0)
    parser.add_argument("--ga_depth_map_is_boundary", action="store_true")
    parser.add_argument("--ga_interaction_weight", type=float, default=0.0)
    parser.add_argument("--ga_interaction_mode", choices=["sqrt", "product"], default="sqrt")
    parser.add_argument("--ga_laplacian_weight", type=float, default=0.0)
    parser.add_argument("--ga_adaptive_weights", action="store_true")
    parser.add_argument("--ga_adaptive_protect_ratio", action="store_true")
    parser.add_argument("--ga_protect_base_ratio", type=float, default=0.1)
    parser.add_argument("--ga_protect_complexity_lambda", type=float, default=0.0)
    parser.add_argument("--ga_protect_min_ratio", type=float, default=0.05)
    parser.add_argument("--ga_protect_max_ratio", type=float, default=0.2)
    parser.add_argument("--ga_protect_nms", action="store_true")
    parser.add_argument("--ga_depth_protect_ratio", type=float, default=0.0)
    return parser.parse_args()


def load_model(model_path, device):
    print("Initializing and loading VGGT model...")
    print(f"USING {model_path}")
    model = VGGT().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    model.eval()
    print("Model loaded")
    return model


def robust_norm(x, valid=None):
    x = np.asarray(x, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(x)
    else:
        valid = valid & np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def colorize(x, cmap=cv2.COLORMAP_TURBO, valid=None):
    norm = robust_norm(x, valid=valid)
    u8 = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cmap)
    if valid is not None:
        bgr[~valid] = 0
    return bgr


def add_label(img_bgr, label):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def to_rgb_u8(image_chw):
    rgb = image_chw.detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def overlay(rgb_u8, heat_bgr, alpha=0.45):
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)


def resize_map(x, hw):
    h, w = hw
    return cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def find_depth_file(depth_scene_dir, image_name):
    stem = os.path.splitext(image_name)[0]
    candidate_names = [
        f"{stem}.npy",
        f"{stem}.npz",
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.tif",
        f"{stem}.tiff",
        f"{stem}_depth.npy",
        f"{stem}_depth.npz",
        f"{stem}_depth.png",
        f"{stem}_depth.jpg",
        f"{stem}_depth.jpeg",
        f"{stem}_depth.tif",
        f"{stem}_depth.tiff",
    ]
    for candidate_name in candidate_names:
        candidate_path = depth_scene_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path
    return None


def load_depth_file(depth_path):
    if depth_path.suffix == ".npy":
        depth = np.load(depth_path)
    elif depth_path.suffix == ".npz":
        depth_npz = np.load(depth_path)
        depth = depth_npz["depth"] if "depth" in depth_npz else depth_npz[list(depth_npz.keys())[0]]
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def build_ga_depth_tensor(ga_depth_dir, scene, image_paths, target_hw):
    if ga_depth_dir is None:
        return None

    depth_scene_dir = Path(ga_depth_dir) / scene
    if not depth_scene_dir.is_dir():
        raise FileNotFoundError(f"GA depth scene directory not found: {depth_scene_dir}")

    target_h, target_w = target_hw
    depth_tensors = []
    for image_path in image_paths:
        depth_path = find_depth_file(depth_scene_dir, Path(image_path).name)
        if depth_path is None:
            raise FileNotFoundError(f"No GA depth file found for {image_path} under {depth_scene_dir}")

        depth = load_depth_file(depth_path)
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_tensors.append(torch.from_numpy(depth))

    return torch.stack(depth_tensors, dim=0)


def save_scene_panels(model, sample, args, frame_indices):
    scene = sample["scene"]
    images = sample["imgs"][: args.max_frames].to(args.device)
    image_paths = sample["image_paths"][: args.max_frames]
    h, w = images.shape[-2:]

    patch_width = w // 14
    patch_height = h // 14
    model.update_patch_dimensions(patch_width, patch_height)

    ga_depth = build_ga_depth_tensor(args.ga_depth_dir, scene, image_paths, images.shape[-2:])
    if ga_depth is not None:
        ga_depth = ga_depth.unsqueeze(0).to(args.device)

    with torch.no_grad():
        predictions = model(images, ga_depth=ga_depth, return_ga_info_maps=True)

    depth = predictions["depth"][0, :, :, :, 0].detach().float().cpu().numpy()
    depth_conf = predictions["depth_conf"][0].detach().float().cpu().numpy()
    ga_maps = predictions["ga_info_maps"]
    info_up = ga_maps["info_up"][:, 0].detach().float().cpu().numpy()
    base_info = ga_maps["base_info_map"][:, 0].detach().float().cpu().numpy()
    depth_info = ga_maps["depth_info_map"][:, 0].detach().float().cpu().numpy()
    interaction = ga_maps["interaction_map"][:, 0].detach().float().cpu().numpy()
    laplacian = ga_maps["laplacian_map"][:, 0].detach().float().cpu().numpy()

    scene_dir = Path(args.output_dir) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    valid_frames = [idx for idx in frame_indices if 0 <= idx < images.shape[0]]
    for idx in valid_frames:
        rgb = to_rgb_u8(images[idx])
        depth_valid = np.isfinite(depth[idx]) & (depth[idx] > 0)
        depth_bgr = colorize(depth[idx], valid=depth_valid)
        conf_bgr = colorize(depth_conf[idx])
        info_bgr = colorize(info_up[idx])
        base_bgr = colorize(resize_map(base_info[idx], (h, w)))
        depth_info_bgr = colorize(resize_map(depth_info[idx], (h, w)))
        interaction_bgr = colorize(resize_map(interaction[idx], (h, w)))
        laplacian_bgr = colorize(resize_map(laplacian[idx], (h, w)))

        panels = [
            add_label(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), f"{scene} frame {idx:03d} input"),
            add_label(depth_bgr, "pred depth"),
            add_label(conf_bgr, "depth confidence"),
            add_label(overlay(rgb, base_bgr), "baseline GA score"),
            add_label(overlay(rgb, depth_info_bgr), "depth boundary prior"),
            add_label(overlay(rgb, info_bgr), "final GA score"),
        ]
        if args.ga_interaction_weight > 0:
            panels.append(add_label(overlay(rgb, interaction_bgr), f"{args.ga_interaction_mode}(edge x variance) overlay"))
        if args.ga_laplacian_weight > 0:
            panels.append(add_label(overlay(rgb, laplacian_bgr), "laplacian proxy overlay"))
        panel = cv2.hconcat(panels)
        out_path = scene_dir / f"frame_{idx:03d}_qual.png"
        cv2.imwrite(str(out_path), panel)
        print(f"Saved {out_path}")


def main():
    args = parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"

    model = load_model(args.model_path, args.device)
    model.set_ga_metric_weights(
        edge_weight=args.ga_edge_weight,
        variance_weight=args.ga_variance_weight,
        depth_boundary_weight=args.ga_depth_boundary_weight,
        depth_map_is_boundary=args.ga_depth_map_is_boundary,
        interaction_weight=args.ga_interaction_weight,
        interaction_mode=args.ga_interaction_mode,
        laplacian_weight=args.ga_laplacian_weight,
        adaptive_weights=args.ga_adaptive_weights,
        adaptive_protect_ratio=args.ga_adaptive_protect_ratio,
        protect_base_ratio=args.ga_protect_base_ratio,
        protect_complexity_lambda=args.ga_protect_complexity_lambda,
        protect_min_ratio=args.ga_protect_min_ratio,
        protect_max_ratio=args.ga_protect_max_ratio,
        protect_nms=args.ga_protect_nms,
        depth_protect_ratio=args.ga_depth_protect_ratio,
    )

    frame_indices = [int(x) for x in args.frames.split(",") if x.strip()]
    dataset = DTUDataset(root_dir=args.dtu_dir, scene_name=args.scene)

    for sample in dataset:
        save_scene_panels(model, sample, args, frame_indices)


if __name__ == "__main__":
    main()
