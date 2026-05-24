#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Depth Anything V2 depth-boundary maps for GA.")
    parser.add_argument("--dtu_dir", required=True, help="Eval DTU root with scan*/images")
    parser.add_argument("--output_dir", required=True, help="Output root for boundary npy files")
    parser.add_argument("--checkpoint", required=True, help="Depth Anything V2 checkpoint path")
    parser.add_argument("--repo_dir", default=None, help="Optional local Depth-Anything-V2 repo path")
    parser.add_argument("--encoder", choices=MODEL_CONFIGS.keys(), default="vits")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene names; default: all scenes")
    parser.add_argument("--save_raw_depth", action="store_true", help="Also save robust-normalized raw depth maps")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robust_norm(x, p_low=1, p_high=99):
    x = x.astype(np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    x[~valid] = 0.0
    return x.astype(np.float32)


def depth_to_boundary(depth):
    depth = robust_norm(depth)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    boundary = np.sqrt(gx * gx + gy * gy)
    return robust_norm(boundary)


def get_image_paths(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in exts)


def load_da2_model(args):
    if args.repo_dir is not None:
        sys.path.insert(0, str(Path(args.repo_dir).resolve()))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise ImportError(
            "Could not import Depth Anything V2. Clone the official repo and pass "
            "--repo_dir /path/to/Depth-Anything-V2, or install it on PYTHONPATH."
        ) from exc

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(args.device).eval()
    return model


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    dtu_dir = Path(args.dtu_dir)
    output_dir = Path(args.output_dir)
    raw_output_dir = output_dir.parent / f"{output_dir.name}_raw"

    if args.scenes:
        scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    else:
        scenes = sorted(path.name for path in dtu_dir.iterdir() if (path / "images").is_dir())

    model = load_da2_model(args)
    print(f"Loaded Depth Anything V2 {args.encoder} from {args.checkpoint}")
    print(f"Scenes: {scenes}")

    for scene in scenes:
        images_dir = dtu_dir / scene / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")

        out_scene_dir = output_dir / scene
        out_scene_dir.mkdir(parents=True, exist_ok=True)
        if args.save_raw_depth:
            (raw_output_dir / scene).mkdir(parents=True, exist_ok=True)

        image_paths = get_image_paths(images_dir)
        print(f"{scene}: {len(image_paths)} images")
        for image_idx, image_path in enumerate(image_paths, start=1):
            out_path = out_scene_dir / f"{image_path.stem}.npy"
            if out_path.exists() and not args.overwrite:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            with torch.no_grad():
                depth = model.infer_image(image, args.input_size)

            depth = np.asarray(depth, dtype=np.float32)
            boundary = depth_to_boundary(depth)
            np.save(out_path, boundary.astype(np.float32))

            if args.save_raw_depth:
                raw_depth = robust_norm(depth)
                np.save(raw_output_dir / scene / f"{image_path.stem}.npy", raw_depth)

            if image_idx % 10 == 0 or image_idx == len(image_paths):
                print(f"  {image_idx:04d}/{len(image_paths):04d}: {image_path.name}")

    print(f"Saved boundary maps to {output_dir}")
    if args.save_raw_depth:
        print(f"Saved raw normalized depth maps to {raw_output_dir}")


if __name__ == "__main__":
    main()
