"""
OOM 임계점 탐색 실험

baseline과 dynamic_all이 각각 몇 장에서 OOM이 나는지,
어느 단계에서 터지는지 찾는 스크립트.

실행 예시:
CUDA_VISIBLE_DEVICES=0 python3 run_oom_test.py \
    --img_dir ./data_scannetpp/a492fe77aa \
    --ckpt_path ./te_dict.pt \
    --output_dir ./output/oom_test \
    --max_frames 1400 \
    --start_frames 900 \
    --step 64
"""

import torch
import os
import gc
import traceback
import numpy as np
import argparse
from datetime import datetime

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import load_image_file_crop
from merging.complexity import reset_frame_counter


# =====================================================================
# OOM 위치 분석
# =====================================================================
def parse_oom_location(tb_str: str) -> str:
    """
    traceback 문자열에서 OOM 발생 위치를 추출합니다.
    """
    lines = tb_str.strip().split("\n")

    relevant = [l for l in lines if "File" in l and "site-packages" not in l]

    if not relevant:
        relevant = [l for l in lines if "File" in l]

    if not relevant:
        return "위치 파악 불가"

    if "patch_embed" in tb_str or "vision_transformer" in tb_str:
        stage = "DINOv2 인코딩 (patch_embed)"
    elif "compute_info_maps" in tb_str:
        stage = "GA Map 계산"
    elif "token_merge" in tb_str or "merge.py" in tb_str:
        stage = "Token Merging"
    elif "attn" in tb_str and "global" in tb_str.lower():
        stage = "Global Attention"
    elif "attn" in tb_str:
        stage = "Attention 연산"
    elif "normMlp" in tb_str or "LayerNormMLP" in tb_str:
        stage = "MLP 레이어"
    elif "aggregator" in tb_str:
        stage = "Aggregator"
    elif "camera_head" in tb_str:
        stage = "Camera Head"
    elif "depth_head" in tb_str:
        stage = "Depth Head"
    else:
        stage = "기타"

    alloc = ""
    for l in lines:
        if "Tried to allocate" in l:
            import re
            m = re.search(r"Tried to allocate ([\d.]+ \w+)", l)
            if m:
                alloc = f"  (할당 시도: {m.group(1)})"
            break

    return f"{stage}{alloc}"


# =====================================================================
# 모델 로드
# =====================================================================
def load_model(ckpt_path, device, mode):
    model = VGGT().to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)

    model.to(torch.bfloat16)
    model.eval()

    if mode == "baseline":
        model.aggregator.use_dynamic_protect = False
        model.aggregator.use_dynamic_grid = False

    elif mode == "dynamic_all":
        model.aggregator.use_dynamic_protect = True
        model.aggregator.use_dynamic_grid = True

    model.aggregator.verbose_protect = False

    return model


# =====================================================================
# 추론 시도
# =====================================================================
def try_inference(model, images_full, n_frames, device):
    """
    n_frames만큼 잘라서 추론 시도.
    반환:
        success: bool
        oom_location: str | None
    """

    # 8-frame alignment
    n = (n_frames // 8) * 8

    if n == 0:
        return False, "프레임 수 부족"

    imgs = images_full[:n].to(device)

    patch_width = imgs.shape[-1] // 14
    patch_height = imgs.shape[-2] // 14

    model.update_patch_dimensions(patch_width, patch_height)

    reset_frame_counter()

    try:
        with torch.no_grad():
            inp = imgs[None]

            aggregated_tokens_list, patch_start_idx = model.aggregator(inp)

        del aggregated_tokens_list
        del patch_start_idx
        del inp
        del imgs

        torch.cuda.empty_cache()
        gc.collect()

        return True, None

    except torch.cuda.OutOfMemoryError:
        tb_str = traceback.format_exc()

        location = parse_oom_location(tb_str)

        torch.cuda.empty_cache()
        gc.collect()

        return False, location

    except Exception as e:
        tb_str = traceback.format_exc()

        torch.cuda.empty_cache()
        gc.collect()

        return False, f"기타 에러: {type(e).__name__}"


# =====================================================================
# OOM 임계점 탐색
# =====================================================================
def find_oom_threshold(
    model,
    images_full,
    device,
    mode_name,
    frame_list,
    result_lines
):
    print(f"\n{'=' * 60}")
    print(f"[{mode_name}] OOM 임계점 탐색")
    print(f"{'=' * 60}")

    last_success = 0
    first_oom = None
    oom_location = None

    detail_rows = []

    for n in frame_list:

        n_aligned = (n // 8) * 8

        if n_aligned == 0:
            continue

        print(f"  {n_aligned:>5}장 시도... ", end="", flush=True)

        success, loc = try_inference(
            model,
            images_full,
            n_aligned,
            device
        )

        if success:
            print("✅ 성공")

            last_success = n_aligned

            detail_rows.append(
                f"  {n_aligned:>5}장  ✅ 성공"
            )

        else:
            print(f"❌ OOM  ← {loc}")

            first_oom = n_aligned
            oom_location = loc

            detail_rows.append(
                f"  {n_aligned:>5}장  ❌ OOM  ← {loc}"
            )

            break

    result_lines.append(f"[{mode_name}]")

    for row in detail_rows:
        result_lines.append(row)

    result_lines.append("")

    result_lines.append(
        f"  → 마지막 성공: {last_success}장"
    )

    if first_oom:
        result_lines.append(
            f"  → 첫 OOM:     {first_oom}장"
        )

        result_lines.append(
            f"  → OOM 위치:   {oom_location}"
        )

    else:
        result_lines.append(
            "  → 테스트 범위 내 OOM 없음"
        )

    result_lines.append("")

    return last_success, first_oom, oom_location


# =====================================================================
# argparse
# =====================================================================
def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./te_dict.pt"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/oom_test"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=512
    )

    parser.add_argument(
        "--step",
        type=int,
        default=None
    )

    # 추가
    parser.add_argument(
        "--start_frames",
        type=int,
        default=8
    )

    return parser


# =====================================================================
# main
# =====================================================================
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device

    result_lines = []

    result_lines.append("=" * 60)
    result_lines.append("LiteVGGT OOM 임계점 탐색 결과")
    result_lines.append(
        f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    result_lines.append(f"데이터셋:  {args.img_dir}")
    result_lines.append("=" * 60)
    result_lines.append("")

    # ==========================================================
    # 이미지 로드
    # ==========================================================
    print("이미지 로드 중...")

    all_paths = sorted([
        os.path.join(args.img_dir, f)
        for f in os.listdir(args.img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    images = []

    for p in all_paths[:args.max_frames]:
        img = load_image_file_crop(p)

        images.append(
            torch.from_numpy(
                np.transpose(img, (2, 0, 1))
            )
        )

    images_full = torch.stack(images, dim=0).cpu()

    total = len(images_full)

    H, W = images_full.shape[-2], images_full.shape[-1]

    print(f"✅ 총 {total}장 로드  ({H}×{W})")

    result_lines.append(f"총 이미지:  {total}장")
    result_lines.append(f"해상도:     {H} × {W}")
    result_lines.append("")

    # ==========================================================
    # 프레임 리스트 생성
    # ==========================================================
    start = ((args.start_frames + 7) // 8) * 8

    if args.step:
        frame_list = list(
            range(start, total + 1, args.step)
        )

    else:
        frame_list = [
            8, 16, 24, 32,
            48, 64, 80, 96,
            128, 160, 192,
            256, 320, 384,
            448, 512
        ]

        frame_list = [
            f for f in frame_list
            if f >= start and f <= total
        ]

    result_lines.append(f"시작 프레임: {start}")
    result_lines.append(f"테스트 목록: {frame_list}")
    result_lines.append("")

    # ==========================================================
    # Baseline
    # ==========================================================
    print("\n모델 로드 (Baseline)...")

    model_b = load_model(
        args.ckpt_path,
        device,
        "baseline"
    )

    b_success, b_oom, b_loc = find_oom_threshold(
        model_b,
        images_full,
        device,
        "Baseline (fixed 10%, stride 2×2)",
        frame_list,
        result_lines
    )

    del model_b

    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================================
    # Dynamic ALL
    # ==========================================================
    print("\n모델 로드 (Dynamic ALL)...")

    model_d = load_model(
        args.ckpt_path,
        device,
        "dynamic_all"
    )

    d_success, d_oom, d_loc = find_oom_threshold(
        model_d,
        images_full,
        device,
        "Dynamic ALL (protect + grid, stride 4×4)",
        frame_list,
        result_lines
    )

    del model_d

    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================================
    # 비교 요약
    # ==========================================================
    result_lines.append("=" * 60)
    result_lines.append("최종 비교 요약")
    result_lines.append("=" * 60)

    result_lines.append(
        f"  {'':20} {'마지막 성공':>12}  {'첫 OOM':>8}  OOM 위치"
    )

    result_lines.append(f"  {'-' * 70}")

    result_lines.append(
        f"  {'Baseline':20} "
        f"{b_success:>10}장  "
        f"{str(b_oom or '-'):>6}장  "
        f"{b_loc or '-'}"
    )

    result_lines.append(
        f"  {'Dynamic ALL':20} "
        f"{d_success:>10}장  "
        f"{str(d_oom or '-'):>6}장  "
        f"{d_loc or '-'}"
    )

    if b_oom and d_oom:
        gain = d_oom - b_oom

        result_lines.append("")
        result_lines.append(
            f"  → Dynamic ALL이 {gain}장 더 처리 가능"
        )

    elif d_oom is None:
        result_lines.append("")
        result_lines.append(
            f"  → Dynamic ALL은 테스트 범위({total}장) 내 OOM 없음!"
        )

    result_lines.append("=" * 60)

    # ==========================================================
    # 저장
    # ==========================================================
    result_path = os.path.join(
        args.output_dir,
        "oom_result.txt"
    )

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    print(f"\n✅ 결과 저장: {result_path}")

    print("\n" + "\n".join(result_lines))


# =====================================================================
# Entry
# =====================================================================
if __name__ == "__main__":
    parser = get_args_parser()

    args = parser.parse_args()

    main(args)