import torch
import os
import csv

# 전역 카운터 (프레임 번호 추적용)
_frame_counter = {"count": 0}

# 전역 로그 저장소
_log_records = []
_log_path = None


def set_log_path(path: str):
    global _log_path
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    _log_path = path
    with open(_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "complexity", "protect_ratio", "stride", "level"])
    print(f"[complexity] 로그 파일 생성: {_log_path}")


def reset_frame_counter():
    global _log_records
    _frame_counter["count"] = 0
    _log_records = []


def flush_log():
    global _log_records, _log_path
    if _log_path is None or len(_log_records) == 0:
        return
    with open(_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(_log_records)
    print(f"\n[complexity] 로그 저장 완료: {len(_log_records)}개 프레임 → {_log_path}")
    _log_records = []


def compute_frame_complexity(info_map: torch.Tensor) -> torch.Tensor:
    N = info_map.shape[0]
    mean_complexity = info_map.view(N, -1).mean(dim=1)
    var_complexity = info_map.view(N, -1).var(dim=1)
    complexity = 0.7 * mean_complexity + 0.3 * var_complexity
    complexity = complexity.clamp(0.0, 1.0)
    return complexity


def get_dynamic_protect_ratio_single(
    info_map_single: torch.Tensor,
    min_ratio: float = 0.05,
    max_ratio: float = 0.20,
    verbose: bool = True,
) -> float:
    global _log_records

    info = info_map_single.float()
    N = info.shape[0]

    flat = info.view(N, -1)
    mean_c = flat.mean(dim=1)
    var_c = flat.var(dim=1)
    complexity = (0.7 * mean_c + 0.3 * var_c).clamp(0.0, 1.0)
    protect_ratios = min_ratio + (max_ratio - min_ratio) * complexity

    for i in range(N):
        frame_num = _frame_counter["count"]
        c = complexity[i].item()
        r = protect_ratios[i].item()

        if c < 0.33:
            level = "단순"
        elif c < 0.66:
            level = "보통"
        else:
            level = "복잡"

        if verbose:
            print(f"  Frame {frame_num:4d} | 복잡도: {c:.4f} | protect_ratio: {r:.4f} | {level}")

        if _log_path is not None:
            _log_records.append([frame_num, round(c, 4), round(r, 4), "-", level])

        _frame_counter["count"] += 1

    return protect_ratios.mean().item()


def get_dynamic_protect_ratios(
    info_map: torch.Tensor,
    min_ratio: float = 0.05,
    max_ratio: float = 0.20,
    verbose: bool = True,
) -> torch.Tensor:
    complexity = compute_frame_complexity(info_map)
    N = info_map.shape[0]
    protect_ratios = min_ratio + (max_ratio - min_ratio) * complexity

    if verbose:
        print("\n=== 프레임별 복잡도 및 protect_ratio ===")
        for i in range(N):
            c = complexity[i].item()
            r = protect_ratios[i].item()
            if c < 0.33:
                level = "단순"
            elif c < 0.66:
                level = "보통"
            else:
                level = "복잡"
            print(f"Frame {i:4d} | 복잡도: {c:.4f} | protect_ratio: {r:.4f} | {level}")
        print(f"\n평균 protect_ratio: {protect_ratios.mean().item():.4f}")
        print(f"최소 protect_ratio: {protect_ratios.min().item():.4f}")
        print(f"최대 protect_ratio: {protect_ratios.max().item():.4f}")
        print("=" * 45)

    return protect_ratios


# =====================================================================
# [새로 추가] 동적 grid stride 결정
# 복잡도가 낮을수록 stride 크게 (Dst 줄여서 압축 UP)
# 복잡도가 높을수록 stride 작게 (Dst 유지해서 품질 보존)
# =====================================================================
def get_dynamic_grid_stride(
    info_map_single: torch.Tensor,
    min_stride: int = 2,
    max_stride: int = 4,
    verbose: bool = True,
) -> int:
    global _log_records

    info = info_map_single.float()
    N = info.shape[0]

    flat = info.view(N, -1)
    mean_c = flat.mean(dim=1)
    var_c = flat.var(dim=1)
    complexity = (0.7 * mean_c + 0.3 * var_c).clamp(0.0, 1.0)
    avg_complexity = complexity.mean().item()

    # complexity=0 → stride=max, complexity=1 → stride=min
    stride_float = max_stride - (max_stride - min_stride) * avg_complexity
    stride = max(min_stride, min(max_stride, round(stride_float)))

    if avg_complexity < 0.33:
        level = "단순"
    elif avg_complexity < 0.66:
        level = "보통"
    else:
        level = "복잡"

    if verbose:
        print(f"  [Grid] 평균 복잡도: {avg_complexity:.4f} ({level}) → stride: {stride}×{stride}")

    if _log_path is not None:
        for i in range(N):
            frame_num = _frame_counter["count"] + i
            c = complexity[i].item()
            _log_records.append([frame_num, round(c, 4), "-", stride, level])

    return int(stride)