import math
import torch
import torch.nn.functional as F


class AdaptiveCacheScheduler:
    """
    Adaptive Cache Scheduling for LiteVGGT.

    핵심 동작:
    - 레이어 0: 항상 재계산 (첫 번째 레이어)
    - 레이어 1~23: 마지막 재계산 이후 토큰 drift를 측정
      - drift > tau  → 재계산, 캐시 갱신
      - drift <= tau → 기존 merge indices 재사용
      - cache_age >= k_max → 강제 재계산

    단일 forward pass 내에서 레이어 간 캐시를 공유합니다.
    (레이어 0이 재계산 → 레이어 1~5 재사용 → 레이어 6이 drift 초과 시 재계산 ...)
    """

    def __init__(
        self,
        num_layers: int = 24,
        tau_base: float = 0.15,
        k_max: int = 6,
        use_layer_wise_tau: bool = False,
    ):
        self.num_layers = num_layers
        self.k_max = k_max

        if use_layer_wise_tau:
            self.tau_l = [
                tau_base * math.exp(-l / num_layers) for l in range(num_layers)
            ]
        else:
            self.tau_l = [tau_base] * num_layers

        # 공유 캐시 상태 (마지막 재계산 시점의 값)
        self.cached_m_u = None        # (merge_fn, unmerge_fn, b_idx)
        self.cached_dst_feat = None   # [B, num_dst, C]
        self.cache_age = 0            # 마지막 재계산 이후 재사용 횟수

    def reset(self):
        """forward pass 시작 시 한 번 호출."""
        self.cached_m_u = None
        self.cached_dst_feat = None
        self.cache_age = 0

    def should_recompute(
        self,
        layer_idx: int,
        x: torch.Tensor,
        b_idx,  # 이전 m_u의 b_idx 또는 None
    ) -> tuple:
        """
        Returns (recompute: bool, reason: str)
        reason: "cold_start" | "forced" | "drift" | "cache_hit"
        """
        # 1. 캐시 없음 (첫 레이어 또는 reset 직후)
        if self.cached_dst_feat is None:
            return True, "cold_start"

        # 2. 레이어 0은 항상 재계산 / 캐시 수명 초과
        if layer_idx == 0 or self.cache_age >= self.k_max:
            return True, "forced"

        # 3. b_idx 없으면 재계산 불가 → 재계산
        if b_idx is None:
            return True, "forced"

        # 4. drift 측정
        try:
            dst_indices = b_idx[0, :, 0]          # [num_dst]
            dst_now = x[:, dst_indices, :]         # [B, num_dst, C]
            cos_sim = F.cosine_similarity(
                dst_now, self.cached_dst_feat, dim=-1
            ).mean()
            drift = 1.0 - cos_sim.item()
        except Exception:
            return True, "forced"

        if drift > self.tau_l[layer_idx]:
            return True, "drift"

        # 캐시 재사용
        self.cache_age += 1
        return False, "cache_hit"

    def update_cache(
        self,
        layer_idx: int,
        m_u: tuple,
        x: torch.Tensor,
        b_idx,
    ):
        """재계산 후 캐시 갱신."""
        self.cached_m_u = m_u
        self.cache_age = 0
        if b_idx is not None:
            try:
                dst_indices = b_idx[0, :, 0]
                self.cached_dst_feat = x[:, dst_indices, :].detach()
            except Exception:
                self.cached_dst_feat = None
        else:
            self.cached_dst_feat = None

    def get_m_u(self, layer_idx: int) -> tuple:
        """캐시된 merge indices 반환."""
        return self.cached_m_u
