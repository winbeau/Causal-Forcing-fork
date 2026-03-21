"""CyclicStrategy — phase-bucket anchor storage.

Stores frame anchors in buckets indexed by t mod period.
On collect, returns anchors from the current-phase bucket.
"""
from __future__ import annotations

from collections import deque

import torch

from .base import CollectedAnchor, FrameAnchor


class CyclicStrategy:
    """Cyclic (phase-bucket) middle strategy.

    Args:
        period: Phase period (e.g. 6 for 6-frame cycle).
        bucket_cap: Max anchors per phase bucket.
        dynamic_rope: Whether to remap anchor time to current sync_t.
    """

    def __init__(self, period: int = 6, bucket_cap: int = 1, dynamic_rope: bool = True):
        self.period = max(1, int(period))
        self.bucket_cap = max(1, int(bucket_cap))
        self.dynamic_rope = bool(dynamic_rope)
        # per (batch*head) -> per phase bucket -> deque of FrameAnchor
        self._buckets: list[list[deque[FrameAnchor]]] = []

    def reset(self, num_seq: int) -> None:
        self._buckets = [
            [deque(maxlen=self.bucket_cap) for _ in range(self.period)] for _ in range(num_seq)
        ]

    def update(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        current_t: int,
        t_vals: list[int] | None = None,
    ) -> None:
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return

        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            t_val = t_vals[frame_idx]
            phase = t_val % self.period
            bucket = self._buckets[idx][phase]
            bucket.append(FrameAnchor(
                k=k_seq[start:end].clone(),
                v=v_seq[start:end].clone(),
                pos=pos_seq[start:end].clone(),
                t=t_val,
            ))

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        phase_idx = current_t % self.period
        result: list[CollectedAnchor] = []
        for anchor in self._buckets[idx][phase_idx]:
            t = anchor.t
            # Skip sink overlap (t=0 when sink exists)
            if t <= sink_max_t:
                continue
            # Skip recent overlap
            if t >= recent_min_t:
                continue
            result.append(CollectedAnchor(
                kind="frame",
                t=anchor.t,
                dynamic_rope=self.dynamic_rope,
                k=anchor.k,
                v=anchor.v,
                pos=anchor.pos,
                token_count=anchor.k.shape[0],
            ))
        return result
