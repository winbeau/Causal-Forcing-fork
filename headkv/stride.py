"""StrideStrategy — evenly-spaced anchor selection.

Stores all frame anchors and returns every `interval`-th frame
between sink and recent windows.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .base import CollectedAnchor, FrameAnchor


class StrideStrategy:
    """Stride middle strategy.

    Args:
        interval: Select every `interval`-th frame (e.g. 6).
        capacity: Max number of stride anchors to keep per sequence.
            -1 means unlimited (store all aligned frames).
            When exceeded, the oldest (smallest t) anchor is evicted (FIFO).
        dynamic_rope: Whether to remap anchor time for RoPE.
    """

    def __init__(self, interval: int = 6, capacity: int = -1, dynamic_rope: bool = True):
        self.interval = max(1, int(interval))
        self.capacity = int(capacity)
        self.dynamic_rope = bool(dynamic_rope)
        # per (batch*head) -> OrderedDict[t -> FrameAnchor]
        self._anchors: list[OrderedDict[int, FrameAnchor]] = []

    def reset(self, num_seq: int) -> None:
        self._anchors = [OrderedDict() for _ in range(num_seq)]

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

        store = self._anchors[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            t_val = t_vals[frame_idx]
            # Only store frames aligned to interval
            if t_val % self.interval != 0:
                continue
            if t_val in store:
                del store[t_val]
            store[t_val] = FrameAnchor(
                k=k_seq[start:end].clone(),
                v=v_seq[start:end].clone(),
                pos=pos_seq[start:end].clone(),
                t=t_val,
            )
            # FIFO eviction: drop oldest anchor when over capacity
            if self.capacity > 0 and len(store) > self.capacity:
                store.popitem(last=False)

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        for t_val, anchor in sorted(self._anchors[idx].items()):
            if t_val <= sink_max_t:
                continue
            if t_val >= recent_min_t:
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

    @staticmethod
    def select_frame_ids(num_frames: int, interval: int = 6) -> list[int]:
        """Return frame indices for stride selection (static utility)."""
        if num_frames <= 0:
            return []
        return sorted(set(f for f in range(0, num_frames, interval)))
