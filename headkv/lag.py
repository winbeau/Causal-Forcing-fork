"""LagStrategy — fixed-offset anchor retrieval.

Maintains a sorted history of frame anchors. On collect, returns the
anchor at t = current_t - offset for each configured offset.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .base import CollectedAnchor, FrameAnchor


class LagStrategy:
    """Lag middle strategy.

    Args:
        offsets: List of positive lag offsets (e.g. [6, 12]).
        history_frames: Max number of frame anchors to retain.
        dynamic_rope: Whether to remap anchor time for RoPE.
    """

    def __init__(
        self,
        offsets: list[int] | None = None,
        history_frames: int = 21,
        dynamic_rope: bool = False,
    ):
        raw_offsets = offsets if offsets is not None else [6]
        self.offsets = sorted(set(o for o in raw_offsets if o > 0))
        self.history_frames = max(1, int(history_frames))
        if self.offsets:
            self.history_frames = max(self.history_frames, max(self.offsets) + 1)
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

        anchors = self._anchors[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        if t_vals is None:
            t_start = int(current_t)
            t_vals = list(range(t_start, t_start + num_frames))
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            t_val = t_vals[frame_idx]
            if t_val in anchors:
                del anchors[t_val]
            anchors[t_val] = FrameAnchor(
                k=k_seq[start:end].clone(),
                v=v_seq[start:end].clone(),
                pos=pos_seq[start:end].clone(),
                t=t_val,
            )
            while len(anchors) > self.history_frames:
                anchors.popitem(last=False)

    def collect(
        self,
        idx: int,
        current_t: int,
        recent_min_t: int,
        sink_max_t: int,
    ) -> list[CollectedAnchor]:
        result: list[CollectedAnchor] = []
        for lag in self.offsets:
            target_t = current_t - lag
            if target_t < 0:
                continue
            if target_t <= sink_max_t:
                continue
            if target_t >= recent_min_t:
                continue
            anchor = self._anchors[idx].get(target_t)
            if anchor is not None:
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
