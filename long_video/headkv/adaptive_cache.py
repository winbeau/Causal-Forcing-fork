import torch
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from time import perf_counter

from .cache import HeadKVCache
from .selectors import (
    _topk_mask,
    _normalize_scores,
    ThreeDIVCSelector,
    SemanticValueSelector,
)
from .rope import (
    apply_rope_to_flat_k,
    map_dynamic_pos_time,
    map_sink_time,
)


class AdaptiveKVCache(HeadKVCache):
    post_prune_rope = True

    def __init__(
        self,
        config,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        is_i2v: bool = False,
        context_len: int = 0,
        sink_len: int = 0,
        tail_len: int = 32760,
        ivc_ratio: float = 0.1,
        semantic_ratio: float = 0.1,
        update_interval: int = 1,
        seed_ratio: float = 0.01,
        trajectory_ratio: float = 0.0,
        trajectory_weight: float = 0.0,
        history_frame_quota: int = 0,
        history_quota_ivc_ratio: float = 0.0,
        post_train_stabilize_t: int = -1,
        post_train_trajectory_scale: float = 1.0,
        post_train_history_ivc_ratio: float = -1.0,
        prune_sink: bool = False,
        prune_tail: bool = False,
        aggressive_all: bool = False,
        sink_grid_decoupling: bool = False,
        decoupled_sink_tokens: int = 0,
        decoupled_sink_time_lag: int = 0,
        sink_time_mapping_mode: str = "lag",
        sink_time_clamp_min: int = 18,
        sink_time_clamp_max: int = 21,
        history_time_mapping_mode: str = "none",
        history_relative_t_max: int = 21,
        history_time_soft_factor: float = 0.5,
        osc_full_kv_retention: bool = False,
        periodic_peak_mask: bool = False,
        periodic_peak_period: int = 6,
        periodic_peak_offsets: list[int] | None = None,
        periodic_peak_start_t: int = 6,
        periodic_peak_only_oscillating: bool = True,
        use_osc_frame_mode: bool = False,
        phase_period: int = 6,
        phase_bucket_capacity_frames: int = 1,
        local_tail_frames: int = 4,
        phase_sink_for_osc_only: bool = True,
        phase_sink_dynamic_rope: bool = True,
        use_osc_lag_mode: bool = False,
        osc_lag_offsets_frames: list[int] | None = None,
        osc_lag_history_frames: int = 21,
        osc_lag_dynamic_rope: bool = False,
        disable_first_sink_for_osc_heads: bool = False,
        use_stable_head_policies: bool = True,
        stable_sink_frames: int | None = None,
        osc_sink_frames: int | None = None,
        stable_recent_frames: int | None = None,
        use_af_head_policies: bool = False,
        af_recent_frames_map: dict | None = None,
        af_phase_bucket_map: dict | None = None,
        af_lag_offsets_map: dict | None = None,
        af_sink_frames_map: dict | None = None,
        af_stride_enabled_map: dict | None = None,
        label_recent_frames_map: dict | None = None,
        label_phase_bucket_map: dict | None = None,
        label_lag_offsets_map: dict | None = None,
        label_sink_frames_map: dict | None = None,
        label_stride_enabled_map: dict | None = None,
        capture_frame_id_mode: str = "mapped",
        readout_cache_enabled: bool = True,
        prompt_value_cache_enabled: bool = False,
    ):
        super().__init__(
            config=config,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            layer_idx=layer_idx,
            is_i2v=is_i2v,
            context_len=context_len,
            prompt_value_cache_enabled=prompt_value_cache_enabled,
        )
        self.sink_len = max(0, int(sink_len))
        self.tail_len = max(0, int(tail_len))
        self.ivc_ratio = float(ivc_ratio)
        self.semantic_ratio = float(semantic_ratio)
        self.update_interval = max(1, int(update_interval))
        self.seed_ratio = float(seed_ratio)
        self.trajectory_ratio = float(trajectory_ratio)
        self.trajectory_weight = float(trajectory_weight)
        self.history_frame_quota = max(0, int(history_frame_quota))
        self.history_quota_ivc_ratio = max(0.0, min(1.0, float(history_quota_ivc_ratio)))
        self.post_train_stabilize_t = int(post_train_stabilize_t)
        self.post_train_trajectory_scale = max(0.0, float(post_train_trajectory_scale))
        self.post_train_history_ivc_ratio = float(post_train_history_ivc_ratio)
        self.prune_sink = bool(prune_sink)
        self.prune_tail = bool(prune_tail)
        self.aggressive_all = bool(aggressive_all)
        self.sink_grid_decoupling = bool(sink_grid_decoupling)
        self.decoupled_sink_tokens = max(0, int(decoupled_sink_tokens))
        self.decoupled_sink_time_lag = max(0, int(decoupled_sink_time_lag))
        self.sink_time_mapping_mode = str(sink_time_mapping_mode)
        self.sink_time_clamp_min = max(0, int(sink_time_clamp_min))
        self.sink_time_clamp_max = max(self.sink_time_clamp_min, int(sink_time_clamp_max))
        self.history_time_mapping_mode = str(history_time_mapping_mode)
        self.history_relative_t_max = max(0, int(history_relative_t_max))
        self.history_time_soft_factor = max(0.0, min(1.0, float(history_time_soft_factor)))
        self.osc_full_kv_retention = bool(osc_full_kv_retention)
        self.periodic_peak_mask = bool(periodic_peak_mask)
        self.periodic_peak_period = max(1, int(periodic_peak_period))
        offs = [0, 1] if periodic_peak_offsets is None else periodic_peak_offsets
        normalized_offs = []
        for o in offs:
            try:
                normalized_offs.append(int(o) % self.periodic_peak_period)
            except (TypeError, ValueError):
                continue
        self.periodic_peak_offsets = sorted(set(normalized_offs)) if normalized_offs else [0, 1]
        self.periodic_peak_start_t = max(0, int(periodic_peak_start_t))
        self.periodic_peak_only_oscillating = bool(periodic_peak_only_oscillating)
        self.use_osc_frame_mode = bool(use_osc_frame_mode)
        self.phase_period = max(1, int(phase_period))
        self.phase_bucket_capacity_frames = max(0, int(phase_bucket_capacity_frames))
        self.local_tail_frames = max(1, int(local_tail_frames))
        self.phase_sink_for_osc_only = bool(phase_sink_for_osc_only)
        self.phase_sink_dynamic_rope = bool(phase_sink_dynamic_rope)
        lag_offs = [6] if osc_lag_offsets_frames is None else osc_lag_offsets_frames
        normalized_lag_offs = []
        for off in lag_offs:
            try:
                off_int = int(off)
            except (TypeError, ValueError):
                continue
            if off_int > 0:
                normalized_lag_offs.append(off_int)
        self.osc_lag_offsets_frames = sorted(set(normalized_lag_offs))
        self.use_osc_lag_mode = bool(use_osc_lag_mode and len(self.osc_lag_offsets_frames) > 0)
        self.osc_lag_history_frames = max(1, int(osc_lag_history_frames))
        if self.use_osc_lag_mode:
            self.osc_lag_history_frames = max(self.osc_lag_history_frames, max(self.osc_lag_offsets_frames) + 1)
        self.osc_lag_dynamic_rope = bool(osc_lag_dynamic_rope)
        self.disable_first_sink_for_osc_heads = bool(disable_first_sink_for_osc_heads)
        self.use_stable_head_policies = bool(use_stable_head_policies)
        self.stable_sink_frames = (
            None if stable_sink_frames is None else max(1, int(stable_sink_frames))
        )
        self.osc_sink_frames = (
            None if osc_sink_frames is None else max(1, int(osc_sink_frames))
        )
        self.stable_recent_frames = (
            None if stable_recent_frames is None else max(1, int(stable_recent_frames))
        )
        self.use_af_head_policies = bool(use_af_head_policies)
        # For attention visualization only:
        # - mapped: report RoPE-mapped time ids
        # - physical: report original physical frame ids
        mode = str(capture_frame_id_mode).strip().lower()
        if mode not in {"mapped", "physical"}:
            mode = "mapped"
        self.capture_frame_id_mode = mode
        self.readout_cache_enabled = bool(readout_cache_enabled)
        self._base_tail_len = self.tail_len
        max_cap = max(self.capacities) if self.capacities else 0
        min_cap = min(self.capacities) if self.capacities else 0
        self.max_capacity = max_cap
        self.head_labels = (
            config.get_layer_labels(layer_idx)
            if hasattr(config, "get_layer_labels")
            else [1] * self.num_heads
        )
        self.osc_head_flags = [int(lbl) == -1 for lbl in self.head_labels]
        self.af_head_groups = list(getattr(self, "af_group_row", [""] * self.num_heads))
        self.af_recent_frames_map = self._build_af_recent_frames_map(af_recent_frames_map)
        self.af_phase_bucket_map = self._build_af_phase_bucket_map(af_phase_bucket_map)
        self.af_lag_offsets_map = self._build_af_lag_offsets_map(af_lag_offsets_map)
        self.af_sink_frames_map = self._build_af_sink_frames_map(af_sink_frames_map)
        self.af_stride_enabled_map = self._build_af_stride_enabled_map(af_stride_enabled_map)
        self.label_recent_frames_map = self._build_label_recent_frames_map(label_recent_frames_map)
        self.label_phase_bucket_map = self._build_label_phase_bucket_map(label_phase_bucket_map)
        self.label_lag_offsets_map = self._build_label_lag_offsets_map(label_lag_offsets_map)
        self.label_sink_frames_map = self._build_label_sink_frames_map(label_sink_frames_map)
        self.label_stride_enabled_map = self._build_label_stride_enabled_map(label_stride_enabled_map)
        if self.sink_grid_decoupling and min_cap < max_cap:
            # In class-aware configs (e.g. -1 oscillating, 1 stable), reduced-capacity heads
            # are treated as oscillating heads and receive sink-grid decoupling.
            if any(self.osc_head_flags):
                self.decouple_head_flags = self.osc_head_flags.copy()
            else:
                self.decouple_head_flags = [cap < max_cap for cap in self.capacities]
        else:
            # If all capacities are equal, keep behavior backward compatible.
            self.decouple_head_flags = [self.sink_grid_decoupling] * self.num_heads

        self.static_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self.dynamic_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self.update_step = 0
        self.prompt_v: torch.Tensor | None = None
        self.last_flat_pos_ids: torch.Tensor | None = None
        # Workspace buffers for get_decoupled_flat_kv_and_frames (B2 optimization)
        self._ws_k: torch.Tensor | None = None
        self._ws_v: torch.Tensor | None = None
        self._ws_frame_ids: torch.Tensor | None = None
        self._ws_cu_seqlens: torch.Tensor | None = None
        self._ws_rope_pos: torch.Tensor | None = None
        self._readout_cache_valid = False
        self._readout_cache_current_start: int | None = None
        self._readout_cache_sync_t_raw: int | None = None
        self._readout_cache_total_len = 0
        self._readout_cache_max_seqlen = 0
        self._readout_cache_frame_seqlen = 0
        self._readout_cache_tail_dirty = False
        self._readout_static_specs: list[tuple[int, int] | None] = [None] * (batch_size * num_heads)
        self._readout_tail_specs: list[tuple[int, int] | None] = [None] * (batch_size * num_heads)
        self._current_block_token_len: list[int] = [0] * (batch_size * num_heads)
        self._dyn_store_k: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_v: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_pos: list[torch.Tensor | None] = [None] * (batch_size * num_heads)
        self._dyn_store_start: list[int] = [0] * (batch_size * num_heads)
        self._dyn_store_len: list[int] = [0] * (batch_size * num_heads)
        self._profile_enabled = False
        self._profile_stats = {
            "update_ms": 0.0,
            "collect_ms": 0.0,
            "pack_ms": 0.0,
            "rope_ms": 0.0,
        }
        self.ivc_selector = ThreeDIVCSelector()
        self.semantic_selector = SemanticValueSelector()
        self._init_cyclic_anchor_storage()
        # Initialize compositions' middle strategies if available
        if self.compositions_row is not None:
            num_seq = batch_size * num_heads
            for comp in self.compositions_row:
                if comp.has_middle:
                    comp.reset_all(num_seq)

    def set_profile_enabled(self, enabled: bool) -> None:
        self._profile_enabled = bool(enabled)
        if not enabled:
            self.reset_profile_stats()

    def reset_profile_stats(self) -> None:
        self._profile_stats = {
            "update_ms": 0.0,
            "collect_ms": 0.0,
            "pack_ms": 0.0,
            "rope_ms": 0.0,
        }

    def pop_profile_stats(self) -> dict[str, float]:
        stats = dict(self._profile_stats)
        self.reset_profile_stats()
        return stats

    def _record_profile(self, key: str, start_time: float) -> None:
        if not self._profile_enabled:
            return
        self._profile_stats[key] += (perf_counter() - start_time) * 1000.0

    def _init_cyclic_anchor_storage(self) -> None:
        num_seq = self.batch_size * self.num_heads
        # per (batch, head) -> per phase bucket -> deque of frame-level anchors
        self.cyclic_buckets: list[list[deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]]] = [
            [deque() for _ in range(self.phase_period)] for _ in range(num_seq)
        ]
        # per (batch, head) -> OrderedDict[t -> frame-level anchor]
        self.lag_anchor_frames: list[OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]] = [
            OrderedDict() for _ in range(num_seq)
        ]

    def _sync_dynamic_views(self, idx: int) -> None:
        store_k = self._dyn_store_k[idx]
        if store_k is None:
            self.dynamic_k[idx] = None
            self.dynamic_v[idx] = None
            self.dynamic_pos[idx] = None
            return

        start = self._dyn_store_start[idx]
        end = start + self._dyn_store_len[idx]
        self.dynamic_k[idx] = self._dyn_store_k[idx][start:end]
        self.dynamic_v[idx] = self._dyn_store_v[idx][start:end]
        self.dynamic_pos[idx] = self._dyn_store_pos[idx][start:end]

    def _set_dynamic_store(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        reserve_extra: int = 0,
    ) -> None:
        reserve = max(0, int(reserve_extra))
        length = int(k_seq.shape[0])
        capacity = max(length, length + reserve)
        self._dyn_store_k[idx] = torch.empty((capacity, self.head_dim), device=k_seq.device, dtype=k_seq.dtype)
        self._dyn_store_v[idx] = torch.empty((capacity, self.head_dim), device=v_seq.device, dtype=v_seq.dtype)
        self._dyn_store_pos[idx] = torch.empty((capacity, 3), device=pos_seq.device, dtype=pos_seq.dtype)
        if length > 0:
            self._dyn_store_k[idx][:length] = k_seq
            self._dyn_store_v[idx][:length] = v_seq
            self._dyn_store_pos[idx][:length] = pos_seq
        self._dyn_store_start[idx] = 0
        self._dyn_store_len[idx] = length
        self._sync_dynamic_views(idx)

    def _set_dynamic_empty(
        self,
        idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        empty_k = torch.empty((0, self.head_dim), device=device, dtype=dtype)
        empty_v = torch.empty((0, self.head_dim), device=device, dtype=dtype)
        empty_pos = torch.empty((0, 3), device=device, dtype=torch.long)
        self._set_dynamic_store(idx, empty_k, empty_v, empty_pos, reserve_extra=0)

    def _ensure_dynamic_capacity(
        self,
        idx: int,
        append_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        store_k = self._dyn_store_k[idx]
        if store_k is None:
            alloc = max(append_len + max(16, append_len // 2), append_len)
            self._dyn_store_k[idx] = torch.empty((alloc, self.head_dim), device=device, dtype=dtype)
            self._dyn_store_v[idx] = torch.empty((alloc, self.head_dim), device=device, dtype=dtype)
            self._dyn_store_pos[idx] = torch.empty((alloc, 3), device=device, dtype=torch.long)
            self._dyn_store_start[idx] = 0
            self._dyn_store_len[idx] = 0
            return

        start = self._dyn_store_start[idx]
        length = self._dyn_store_len[idx]
        end = start + length
        if store_k.shape[0] - end >= append_len:
            return

        total_free = store_k.shape[0] - length
        if start > 0 and total_free >= append_len:
            if length > 0:
                self._dyn_store_k[idx][:length] = self._dyn_store_k[idx][start:end].clone()
                self._dyn_store_v[idx][:length] = self._dyn_store_v[idx][start:end].clone()
                self._dyn_store_pos[idx][:length] = self._dyn_store_pos[idx][start:end].clone()
            self._dyn_store_start[idx] = 0
            return

        new_capacity = max(length + append_len, int(store_k.shape[0] * 1.5), length + append_len + 64)
        new_k = torch.empty((new_capacity, self.head_dim), device=device, dtype=dtype)
        new_v = torch.empty((new_capacity, self.head_dim), device=device, dtype=dtype)
        new_pos = torch.empty((new_capacity, 3), device=device, dtype=torch.long)
        if length > 0:
            new_k[:length] = self._dyn_store_k[idx][start:end]
            new_v[:length] = self._dyn_store_v[idx][start:end]
            new_pos[:length] = self._dyn_store_pos[idx][start:end]
        self._dyn_store_k[idx] = new_k
        self._dyn_store_v[idx] = new_v
        self._dyn_store_pos[idx] = new_pos
        self._dyn_store_start[idx] = 0

    def _append_dynamic(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
    ) -> None:
        append_len = int(k_seq.shape[0])
        if append_len <= 0:
            self._sync_dynamic_views(idx)
            return
        self._ensure_dynamic_capacity(idx, append_len, device=k_seq.device, dtype=k_seq.dtype)
        start = self._dyn_store_start[idx]
        length = self._dyn_store_len[idx]
        end = start + length
        self._dyn_store_k[idx][end:end + append_len] = k_seq
        self._dyn_store_v[idx][end:end + append_len] = v_seq
        self._dyn_store_pos[idx][end:end + append_len] = pos_seq
        self._dyn_store_len[idx] = length + append_len
        self._sync_dynamic_views(idx)

    def _overwrite_dynamic_tail(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
    ) -> None:
        overwrite_len = int(k_seq.shape[0])
        if overwrite_len <= 0:
            self._sync_dynamic_views(idx)
            return
        if self._dyn_store_k[idx] is None or self._dyn_store_len[idx] < overwrite_len:
            self._set_dynamic_store(idx, k_seq, v_seq, pos_seq, reserve_extra=max(16, overwrite_len // 2))
            return
        start = self._dyn_store_start[idx]
        end = start + self._dyn_store_len[idx]
        tail_start = end - overwrite_len
        self._dyn_store_k[idx][tail_start:end] = k_seq
        self._dyn_store_v[idx][tail_start:end] = v_seq
        self._dyn_store_pos[idx][tail_start:end] = pos_seq
        self._sync_dynamic_views(idx)

    def _keep_dynamic_suffix(self, idx: int, keep_len: int) -> None:
        if self._dyn_store_k[idx] is None:
            return
        keep = max(0, int(keep_len))
        length = self._dyn_store_len[idx]
        if keep >= length:
            self._sync_dynamic_views(idx)
            return
        self._dyn_store_start[idx] += length - keep
        self._dyn_store_len[idx] = keep
        self._sync_dynamic_views(idx)

    @staticmethod
    def _normalize_af_group_key(key: object) -> str:
        raw = str(key).strip().upper()
        if not raw:
            return ""
        if raw in {"A", "B", "C", "D", "E", "F"}:
            return raw
        if raw.startswith(("A_", "B_", "C_", "D_", "E_", "F_")):
            return raw[0]
        return ""

    @staticmethod
    def _normalize_label_key(key: object) -> str:
        raw = str(key).strip()
        if not raw:
            return ""
        try:
            return str(int(raw))
        except (TypeError, ValueError):
            return raw

    @staticmethod
    def _map_items(user_map: Mapping | None):
        if not isinstance(user_map, Mapping):
            return ()
        return user_map.items()

    @staticmethod
    def _as_sequence(value: object) -> list[object]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(value)
        return [value]

    def _build_label_recent_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_phase_bucket_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(0, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_lag_offsets_map(self, user_map: Mapping | None) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            vals = self._as_sequence(val)
            offs: list[int] = []
            for item in vals:
                try:
                    off = int(item)
                except (TypeError, ValueError):
                    continue
                if off > 0:
                    offs.append(off)
            out[label] = sorted(set(offs))
        return out

    def _build_label_sink_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            try:
                out[label] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_label_stride_enabled_map(self, user_map: Mapping | None) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, val in self._map_items(user_map):
            label = self._normalize_label_key(key)
            if not label:
                continue
            out[label] = bool(val)
        return out

    def _build_af_recent_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out = {"A": 4, "B": 3, "C": 4, "D": 3, "E": 2, "F": 5}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_phase_bucket_map(self, user_map: Mapping | None) -> dict[str, int]:
        out = {"A": 0, "B": 1, "C": 1, "D": 1, "E": 0, "F": 0}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(0, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_lag_offsets_map(self, user_map: Mapping | None) -> dict[str, list[int]]:
        out = {"A": [], "B": [], "C": [], "D": [6], "E": [], "F": []}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            offs = []
            vals = self._as_sequence(val)
            for item in vals:
                try:
                    off = int(item)
                except (TypeError, ValueError):
                    continue
                if off > 0:
                    offs.append(off)
            out[group] = sorted(set(offs))
        return out

    def _build_af_sink_frames_map(self, user_map: Mapping | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            try:
                out[group] = max(1, int(val))
            except (TypeError, ValueError):
                continue
        return out

    def _build_af_stride_enabled_map(self, user_map: Mapping | None) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for key, val in self._map_items(user_map):
            group = self._normalize_af_group_key(key)
            if not group:
                continue
            out[group] = bool(val)
        return out

    def _af_group(self, head_idx: int) -> str:
        if head_idx < 0 or head_idx >= len(self.af_head_groups):
            return ""
        return self._normalize_af_group_key(self.af_head_groups[head_idx])

    def _head_label_key(self, head_idx: int) -> str:
        if head_idx < 0 or head_idx >= len(self.head_labels):
            return ""
        return self._normalize_label_key(self.head_labels[head_idx])

    def _head_sink_frames(self, head_idx: int) -> int | None:
        label = self._head_label_key(head_idx)
        if label in self.label_sink_frames_map:
            return max(1, int(self.label_sink_frames_map[label]))
        if self.use_af_head_policies and self.af_sink_frames_map:
            group = self._af_group(head_idx)
            if group in self.af_sink_frames_map:
                return max(1, int(self.af_sink_frames_map[group]))
        if self.osc_head_flags[head_idx]:
            if self.osc_sink_frames is not None:
                return max(1, int(self.osc_sink_frames))
            return None
        if self.stable_sink_frames is not None:
            return max(1, int(self.stable_sink_frames))
        return None

    def _has_explicit_recent_override(self, head_idx: int) -> bool:
        label = self._head_label_key(head_idx)
        if label in self.label_recent_frames_map:
            return True
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_recent_frames_map:
                return True
        return self.stable_recent_frames is not None

    def _head_recent_frames(self, head_idx: int) -> int:
        label = self._head_label_key(head_idx)
        if label in self.label_recent_frames_map:
            return max(1, int(self.label_recent_frames_map[label]))
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_recent_frames_map:
                return max(1, int(self.af_recent_frames_map[group]))
        if not self.osc_head_flags[head_idx] and self.stable_recent_frames is not None:
            return max(1, int(self.stable_recent_frames))
        return self.local_tail_frames

    def _head_phase_bucket_capacity(self, head_idx: int) -> int:
        label = self._head_label_key(head_idx)
        if label in self.label_phase_bucket_map:
            return max(0, int(self.label_phase_bucket_map[label]))
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_phase_bucket_map:
                return max(0, int(self.af_phase_bucket_map[group]))
        return self.phase_bucket_capacity_frames

    def _head_lag_offsets(self, head_idx: int) -> list[int]:
        label = self._head_label_key(head_idx)
        if label in self.label_lag_offsets_map:
            return list(self.label_lag_offsets_map[label])
        if self.use_af_head_policies:
            group = self._af_group(head_idx)
            if group in self.af_lag_offsets_map:
                return list(self.af_lag_offsets_map[group])
            return []
        if not self.use_osc_lag_mode:
            return []
        return list(self.osc_lag_offsets_frames)

    def _is_phase_sink_head(self, head_idx: int) -> bool:
        if self.use_af_head_policies:
            return self._head_phase_bucket_capacity(head_idx) > 0
        if not self.phase_sink_for_osc_only:
            return True
        return self.osc_head_flags[head_idx]

    def _update_cyclic_anchors(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        t_start: int | None = None,
    ) -> None:
        if not self.use_osc_frame_mode:
            return
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return
        head_idx = idx % self.num_heads
        bucket_cap = self._head_phase_bucket_capacity(head_idx)
        if bucket_cap <= 0:
            return
        if not self._is_phase_sink_head(head_idx):
            return

        num_frames = k_seq.shape[0] // frame_seqlen
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            frame_pos = pos_seq[start:end]
            if frame_pos.shape[0] != frame_seqlen:
                continue
            t_val = int(t_start + frame_idx) if t_start is not None else int(frame_pos[0, 0].item())
            phase = t_val % self.phase_period
            bucket = self.cyclic_buckets[idx][phase]
            bucket.append(
                (
                    k_seq[start:end].clone(),
                    v_seq[start:end].clone(),
                    frame_pos.clone(),
                    t_val,
                )
            )
            while len(bucket) > bucket_cap:
                bucket.popleft()

    def _update_lag_anchors(
        self,
        idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
        t_start: int | None = None,
    ) -> None:
        if not self.use_osc_lag_mode and not self.use_af_head_policies:
            return
        if frame_seqlen <= 0 or k_seq.shape[0] < frame_seqlen:
            return
        if k_seq.shape[0] % frame_seqlen != 0:
            return
        head_idx = idx % self.num_heads
        if not self._is_phase_sink_head(head_idx):
            return
        lag_offsets = self._head_lag_offsets(head_idx)
        if len(lag_offsets) == 0:
            return

        anchors = self.lag_anchor_frames[idx]
        num_frames = k_seq.shape[0] // frame_seqlen
        for frame_idx in range(num_frames):
            start = frame_idx * frame_seqlen
            end = start + frame_seqlen
            frame_pos = pos_seq[start:end]
            if frame_pos.shape[0] != frame_seqlen:
                continue
            t_val = int(t_start + frame_idx) if t_start is not None else int(frame_pos[0, 0].item())
            if t_val in anchors:
                del anchors[t_val]
            anchors[t_val] = (
                k_seq[start:end].clone(),
                v_seq[start:end].clone(),
                frame_pos.clone(),
                t_val,
            )
            while len(anchors) > self.osc_lag_history_frames:
                anchors.popitem(last=False)

    @staticmethod
    def _find_anchor_by_t(
        anchors,
        target_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        if isinstance(anchors, Mapping):
            return anchors.get(target_t)
        for item in reversed(anchors):
            if item[3] == target_t:
                return item
        return None

    def _stable_strategy_kind(self, head_idx: int) -> str | None:
        label = self._head_label_key(head_idx)
        if label in self.label_stride_enabled_map:
            if self.osc_head_flags[head_idx]:
                return None
            return "stride" if self.label_stride_enabled_map[label] else "recent_only"
        # AF per-group stride override
        if self.use_af_head_policies and self.af_stride_enabled_map:
            group = self._af_group(head_idx)
            if group and self.af_stride_enabled_map.get(group, False):
                return "stride"
        if not self.use_stable_head_policies:
            return None
        if self.osc_head_flags[head_idx]:
            return None
        if getattr(self, "policies_row", None) is None or head_idx >= len(self.policies_row):
            return None
        impl = self.policies_row[head_idx]
        kind = getattr(impl, "policy_type", None)
        if kind in {"stride", "recent_only"}:
            return kind
        return None

    def _stride_frame_ids(self, head_idx: int, kind: str, num_frames: int) -> list[int]:
        if num_frames <= 0:
            return []

        # Optional explicit stable-tail policy:
        # - stride: source-like periodic keep (start at frame idx 3, every phase_period)
        #            + recent K
        # - recent_only: recent K only
        # This supports patterns like "sink3 + every6 (before recent) + recent4".
        if self._has_explicit_recent_override(head_idx):
            keep: list[int] = []
            recent = min(num_frames, self._head_recent_frames(head_idx))
            recent_start = max(0, num_frames - recent)
            keep.extend(range(recent_start, num_frames))
            if kind == "stride":
                step = max(1, int(self.phase_period))
                f_idx = 0
                while f_idx < recent_start:
                    keep.append(f_idx)
                    f_idx += step
            return sorted(set(keep))

        # Backward-compatible default stable policy behavior.
        keep: list[int] = []
        head_frames = min(3, num_frames)
        for f_idx in range(head_frames):
            keep.append(f_idx)
        last = num_frames - 1
        if kind == "stride" and num_frames > 4:
            f_idx = 3
            while f_idx < last:
                keep.append(f_idx)
                f_idx += 6
        if last not in keep:
            keep.append(last)
        return sorted(set(keep))

    def _apply_stable_strategy(
        self,
        head_idx: int,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        frame_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kind = self._stable_strategy_kind(head_idx)
        if kind is None:
            return k_seq, v_seq, pos_seq
        if frame_seqlen <= 0 or pos_seq.shape[0] < frame_seqlen:
            return k_seq, v_seq, pos_seq
        # Build ordered unique frame timeline from pos ids.
        t_vals = pos_seq[:, 0]
        uniq_t = torch.unique(t_vals, sorted=True)
        num_frames = int(uniq_t.shape[0])
        if num_frames <= 1:
            return k_seq, v_seq, pos_seq

        keep_f = self._stride_frame_ids(head_idx, kind, num_frames=num_frames)
        keep_t = set(int(uniq_t[i].item()) for i in keep_f)
        keep_t_tensor = torch.tensor(sorted(keep_t), device=t_vals.device, dtype=t_vals.dtype)
        keep_mask = torch.isin(t_vals, keep_t_tensor)
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            return k_seq, v_seq, pos_seq
        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

    def _map_sink_time(self, sync_t_raw: int) -> int:
        return map_sink_time(
            sync_t_raw,
            sink_time_mapping_mode=self.sink_time_mapping_mode,
            sink_time_clamp_min=self.sink_time_clamp_min,
            sink_time_clamp_max=self.sink_time_clamp_max,
            decoupled_sink_time_lag=self.decoupled_sink_time_lag,
        )

    def _map_dynamic_pos_time(self, dyn_pos: torch.Tensor, current_t: int) -> torch.Tensor:
        return map_dynamic_pos_time(
            dyn_pos,
            current_t=current_t,
            history_time_mapping_mode=self.history_time_mapping_mode,
            history_relative_t_max=self.history_relative_t_max,
            history_time_soft_factor=self.history_time_soft_factor,
        )

    def _capture_sink_if_needed(
        self,
        idx: int,
        head_idx: int,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        p_in: torch.Tensor,
        current_start: int | None,
        overwrite: bool,
        freqs: torch.Tensor | None = None,
        prompt_head: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For T2V decoupling mode, keep the first sink_len tokens in a static bucket (raw, unrotated).
        The sink can be refreshed on overwrite at the same start position (e.g. clean pass).
        """
        if self.is_i2v:
            return k_in, v_in, p_in
        if not self.sink_grid_decoupling:
            return k_in, v_in, p_in
        head_idx = idx % self.num_heads
        if not self.decouple_head_flags[head_idx]:
            return k_in, v_in, p_in
        frame_seqlen = int(self.frame_seq_length) if self.frame_seq_length is not None else 0
        if frame_seqlen <= 0 and p_in.shape[0] > 0:
            first_t = int(p_in[0, 0].item())
            frame_seqlen = int((p_in[:, 0] == first_t).sum().item())

        sink_len = self.sink_len
        if frame_seqlen > 0:
            sink_frames = self._head_sink_frames(head_idx)
            if sink_frames is not None:
                sink_len = sink_frames * frame_seqlen

        if sink_len <= 0:
            return k_in, v_in, p_in
        if current_start is None or current_start != 0:
            return k_in, v_in, p_in
        if self.disable_first_sink_for_osc_heads and self.osc_head_flags[head_idx]:
            return k_in, v_in, p_in

        take = min(sink_len, k_in.shape[0])
        if take <= 0:
            return k_in, v_in, p_in

        sink_k = k_in[:take]
        sink_v = v_in[:take]
        sink_p = p_in[:take]
        if self.decoupled_sink_tokens > 0 and take > self.decoupled_sink_tokens:
            budget = self.decoupled_sink_tokens
            if freqs is not None:
                sink_mask = self._ranked_select(
                    pos_seg=sink_p,
                    v_seg=sink_v,
                    budget=budget,
                    freqs=freqs,
                    prompt_head=prompt_head,
                    apply_selection=True,
                )
                select_idx = torch.nonzero(sink_mask, as_tuple=False).squeeze(1).sort().values
            else:
                select_idx = torch.linspace(
                    0,
                    take - 1,
                    steps=budget,
                    device=k_in.device,
                ).round().to(torch.long)
            sink_k = sink_k[select_idx]
            sink_v = sink_v[select_idx]
            sink_p = sink_p[select_idx]

        should_write_static = (self.static_k[idx] is None) or overwrite
        if should_write_static:
            self.static_k[idx] = sink_k.clone()
            self.static_v[idx] = sink_v.clone()
            self.static_pos[idx] = sink_p.clone()

        return k_in[take:], v_in[take:], p_in[take:]

    def _periodic_peak_local_mask(self, t_vals: torch.Tensor) -> torch.Tensor:
        if not self.periodic_peak_mask or t_vals.numel() == 0:
            return torch.zeros_like(t_vals, dtype=torch.bool)
        valid = t_vals >= self.periodic_peak_start_t
        rel = (t_vals - self.periodic_peak_start_t).remainder(self.periodic_peak_period)
        mask = torch.zeros_like(valid, dtype=torch.bool)
        for off in self.periodic_peak_offsets:
            mask |= (rel == off)
        return valid & mask

    def set_prompt_values(self, prompt_v: torch.Tensor | None) -> None:
        self.prompt_v = prompt_v

    def _effective_selection_params(self, pos_seg: torch.Tensor) -> tuple[float, float, float]:
        traj_ratio = self.trajectory_ratio
        traj_weight = self.trajectory_weight
        quota_ivc_ratio = self.history_quota_ivc_ratio
        if pos_seg.numel() == 0:
            return traj_ratio, traj_weight, quota_ivc_ratio

        if self.post_train_stabilize_t >= 0:
            current_t = int(pos_seg[:, 0].max().item())
            if current_t >= self.post_train_stabilize_t:
                traj_ratio = traj_ratio * self.post_train_trajectory_scale
                traj_weight = traj_weight * self.post_train_trajectory_scale
                if self.post_train_history_ivc_ratio >= 0.0:
                    quota_ivc_ratio = max(quota_ivc_ratio, min(1.0, self.post_train_history_ivc_ratio))
        return traj_ratio, traj_weight, quota_ivc_ratio

    def reset(self):
        super().reset()
        self.static_pos = [None] * (self.batch_size * self.num_heads)
        self.dynamic_pos = [None] * (self.batch_size * self.num_heads)
        self.update_step = 0
        self.last_flat_pos_ids = None
        # Clear workspace buffers
        self._ws_k = None
        self._ws_v = None
        self._ws_frame_ids = None
        self._ws_cu_seqlens = None
        self._ws_rope_pos = None
        num_seq = self.batch_size * self.num_heads
        self._current_block_token_len = [0] * num_seq
        self._dyn_store_k = [None] * num_seq
        self._dyn_store_v = [None] * num_seq
        self._dyn_store_pos = [None] * num_seq
        self._dyn_store_start = [0] * num_seq
        self._dyn_store_len = [0] * num_seq
        self._invalidate_readout_cache()
        self.reset_profile_stats()
        self._init_cyclic_anchor_storage()
        # Reset compositions' middle strategies if available
        if self.compositions_row is not None:
            num_seq = self.batch_size * self.num_heads
            for comp in self.compositions_row:
                if comp.has_middle:
                    comp.reset_all(num_seq)

    def _build_pos_ids(self, grid_sizes: torch.Tensor, seq_len: int, current_start: int, device: torch.device) -> torch.Tensor:
        pos = torch.zeros((self.batch_size, seq_len, 3), dtype=torch.long, device=device)
        for b in range(self.batch_size):
            f, h, w = [int(x) for x in grid_sizes[b].tolist()]
            frame_seqlen = max(1, h * w)
            start_frame = current_start // frame_seqlen
            idx = torch.arange(seq_len, device=device, dtype=torch.long)
            t = idx // frame_seqlen + start_frame
            y = (idx % frame_seqlen) // max(1, w)
            x = idx % max(1, w)
            pos_b = torch.stack([t, y, x], dim=-1)
            valid = idx < (f * h * w)
            pos[b, valid] = pos_b[valid]
        return pos

    def _segment_indices(
        self,
        length: int,
        device: torch.device,
        sink_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective_sink_len = self.sink_len if sink_len is None else max(0, int(sink_len))
        sink_end = min(effective_sink_len, length)
        tail_start = max(sink_end, length - self.tail_len)

        sink_idx = torch.arange(0, sink_end, device=device, dtype=torch.long)
        history_idx = torch.arange(sink_end, tail_start, device=device, dtype=torch.long)
        tail_idx = torch.arange(tail_start, length, device=device, dtype=torch.long)

        if self.aggressive_all:
            mandatory = torch.empty(0, device=device, dtype=torch.long)
            candidate = torch.arange(0, length, device=device, dtype=torch.long)
            return mandatory, candidate

        mandatory_parts = []
        candidate_parts = [history_idx]
        if self.prune_sink:
            candidate_parts.append(sink_idx)
        else:
            mandatory_parts.append(sink_idx)
        if self.prune_tail:
            candidate_parts.append(tail_idx)
        else:
            mandatory_parts.append(tail_idx)

        mandatory = torch.cat(mandatory_parts) if mandatory_parts else torch.empty(0, device=device, dtype=torch.long)
        candidate = torch.cat(candidate_parts) if candidate_parts else torch.empty(0, device=device, dtype=torch.long)
        return mandatory, candidate

    def _ranked_select(
        self,
        pos_seg: torch.Tensor,
        v_seg: torch.Tensor,
        budget: int,
        freqs: torch.Tensor | None,
        prompt_head: torch.Tensor | None,
        apply_selection: bool,
    ) -> torch.Tensor:
        n = v_seg.shape[0]
        select = torch.zeros(n, dtype=torch.bool, device=v_seg.device)
        if n == 0 or budget <= 0:
            return select

        if not apply_selection:
            select[-min(n, budget):] = True
            return select

        traj_ratio_eff, traj_weight_eff, _ = self._effective_selection_params(pos_seg)
        ivc_scores = None
        sem_scores = None
        traj_scores = None

        if self.ivc_ratio > 0 and freqs is not None:
            ivc_scores = self.ivc_selector.get_ivc_scores(pos_seg, d_model=self.head_dim, freqs=freqs)
            k_ivc = max(1, int(round(n * self.ivc_ratio)))
            select |= _topk_mask(ivc_scores, k=k_ivc)

        if self.semantic_ratio > 0:
            sem_scores = self.semantic_selector.get_semantic_scores(
                v_seg,
                prompt_v=prompt_head,
                seed_ratio=self.seed_ratio,
            )
            k_sem = max(1, int(round(n * self.semantic_ratio)))
            select |= _topk_mask(sem_scores, k=k_sem)

        if traj_ratio_eff > 0:
            traj_scores = self.get_trajectory_scores(pos_seg=pos_seg, v_seg=v_seg)
            k_traj = max(1, int(round(n * traj_ratio_eff)))
            select |= _topk_mask(traj_scores, k=k_traj)

        combined = torch.zeros(n, dtype=torch.float32, device=v_seg.device)
        if ivc_scores is not None:
            combined = combined + _normalize_scores(ivc_scores) * max(self.ivc_ratio, 1e-6)
        if sem_scores is not None:
            combined = combined + _normalize_scores(sem_scores) * max(self.semantic_ratio, 1e-6)
        if traj_scores is not None:
            traj_w = traj_weight_eff if traj_weight_eff > 0 else traj_ratio_eff
            combined = combined + _normalize_scores(traj_scores) * max(traj_w, 1e-6)
        if torch.all(combined == 0):
            combined = torch.arange(n, device=v_seg.device, dtype=torch.float32)

        num_selected = int(select.sum().item())
        if num_selected > budget:
            keep_idx = torch.topk(combined.masked_fill(~select, float("-inf")), k=budget, largest=True, sorted=False).indices
            select = torch.zeros_like(select)
            select[keep_idx] = True
            return select

        if num_selected < budget:
            remainder = ~select
            fill_scores = combined.masked_fill(~remainder, float("-inf"))
            add_k = min(int(remainder.sum().item()), budget - num_selected)
            if add_k > 0:
                add_idx = torch.topk(fill_scores, k=add_k, largest=True, sorted=False).indices
                select[add_idx] = True
        return select

    @staticmethod
    def get_trajectory_scores(pos_seg: torch.Tensor, v_seg: torch.Tensor) -> torch.Tensor:
        """
        Compute motion saliency by tracking value-vector changes along the same spatial lattice (y, x)
        over time t. Higher score means stronger temporal change for that trajectory.
        """
        n = v_seg.shape[0]
        if n <= 1:
            return torch.zeros(n, dtype=torch.float32, device=v_seg.device)
        if pos_seg.ndim != 2 or pos_seg.shape[1] != 3:
            raise ValueError(f"pos_seg must be [N,3], got {tuple(pos_seg.shape)}")

        pos = pos_seg.to(dtype=torch.long)
        y = pos[:, 1]
        x = pos[:, 2]
        t = pos[:, 0]
        max_x = int(x.max().item()) + 1 if x.numel() > 0 else 1
        yx_id = y * max(1, max_x) + x

        # Lexicographic order by (yx, t)
        sort_key = yx_id * max(1, int(t.max().item()) + 1) + t.clamp(min=0)
        perm = torch.argsort(sort_key)

        yx_s = yx_id[perm]
        v_s = v_seg[perm].float()
        scores_s = torch.zeros(n, dtype=torch.float32, device=v_seg.device)

        dv = (v_s[1:] - v_s[:-1]).norm(dim=-1)
        same_track = yx_s[1:] == yx_s[:-1]
        dv = dv * same_track.float()

        # Assign motion evidence to both endpoints of each temporal edge.
        scores_s[1:] = torch.maximum(scores_s[1:], dv)
        scores_s[:-1] = torch.maximum(scores_s[:-1], dv)

        scores = torch.zeros_like(scores_s)
        scores[perm] = scores_s
        return scores

    def update_cache(
        self,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        pos_seq: torch.Tensor,
        budget: int,
        freqs: torch.Tensor | None,
        prompt_head: torch.Tensor | None,
        apply_selection: bool,
        sink_len: int | None = None,
        head_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if budget <= 0:
            return (
                k_seq.new_empty((0, self.head_dim)),
                v_seq.new_empty((0, self.head_dim)),
                pos_seq.new_empty((0, 3), dtype=torch.long),
            )

        length = k_seq.shape[0]
        if length <= budget and not apply_selection:
            return k_seq, v_seq, pos_seq

        mandatory, candidate = self._segment_indices(length=length, device=k_seq.device, sink_len=sink_len)

        if mandatory.shape[0] >= budget:
            keep_idx = mandatory.sort().values[-budget:]
            return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        remain_budget = budget - mandatory.shape[0]
        if candidate.numel() == 0:
            keep_idx = mandatory.sort().values
            return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        if self.periodic_peak_mask:
            is_osc = True if head_idx is None else self.osc_head_flags[head_idx]
            should_apply = (not self.periodic_peak_only_oscillating) or is_osc
            if should_apply:
                cand_pos = pos_seq[candidate]
                peak_local_mask = self._periodic_peak_local_mask(cand_pos[:, 0].to(torch.long))
                if torch.any(peak_local_mask):
                    peak_global = candidate[peak_local_mask]
                    mandatory = torch.unique(torch.cat([mandatory, peak_global], dim=0), sorted=False)
                    if mandatory.shape[0] >= budget:
                        keep_idx = mandatory.sort().values[-budget:]
                        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]
                    remain_budget = budget - mandatory.shape[0]
                    candidate = candidate[~peak_local_mask]
                    if candidate.numel() == 0:
                        keep_idx = mandatory.sort().values
                        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        if self.history_frame_quota > 0:
            # Guarantee temporal coverage: each history frame contributes at least a small
            # number of tokens before global ranking.
            cand_pos = pos_seq[candidate]
            cand_v = v_seq[candidate]
            traj_scores = self.get_trajectory_scores(pos_seg=cand_pos, v_seg=cand_v)
            quota_scores = traj_scores
            _, _, quota_ivc_ratio_eff = self._effective_selection_params(cand_pos)
            if freqs is not None and quota_ivc_ratio_eff > 0:
                ivc_scores = self.ivc_selector.get_ivc_scores(cand_pos, d_model=self.head_dim, freqs=freqs)
                ivc_norm = _normalize_scores(ivc_scores)
                traj_norm = _normalize_scores(traj_scores)
                mix = quota_ivc_ratio_eff
                quota_scores = mix * ivc_norm + (1.0 - mix) * traj_norm
            picked_chunks = []
            for t_val in torch.unique(cand_pos[:, 0], sorted=True):
                local = torch.nonzero(cand_pos[:, 0] == t_val, as_tuple=False).squeeze(1)
                if local.numel() == 0:
                    continue
                k_keep = min(self.history_frame_quota, int(local.numel()))
                top_local = local[torch.topk(quota_scores[local], k=k_keep, largest=True, sorted=False).indices]
                picked_chunks.append(top_local)
            if picked_chunks:
                picked_local = torch.unique(torch.cat(picked_chunks, dim=0), sorted=False)
                picked_global = candidate[picked_local]
                mandatory = torch.unique(torch.cat([mandatory, picked_global], dim=0), sorted=False)
                if mandatory.shape[0] >= budget:
                    keep_idx = mandatory.sort().values[-budget:]
                    return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]
                remain_budget = budget - mandatory.shape[0]
                keep_mask = torch.ones(candidate.shape[0], dtype=torch.bool, device=candidate.device)
                keep_mask[picked_local] = False
                candidate = candidate[keep_mask]
                if candidate.numel() == 0:
                    keep_idx = mandatory.sort().values
                    return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

        candidate_pos = pos_seq[candidate]
        candidate_v = v_seq[candidate]
        local_mask = self._ranked_select(
            pos_seg=candidate_pos,
            v_seg=candidate_v,
            budget=remain_budget,
            freqs=freqs,
            prompt_head=prompt_head,
            apply_selection=apply_selection,
        )
        selected = candidate[local_mask]
        keep_idx = torch.cat([mandatory, selected]).sort().values
        if keep_idx.shape[0] > budget:
            keep_idx = keep_idx[-budget:]
        return k_seq[keep_idx], v_seq[keep_idx], pos_seq[keep_idx]

    @torch.no_grad()
    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        current_start: int | None = None,
        grid_sizes: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        prompt_v: torch.Tensor | None = None,
        **kwargs,
    ):
        update_start = perf_counter()
        if prompt_v is not None:
            self.prompt_v = prompt_v
        cache_update_mode = kwargs.get("cache_update_mode", "default")

        b, l_new, h, d = new_k.shape
        assert b == self.batch_size and h == self.num_heads and d == self.head_dim
        current_end = None
        if current_start is not None:
            current_end = current_start + l_new
        overwrite_current_block = False
        if current_end is not None:
            overwrite_current_block = all(
                current_end <= int(self.global_end_index[batch_idx])
                for batch_idx in range(self.batch_size)
            )

        if grid_sizes is None:
            result = super().update(new_k, new_v, current_start=current_start)
            self._record_profile("update_ms", update_start)
            return result

        new_k_flat = new_k.transpose(1, 2).reshape(b * h, l_new, d)
        new_v_flat = new_v.transpose(1, 2).reshape(b * h, l_new, d)
        pos_b = self._build_pos_ids(grid_sizes=grid_sizes, seq_len=l_new, current_start=current_start or 0, device=new_k.device)
        pos_flat = pos_b.unsqueeze(1).expand(b, h, l_new, 3).reshape(b * h, l_new, 3)
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
        frame_seqlen = int(frame_tokens[0].item())
        frame_start_t = 0 if frame_seqlen <= 0 else int((current_start or 0) // frame_seqlen)
        if self.use_osc_frame_mode:
            # Keep the dynamic region as a short local neighborhood for smoother transitions.
            self.tail_len = self.local_tail_frames * frame_seqlen
        else:
            self.tail_len = self._base_tail_len

        prompt_per_head = self.semantic_selector.prepare_prompt_values(
            self.prompt_v,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        self.update_step += 1
        should_reselect = (self.update_step % self.update_interval == 0)
        structural_change = False

        for i in range(b * h):
            batch_idx = i // h
            head_idx = i % h
            full_cap = self.capacities[head_idx]
            if self.osc_full_kv_retention and self.osc_head_flags[head_idx]:
                full_cap = self.max_capacity
            prompt_head = None
            if prompt_per_head is not None:
                prompt_head = prompt_per_head[head_idx]

            if self.use_osc_frame_mode and cache_update_mode in {"default", "clean"}:
                composition = (
                    self.compositions_row[head_idx]
                    if self.compositions_row is not None and head_idx < len(self.compositions_row)
                    else None
                )
                if composition is not None and composition.has_middle:
                    composition.update_all(
                        idx=i,
                        k_seq=new_k_flat[i],
                        v_seq=new_v_flat[i],
                        pos_seq=pos_flat[i],
                        frame_seqlen=frame_seqlen,
                        current_t=frame_start_t,
                    )
                else:
                    self._update_cyclic_anchors(
                        idx=i,
                        k_seq=new_k_flat[i],
                        v_seq=new_v_flat[i],
                        pos_seq=pos_flat[i],
                        frame_seqlen=frame_seqlen,
                        t_start=frame_start_t,
                    )
                    self._update_lag_anchors(
                        idx=i,
                        k_seq=new_k_flat[i],
                        v_seq=new_v_flat[i],
                        pos_seq=pos_flat[i],
                        frame_seqlen=frame_seqlen,
                        t_start=frame_start_t,
                    )

            if self.is_i2v and self.static_k[i] is None:
                if l_new >= self.context_len:
                    self.static_k[i] = new_k_flat[i, :self.context_len].clone()
                    self.static_v[i] = new_v_flat[i, :self.context_len].clone()
                    self.static_pos[i] = pos_flat[i, :self.context_len].clone()
                    if l_new > self.context_len:
                        self._set_dynamic_store(
                            i,
                            new_k_flat[i, self.context_len:],
                            new_v_flat[i, self.context_len:],
                            pos_flat[i, self.context_len:],
                            reserve_extra=max(16, (l_new - self.context_len) // 2),
                        )
                    else:
                        self._set_dynamic_empty(i, device=new_k.device, dtype=new_k.dtype)
                    self._current_block_token_len[i] = 0
                    structural_change = True
                    continue

            curr_dyn_k = self.dynamic_k[i]
            curr_dyn_v = self.dynamic_v[i]
            curr_dyn_pos = self.dynamic_pos[i]
            overwrite = False
            if current_end is not None:
                overwrite = current_end <= int(self.global_end_index[batch_idx])

            incoming_k = new_k_flat[i]
            incoming_v = new_v_flat[i]
            incoming_p = pos_flat[i]
            incoming_k, incoming_v, incoming_p = self._capture_sink_if_needed(
                idx=i,
                head_idx=head_idx,
                k_in=incoming_k,
                v_in=incoming_v,
                p_in=incoming_p,
                current_start=current_start,
                overwrite=overwrite,
                freqs=freqs,
                prompt_head=prompt_head,
            )
            l_in = incoming_k.shape[0]
            self._current_block_token_len[i] = int(l_in)

            if curr_dyn_k is None:
                self._set_dynamic_store(
                    i,
                    incoming_k,
                    incoming_v,
                    incoming_p,
                    reserve_extra=max(16, l_in // 2),
                )
                structural_change = True
            elif overwrite:
                if l_in == 0:
                    self._sync_dynamic_views(i)
                elif curr_dyn_k.shape[0] >= l_in:
                    self._overwrite_dynamic_tail(i, incoming_k, incoming_v, incoming_p)
                else:
                    self._set_dynamic_store(
                        i,
                        incoming_k,
                        incoming_v,
                        incoming_p,
                        reserve_extra=max(16, l_in // 2),
                    )
                    structural_change = True
            else:
                self._append_dynamic(i, incoming_k, incoming_v, incoming_p)
                structural_change = True

            k_merged = self.dynamic_k[i]
            v_merged = self.dynamic_v[i]
            p_merged = self.dynamic_pos[i]

            stable_kind = self._stable_strategy_kind(head_idx)
            dyn_cap = full_cap
            if self.use_osc_frame_mode and not self.is_i2v:
                # Oscillating heads keep short local recent tail; stable heads can keep wider history
                # and let stable policies downsample at frame level.
                if stable_kind is None:
                    dyn_cap = min(full_cap, self._head_recent_frames(head_idx) * frame_seqlen)
                else:
                    dyn_cap = full_cap
            elif self.is_i2v:
                dyn_cap = max(0, full_cap - self.context_len)
            elif self.sink_grid_decoupling and self.static_k[i] is not None and self.decouple_head_flags[head_idx]:
                dyn_cap = max(0, full_cap - int(self.static_k[i].shape[0]))

            if dyn_cap <= 0:
                self._set_dynamic_empty(i, device=new_k.device, dtype=new_k.dtype)
                structural_change = True
                continue

            if self.use_osc_frame_mode:
                if k_merged.shape[0] > dyn_cap:
                    self._keep_dynamic_suffix(i, dyn_cap)
                    k_merged = self.dynamic_k[i]
                    v_merged = self.dynamic_v[i]
                    p_merged = self.dynamic_pos[i]
                    structural_change = True
                if stable_kind is not None:
                    k_merged, v_merged, p_merged = self._apply_stable_strategy(
                        head_idx=head_idx,
                        k_seq=k_merged,
                        v_seq=v_merged,
                        pos_seq=p_merged,
                        frame_seqlen=frame_seqlen,
                    )
                    reserve_extra = min(max(16, l_in), max(0, dyn_cap - int(k_merged.shape[0])))
                    self._set_dynamic_store(i, k_merged, v_merged, p_merged, reserve_extra=reserve_extra)
                    structural_change = True
            else:
                needs_compaction = (k_merged.shape[0] > dyn_cap)
                allow_reselect = cache_update_mode in {"default", "clean"}
                apply_selection = allow_reselect and should_reselect and freqs is not None
                segment_sink_len = self.sink_len
                if self.sink_grid_decoupling and self.static_k[i] is not None and self.decouple_head_flags[head_idx]:
                    # In decoupling mode, sink tokens are fully externalized in static_k/static_v.
                    # Dynamic history should not re-introduce a sink-protected prefix.
                    segment_sink_len = 0
                if needs_compaction or apply_selection:
                    k_merged, v_merged, p_merged = self.update_cache(
                        k_seq=k_merged,
                        v_seq=v_merged,
                        pos_seq=p_merged,
                        budget=dyn_cap,
                        freqs=freqs,
                        prompt_head=prompt_head,
                        apply_selection=apply_selection,
                        sink_len=segment_sink_len,
                        head_idx=head_idx,
                    )
                    reserve_extra = min(max(16, l_in), max(0, dyn_cap - int(k_merged.shape[0])))
                    self._set_dynamic_store(i, k_merged, v_merged, p_merged, reserve_extra=reserve_extra)
                    structural_change = True
                elif k_merged.shape[0] > dyn_cap:
                    self._keep_dynamic_suffix(i, dyn_cap)
                    structural_change = True

        if current_end is not None:
            for batch_idx in range(self.batch_size):
                self.global_end_index[batch_idx] = current_end
        can_reuse_same_block = (
            self.readout_cache_enabled
            and current_start is not None
            and overwrite_current_block
            and self._readout_cache_valid
            and self._readout_cache_current_start == int(current_start)
            and not structural_change
            and (cache_update_mode == "noisy" or (cache_update_mode == "clean" and not self.use_osc_frame_mode))
        )
        if can_reuse_same_block:
            self._readout_cache_tail_dirty = True
        else:
            self._invalidate_readout_cache()
        self._record_profile("update_ms", update_start)

    def get_flat_kv_and_pos(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        total_k = []
        total_v = []
        total_pos = []
        lengths = []

        for i in range(self.batch_size * self.num_heads):
            k_parts = []
            v_parts = []
            p_parts = []

            if self.static_k[i] is not None:
                k_parts.append(self.static_k[i])
                v_parts.append(self.static_v[i])
                p_parts.append(self.static_pos[i])
            if self.dynamic_k[i] is not None:
                k_parts.append(self.dynamic_k[i])
                v_parts.append(self.dynamic_v[i])
                p_parts.append(self.dynamic_pos[i])

            if len(k_parts) == 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                t_k = torch.empty(0, self.head_dim, device=device)
                t_v = torch.empty(0, self.head_dim, device=device)
                t_pos = torch.empty(0, 3, dtype=torch.long, device=device)
            elif len(k_parts) == 1:
                t_k = k_parts[0]
                t_v = v_parts[0]
                t_pos = p_parts[0]
            else:
                t_k = torch.cat(k_parts, dim=0)
                t_v = torch.cat(v_parts, dim=0)
                t_pos = torch.cat(p_parts, dim=0)

            total_k.append(t_k)
            total_v.append(t_v)
            total_pos.append(t_pos)
            lengths.append(t_k.shape[0])

        k_flat = torch.cat(total_k, dim=0)
        v_flat = torch.cat(total_v, dim=0)
        pos_flat = torch.cat(total_pos, dim=0)
        cu_seqlens_k = torch.tensor([0] + lengths, dtype=torch.int32, device=k_flat.device).cumsum(0, dtype=torch.int32)
        max_seqlen_k = max(lengths) if lengths else 0
        self.last_flat_pos_ids = pos_flat
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_flat

    def get_flat_kv(self, **kwargs):
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, _ = self.get_flat_kv_and_pos()
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k

    def _ensure_workspace(
        self,
        total_len: int,
        num_seq: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return workspace views for flattened decoupled readout.

        Allocates with 20% headroom on first call or when capacity is insufficient.
        Otherwise returns sliced views of existing buffers (cu_seqlens is zero-filled).
        """
        alloc_len = int(total_len * 1.2) + 64  # 20% headroom
        cu_len = num_seq + 1

        need_alloc = (
            self._ws_k is None
            or self._ws_k.shape[0] < total_len
            or self._ws_k.device != device
            or self._ws_k.dtype != dtype
        )
        if need_alloc:
            self._ws_k = torch.empty(alloc_len, self.head_dim, device=device, dtype=dtype)
            self._ws_v = torch.empty(alloc_len, self.head_dim, device=device, dtype=dtype)
            self._ws_frame_ids = torch.empty(alloc_len, dtype=torch.long, device=device)
            self._ws_rope_pos = torch.empty(alloc_len, 3, dtype=torch.long, device=device)

        if self._ws_cu_seqlens is None or self._ws_cu_seqlens.shape[0] < cu_len or self._ws_cu_seqlens.device != device:
            self._ws_cu_seqlens = torch.zeros(cu_len, dtype=torch.int32, device=device)
        else:
            self._ws_cu_seqlens[:cu_len].zero_()

        return (
            self._ws_k[:total_len],
            self._ws_v[:total_len],
            self._ws_frame_ids[:total_len],
            self._ws_cu_seqlens[:cu_len],
            self._ws_rope_pos[:total_len],
        )

    def _invalidate_readout_cache(self) -> None:
        self._readout_cache_valid = False
        self._readout_cache_current_start = None
        self._readout_cache_sync_t_raw = None
        self._readout_cache_total_len = 0
        self._readout_cache_max_seqlen = 0
        self._readout_cache_frame_seqlen = 0
        self._readout_cache_tail_dirty = False
        num_seq = self.batch_size * self.num_heads
        self._readout_static_specs = [None] * num_seq
        self._readout_tail_specs = [None] * num_seq

    def _cache_readout_layout(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        frame_seqlen: int,
        total_len: int,
        max_seqlen_k: int,
    ) -> None:
        self._readout_cache_valid = True
        self._readout_cache_current_start = int(current_start)
        self._readout_cache_sync_t_raw = int(sync_t_raw)
        self._readout_cache_total_len = int(total_len)
        self._readout_cache_max_seqlen = int(max_seqlen_k)
        self._readout_cache_frame_seqlen = int(frame_seqlen)
        self._readout_cache_tail_dirty = False

    def _can_reuse_readout_cache(self, current_start: int, sync_t_raw: int, frame_seqlen: int) -> bool:
        if not self.readout_cache_enabled:
            return False
        if not self._readout_cache_valid:
            return False
        if self._ws_k is None or self._ws_v is None or self._ws_frame_ids is None or self._ws_cu_seqlens is None:
            return False
        return (
            self._readout_cache_current_start == int(current_start)
            and self._readout_cache_sync_t_raw == int(sync_t_raw)
            and self._readout_cache_frame_seqlen == int(frame_seqlen)
        )

    def _cached_readout_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        total_len = self._readout_cache_total_len
        num_seq = self.batch_size * self.num_heads
        return (
            self._ws_k[:total_len],
            self._ws_v[:total_len],
            self._ws_cu_seqlens[: num_seq + 1],
            self._readout_cache_max_seqlen,
            self._ws_frame_ids[:total_len],
        )

    def _refresh_cached_readout_mutable_segments(
        self,
        *,
        current_start: int,
        sync_t_raw: int,
        sync_t: int,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        if not self._readout_cache_valid or self._ws_rope_pos is None:
            return False

        capture_physical = self.capture_frame_id_mode == "physical"
        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()
        rewrite_static = int(current_start) == 0
        num_seq = self.batch_size * self.num_heads

        for i in range(num_seq):
            head_idx = i % self.num_heads
            static_spec = self._readout_static_specs[i]
            if rewrite_static and static_spec is not None:
                stat_k = self.static_k[i]
                stat_v = self.static_v[i]
                stat_pos = self.static_pos[i]
                offset, n_s = static_spec
                if (
                    stat_k is None
                    or stat_v is None
                    or stat_pos is None
                    or stat_k.shape[0] != n_s
                ):
                    return False
                self._ws_k[offset:offset + n_s] = stat_k
                self._ws_v[offset:offset + n_s] = stat_v
                rope_pos = self._ws_rope_pos[offset:offset + n_s]
                rope_pos.copy_(stat_pos)
                if self.decouple_head_flags[head_idx]:
                    rope_pos[:, 0] = sync_t
                    if capture_physical:
                        self._ws_frame_ids[offset:offset + n_s] = stat_pos[:, 0].to(dtype=torch.long)
                    else:
                        self._ws_frame_ids[offset:offset + n_s] = sync_t
                else:
                    self._ws_frame_ids[offset:offset + n_s] = stat_pos[:, 0].to(dtype=torch.long)
                rope_start = perf_counter()
                apply_rope_to_flat_k(
                    self._ws_k[offset:offset + n_s],
                    rope_pos,
                    freqs=freqs,
                    freq_parts=freq_parts,
                    out=self._ws_k[offset:offset + n_s],
                )
                self._record_profile("rope_ms", rope_start)

            tail_spec = self._readout_tail_specs[i]
            if tail_spec is None:
                continue
            dyn_k = self.dynamic_k[i]
            dyn_v = self.dynamic_v[i]
            dyn_pos = self.dynamic_pos[i]
            offset, tail_len = tail_spec
            if (
                dyn_k is None
                or dyn_v is None
                or dyn_pos is None
                or dyn_k.shape[0] < tail_len
            ):
                return False
            dyn_k_tail = dyn_k[-tail_len:]
            dyn_v_tail = dyn_v[-tail_len:]
            dyn_pos_tail = dyn_pos[-tail_len:]

            self._ws_k[offset:offset + tail_len] = dyn_k_tail
            self._ws_v[offset:offset + tail_len] = dyn_v_tail
            rope_pos = self._ws_rope_pos[offset:offset + tail_len]
            rope_pos.copy_(dyn_pos_tail)
            if self.history_time_mapping_mode != "none":
                map_dynamic_pos_time(
                    rope_pos,
                    current_t=sync_t_raw,
                    history_time_mapping_mode=self.history_time_mapping_mode,
                    history_relative_t_max=self.history_relative_t_max,
                    history_time_soft_factor=self.history_time_soft_factor,
                    inplace=True,
                )
            if capture_physical:
                self._ws_frame_ids[offset:offset + tail_len] = dyn_pos_tail[:, 0].to(dtype=torch.long)
            else:
                self._ws_frame_ids[offset:offset + tail_len] = rope_pos[:, 0].to(dtype=torch.long)
            rope_start = perf_counter()
            apply_rope_to_flat_k(
                self._ws_k[offset:offset + tail_len],
                rope_pos,
                freqs=freqs,
                freq_parts=freq_parts,
                out=self._ws_k[offset:offset + tail_len],
            )
            self._record_profile("rope_ms", rope_start)

        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._readout_cache_tail_dirty = False
        return True

    def get_decoupled_flat_kv(
        self,
        current_start: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, _ = self.get_decoupled_flat_kv_and_frames(
            current_start=current_start,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )
        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k

    def _collect_middle_cache(
        self,
        seq_idx: int,
        head_idx: int,
        sync_t_raw: int,
        has_stat: bool,
    ) -> tuple[tuple[str, object], int]:
        n = 0
        composition = (
            self.compositions_row[head_idx]
            if self.compositions_row is not None and head_idx < len(self.compositions_row)
            else None
        )
        collect_start = perf_counter()
        if composition is not None and composition.has_middle:
            tail_min_t = sync_t_raw - composition.recent_frames + 1
            sink_max_t = 0 if has_stat else -1
            collected = composition.collect_all(seq_idx, sync_t_raw, tail_min_t, sink_max_t)
            self._record_profile("collect_ms", collect_start)
            for anchor in collected:
                n += int(anchor.token_count)
            return ("comp", collected), n

        inline = []
        if self.use_osc_frame_mode and self._is_phase_sink_head(head_idx):
            phase_idx = sync_t_raw % self.phase_period
            tail_min_t_cyc = sync_t_raw - self._head_recent_frames(head_idx) + 1
            for anchor in self.cyclic_buckets[seq_idx][phase_idx]:
                anchor_t = anchor[3]
                if has_stat and anchor_t == 0:
                    continue
                if anchor_t >= tail_min_t_cyc:
                    continue
                inline.append(("cyc", anchor))
                n += anchor[0].shape[0]
        if self._is_phase_sink_head(head_idx):
            lag_offsets = self._head_lag_offsets(head_idx)
            if len(lag_offsets) > 0:
                tail_min_t_lag = sync_t_raw - self._head_recent_frames(head_idx) + 1
            for lag in lag_offsets:
                target_t = sync_t_raw - lag
                if target_t < 0:
                    continue
                if has_stat and target_t == 0:
                    continue
                if target_t >= tail_min_t_lag:
                    continue
                anchor = self._find_anchor_by_t(self.lag_anchor_frames[seq_idx], target_t)
                if anchor is None:
                    continue
                inline.append(("lag", lag, anchor))
                n += anchor[0].shape[0]
        self._record_profile("collect_ms", collect_start)
        return ("inline", inline), n

    def _write_anchor_segment(
        self,
        out_k: torch.Tensor,
        out_v: torch.Tensor,
        out_frame_ids: torch.Tensor,
        offset: int,
        *,
        anchor_k: torch.Tensor,
        anchor_v: torch.Tensor,
        anchor_pos: torch.Tensor,
        effective_t: int | None,
        freqs: torch.Tensor,
        freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        capture_physical: bool,
    ) -> int:
        n = anchor_k.shape[0]
        out_v[offset:offset + n] = anchor_v
        if effective_t is None:
            rope_pos = anchor_pos
            mapped_t = anchor_pos[:, 0].to(dtype=torch.long)
        else:
            rope_pos = anchor_pos.clone()
            rope_pos[:, 0] = int(effective_t)
            mapped_t = rope_pos[:, 0].to(dtype=torch.long)
        rope_start = perf_counter()
        out_k[offset:offset + n] = apply_rope_to_flat_k(
            anchor_k,
            rope_pos,
            freqs=freqs,
            freq_parts=freq_parts,
        )
        self._record_profile("rope_ms", rope_start)
        if capture_physical:
            out_frame_ids[offset:offset + n] = anchor_pos[:, 0].to(dtype=torch.long)
        else:
            out_frame_ids[offset:offset + n] = mapped_t
        return offset + n

    def get_decoupled_flat_kv_and_frames(
        self,
        current_start: int,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        Build flattened KV for sink-grid decoupling:
        - static sink is rotated to the current query frame index (time-synchronized),
        - dynamic history is rotated with its own saved position ids.

        Two-phase implementation:
        Phase A — compute per-head token counts (no tensor allocation).
        Phase B — write K/V/pos directly into pre-allocated workspace buffers,
                  then apply RoPE in a single batched call (no clone/cat/scatter).
        """
        if not self.sink_grid_decoupling:
            k_flat, v_flat, cu_seqlens_k, max_seqlen_k = self.get_flat_kv()
            frame_ids = (
                self.last_flat_pos_ids[:, 0].to(dtype=torch.long)
                if self.last_flat_pos_ids is not None
                else torch.empty(0, dtype=torch.long, device=k_flat.device)
            )
            return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, frame_ids

        if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
            raise ValueError(f"grid_sizes must be [B,3], got {tuple(grid_sizes.shape)}")
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}")
        frame_seqlen = int(frame_tokens[0].item())
        sync_t_raw = 0 if frame_seqlen <= 0 else int(current_start // frame_seqlen)
        sync_t = self._map_sink_time(sync_t_raw)

        # Pre-split freqs once for all heads (avoids ~150k redundant .split() calls)
        c = self.head_dim // 2
        split_sizes = [c - 2 * (c // 3), c // 3, c // 3]
        freq_parts = tuple(freqs.to(device=freqs.device).split(split_sizes, dim=1))

        if self._can_reuse_readout_cache(current_start=current_start, sync_t_raw=sync_t_raw, frame_seqlen=frame_seqlen):
            if self._readout_cache_tail_dirty:
                if self._refresh_cached_readout_mutable_segments(
                    current_start=current_start,
                    sync_t_raw=sync_t_raw,
                    sync_t=sync_t,
                    freqs=freqs,
                    freq_parts=freq_parts,
                ):
                    return self._cached_readout_views()
                self._invalidate_readout_cache()
            else:
                return self._cached_readout_views()

        num_seq = self.batch_size * self.num_heads

        # ========== Phase A: Compute per-head token counts ==========
        lengths = [0] * num_seq
        _cached = [None] * num_seq  # per-seq cached anchor data

        for i in range(num_seq):
            head_idx = i % self.num_heads
            n = 0
            stat_k = self.static_k[i]
            has_stat = stat_k is not None and stat_k.shape[0] > 0
            if has_stat:
                n += stat_k.shape[0]
            dyn_k = self.dynamic_k[i]
            if dyn_k is not None and dyn_k.shape[0] > 0:
                n += dyn_k.shape[0]
            anchor_cache, anchor_n = self._collect_middle_cache(i, head_idx, sync_t_raw, has_stat)
            _cached[i] = anchor_cache
            n += anchor_n

            lengths[i] = n

        # ========== Pre-allocate workspace buffers ==========
        total_len = sum(lengths)
        max_seqlen_k = max(lengths) if lengths else 0
        device = freqs.device
        # Determine dtype from first available tensor
        dtype = torch.bfloat16
        for i in range(num_seq):
            if self.static_k[i] is not None and self.static_k[i].shape[0] > 0:
                dtype = self.static_k[i].dtype
                break
            if self.dynamic_k[i] is not None and self.dynamic_k[i].shape[0] > 0:
                dtype = self.dynamic_k[i].dtype
                break

        k_flat, v_flat, frame_ids_flat, cu_seqlens_k, rope_pos_flat = self._ensure_workspace(
            total_len, num_seq, device, dtype,
        )

        # ========== Phase B: Direct-write K/V/pos into workspace ==========
        # K data is written into k_flat, pos data into rope_pos_flat.
        # After the loop, a single batched RoPE call rotates k_flat in-place.
        offset = 0
        capture_physical = self.capture_frame_id_mode == "physical"
        rope_before = self._profile_stats["rope_ms"]
        pack_start = perf_counter()
        self._readout_static_specs = [None] * num_seq
        self._readout_tail_specs = [None] * num_seq

        for i in range(num_seq):
            head_idx = i % self.num_heads

            # --- Static sink ---
            stat_k = self.static_k[i]
            stat_v = self.static_v[i]
            stat_pos = self.static_pos[i]
            if stat_k is not None and stat_k.shape[0] > 0:
                n_s = stat_k.shape[0]
                self._readout_static_specs[i] = (offset, n_s)
                k_flat[offset:offset + n_s] = stat_k
                v_flat[offset:offset + n_s] = stat_v
                if self.decouple_head_flags[head_idx]:
                    # Write pos with time dimension replaced by sync_t (no clone needed)
                    rope_pos_flat[offset:offset + n_s] = stat_pos
                    rope_pos_flat[offset:offset + n_s, 0] = sync_t
                    if capture_physical:
                        frame_ids_flat[offset:offset + n_s] = stat_pos[:, 0].to(dtype=torch.long)
                    else:
                        frame_ids_flat[offset:offset + n_s] = sync_t
                else:
                    rope_pos_flat[offset:offset + n_s] = stat_pos
                    frame_ids_flat[offset:offset + n_s] = stat_pos[:, 0].to(dtype=torch.long)
                offset += n_s

            # --- Dynamic history ---
            dyn_k = self.dynamic_k[i]
            dyn_v = self.dynamic_v[i]
            dyn_pos = self.dynamic_pos[i]
            if dyn_k is not None and dyn_k.shape[0] > 0:
                n_d = dyn_k.shape[0]
                dyn_offset = offset
                k_flat[offset:offset + n_d] = dyn_k
                v_flat[offset:offset + n_d] = dyn_v
                rope_pos_flat[offset:offset + n_d] = dyn_pos
                dyn_pos_ws = rope_pos_flat[offset:offset + n_d]
                if self.history_time_mapping_mode != "none":
                    map_dynamic_pos_time(
                        dyn_pos_ws,
                        current_t=sync_t_raw,
                        history_time_mapping_mode=self.history_time_mapping_mode,
                        history_relative_t_max=self.history_relative_t_max,
                        history_time_soft_factor=self.history_time_soft_factor,
                        inplace=True,
                    )
                if capture_physical:
                    frame_ids_flat[offset:offset + n_d] = dyn_pos[:, 0].to(dtype=torch.long)
                else:
                    frame_ids_flat[offset:offset + n_d] = dyn_pos_ws[:, 0].to(dtype=torch.long)
                tail_len = min(max(0, int(self._current_block_token_len[i])), n_d)
                if tail_len > 0:
                    self._readout_tail_specs[i] = (dyn_offset + n_d - tail_len, tail_len)
                offset += n_d

            # --- Anchors (from cached Phase A data) ---
            anchor_type, anchor_data = _cached[i]
            if anchor_type == 'comp':
                for anchor in anchor_data:
                    n_a = anchor.k.shape[0]
                    k_flat[offset:offset + n_a] = anchor.k
                    v_flat[offset:offset + n_a] = anchor.v
                    rope_pos_flat[offset:offset + n_a] = anchor.pos
                    if anchor.dynamic_rope:
                        rope_pos_flat[offset:offset + n_a, 0] = sync_t
                    if capture_physical:
                        frame_ids_flat[offset:offset + n_a] = anchor.pos[:, 0].to(dtype=torch.long)
                    else:
                        frame_ids_flat[offset:offset + n_a] = rope_pos_flat[offset:offset + n_a, 0].to(dtype=torch.long)
                    offset += n_a
            else:  # inline
                for anchor_info in anchor_data:
                    if anchor_info[0] == 'cyc':
                        anchor_k, anchor_v, anchor_pos, anchor_t = anchor_info[1]
                        n_a = anchor_k.shape[0]
                        k_flat[offset:offset + n_a] = anchor_k
                        v_flat[offset:offset + n_a] = anchor_v
                        rope_pos_flat[offset:offset + n_a] = anchor_pos
                        if self.phase_sink_dynamic_rope:
                            rope_pos_flat[offset:offset + n_a, 0] = sync_t
                    else:  # lag
                        lag = anchor_info[1]
                        anchor_k, anchor_v, anchor_pos, _ = anchor_info[2]
                        n_a = anchor_k.shape[0]
                        k_flat[offset:offset + n_a] = anchor_k
                        v_flat[offset:offset + n_a] = anchor_v
                        rope_pos_flat[offset:offset + n_a] = anchor_pos
                        if self.osc_lag_dynamic_rope:
                            rope_pos_flat[offset:offset + n_a, 0] = max(0, sync_t - lag)
                    if capture_physical:
                        frame_ids_flat[offset:offset + n_a] = anchor_pos[:, 0].to(dtype=torch.long)
                    else:
                        frame_ids_flat[offset:offset + n_a] = rope_pos_flat[offset:offset + n_a, 0].to(dtype=torch.long)
                    offset += n_a

            cu_seqlens_k[i + 1] = offset

        # ========== Single batched RoPE call — writes directly into k_flat ==========
        if offset > 0:
            rope_start = perf_counter()
            apply_rope_to_flat_k(
                k_flat[:offset], rope_pos_flat[:offset], freqs=freqs,
                freq_parts=freq_parts, out=k_flat[:offset],
            )
            self._record_profile("rope_ms", rope_start)
        self._profile_stats["pack_ms"] += max(
            0.0,
            (perf_counter() - pack_start) * 1000.0
            - (self._profile_stats["rope_ms"] - rope_before),
        )
        self._cache_readout_layout(
            current_start=current_start,
            sync_t_raw=sync_t_raw,
            frame_seqlen=frame_seqlen,
            total_len=total_len,
            max_seqlen_k=max_seqlen_k,
        )

        return k_flat, v_flat, cu_seqlens_k, max_seqlen_k, frame_ids_flat

    @staticmethod
    def apply_rope_to_flat_k(k_flat: torch.Tensor, pos_3d: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return apply_rope_to_flat_k(k_flat, pos_3d, freqs)
