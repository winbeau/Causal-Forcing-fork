import torch

# Auto-detect Triton availability for fused RoPE kernel
_TRITON_ROPE_AVAILABLE = False
_triton_apply_rope_to_flat_k = None
_triton_apply_temporal_rope_delta = None
try:
    from .triton_rope import triton_apply_rope_to_flat_k as _triton_fn
    from .triton_rope import triton_apply_temporal_rope_delta as _triton_delta_fn
    _triton_apply_rope_to_flat_k = _triton_fn
    _triton_apply_temporal_rope_delta = _triton_delta_fn
    _TRITON_ROPE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


def triton_rope_available() -> bool:
    return _TRITON_ROPE_AVAILABLE


def _pytorch_apply_rope_to_flat_k(
    k_flat: torch.Tensor,
    pos_3d: torch.Tensor,
    freqs: torch.Tensor,
    freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Original PyTorch implementation (fallback)."""
    if k_flat.numel() == 0:
        return out if out is not None else k_flat

    if k_flat.shape[1] % 2 != 0:
        raise ValueError(f"Head dim must be even, got {k_flat.shape[1]}")

    device = k_flat.device
    freqs = freqs.to(device=device)
    pos_3d = pos_3d.to(device=device, dtype=torch.long)

    c = k_flat.shape[1] // 2
    if freq_parts is not None:
        ft, fy, fx = freq_parts
    else:
        split = [c - 2 * (c // 3), c // 3, c // 3]
        ft, fy, fx = freqs.split(split, dim=1)

    parts = []
    if ft.shape[1] > 0:
        t_idx = pos_3d[:, 0].clamp(min=0, max=max(0, ft.shape[0] - 1))
        parts.append(ft[t_idx])
    if fy.shape[1] > 0:
        y_idx = pos_3d[:, 1].clamp(min=0, max=max(0, fy.shape[0] - 1))
        parts.append(fy[y_idx])
    if fx.shape[1] > 0:
        x_idx = pos_3d[:, 2].clamp(min=0, max=max(0, fx.shape[0] - 1))
        parts.append(fx[x_idx])

    compute_dtype = torch.float32 if k_flat.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64
    complex_dtype = torch.complex64 if compute_dtype == torch.float32 else torch.complex128

    if parts:
        freqs_sel = torch.cat(parts, dim=1).to(dtype=complex_dtype)
    else:
        freqs_sel = torch.ones((k_flat.shape[0], c), dtype=complex_dtype, device=device)

    x_c = torch.view_as_complex(k_flat.to(compute_dtype).reshape(k_flat.shape[0], -1, 2))
    x_rot = torch.view_as_real(x_c * freqs_sel).flatten(1)
    result = x_rot.to(dtype=k_flat.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _pytorch_apply_temporal_rope_delta(
    k_flat: torch.Tensor,
    old_t: torch.Tensor | int,
    new_t: torch.Tensor | int,
    ft: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if k_flat.numel() == 0:
        return out if out is not None else k_flat

    if k_flat.shape[1] % 2 != 0:
        raise ValueError(f"Head dim must be even, got {k_flat.shape[1]}")

    device = k_flat.device
    ft = ft.to(device=device)
    c_half = k_flat.shape[1] // 2
    ft_cols = int(ft.shape[1])
    if ft_cols <= 0:
        if out is not None:
            out.copy_(k_flat)
            return out
        return k_flat.clone()
    if ft_cols > c_half:
        raise ValueError(f"Temporal freq columns {ft_cols} exceed head_dim/2 {c_half}")

    old_idx = old_t if isinstance(old_t, torch.Tensor) else torch.full(
        (k_flat.shape[0],), int(old_t), device=device, dtype=torch.long
    )
    new_idx = new_t if isinstance(new_t, torch.Tensor) else torch.full(
        (k_flat.shape[0],), int(new_t), device=device, dtype=torch.long
    )
    old_idx = old_idx.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, ft.shape[0] - 1))
    new_idx = new_idx.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, ft.shape[0] - 1))

    delta = ft[new_idx] * ft[old_idx].conj()
    compute_dtype = torch.float32 if k_flat.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float64
    complex_dtype = torch.complex64 if compute_dtype == torch.float32 else torch.complex128

    result = k_flat.clone() if out is None else out
    if out is not None:
        result.copy_(k_flat)

    rotated = k_flat[:, : ft_cols * 2].to(compute_dtype)
    x_c = torch.view_as_complex(rotated.reshape(k_flat.shape[0], ft_cols, 2))
    delta_c = delta.to(dtype=complex_dtype)
    x_rot = torch.view_as_real(x_c * delta_c).reshape(k_flat.shape[0], ft_cols * 2).to(dtype=k_flat.dtype)
    result[:, : ft_cols * 2].copy_(x_rot)
    return result


def apply_rope_to_flat_k(
    k_flat: torch.Tensor,
    pos_3d: torch.Tensor,
    freqs: torch.Tensor,
    freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply 3D RoPE to flat K tensor. Auto-dispatches to Triton when available."""
    if k_flat.numel() == 0:
        return out if out is not None else k_flat
    if _TRITON_ROPE_AVAILABLE and k_flat.is_cuda:
        return _triton_apply_rope_to_flat_k(k_flat, pos_3d, freqs, freq_parts, out=out)
    return _pytorch_apply_rope_to_flat_k(k_flat, pos_3d, freqs, freq_parts, out=out)


def apply_temporal_rope_delta(
    k_flat: torch.Tensor,
    old_t: torch.Tensor | int,
    new_t: torch.Tensor | int,
    ft: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update an already-roped K tensor when only the temporal RoPE index changes."""
    if k_flat.numel() == 0:
        return out if out is not None else k_flat
    if _TRITON_ROPE_AVAILABLE and k_flat.is_cuda:
        return _triton_apply_temporal_rope_delta(k_flat, old_t, new_t, ft, out=out)
    return _pytorch_apply_temporal_rope_delta(k_flat, old_t, new_t, ft, out=out)


def map_sink_time(
    sync_t_raw: int,
    sink_time_mapping_mode: str,
    sink_time_clamp_min: int,
    sink_time_clamp_max: int,
    decoupled_sink_time_lag: int,
) -> int:
    mode = sink_time_mapping_mode
    if mode == "window_clamp":
        # Keep sink-query relative distance within a training-domain window [min, max].
        # For t <= max, this keeps continuity (effectively sync_t=0). For long videos,
        # relative distance is clamped to max and avoids large extrapolation.
        delta_t = min(max(sync_t_raw, sink_time_clamp_min), sink_time_clamp_max)
        return max(0, sync_t_raw - delta_t)
    # Default: classic fixed lag.
    return max(0, sync_t_raw - decoupled_sink_time_lag)


def map_dynamic_pos_time(
    dyn_pos: torch.Tensor,
    current_t: int,
    history_time_mapping_mode: str,
    history_relative_t_max: int,
    history_time_soft_factor: float,
    inplace: bool = False,
) -> torch.Tensor:
    mode = history_time_mapping_mode
    if mode == "none" or history_relative_t_max <= 0:
        return dyn_pos

    if mode == "relative_softcap":
        mapped = dyn_pos if inplace else dyn_pos.clone()
        t = mapped[:, 0].to(dtype=torch.long)
        rel = (current_t - t).clamp(min=0)
        over = (rel - history_relative_t_max).clamp(min=0)
        # Softly compress long-range relative distance instead of hard clipping.
        compressed_over = torch.round(over.to(torch.float32) * history_time_soft_factor).to(torch.long)
        rel_mapped = torch.where(rel <= history_relative_t_max, rel, history_relative_t_max + compressed_over)
        mapped[:, 0] = (current_t - rel_mapped).clamp(min=0)
        return mapped

    if mode != "relative_clamp":
        return dyn_pos
    mapped = dyn_pos if inplace else dyn_pos.clone()
    t = mapped[:, 0].to(dtype=torch.long)
    rel = current_t - t
    rel = rel.clamp(min=0, max=history_relative_t_max)
    mapped[:, 0] = current_t - rel
    return mapped
