"""Triton-fused 3D RoPE kernel for HeadKV.

Fuses the following 7-10 PyTorch ops into a single kernel launch:
1. 3D position indexing (clamp + gather for ft/fy/fx)
2. Complex multiply (view_as_complex + multiply + view_as_real)
3. dtype cast (float32 -> bfloat16/float16)

This is the absolute hotspot: called ~226k times per prompt, each with
small batches (1-30 tokens), making kernel launch overhead dominant.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rope_3d_kernel(
    # K input/output (in-place or out-of-place)
    k_ptr,
    out_ptr,
    # 3D position indices [N, 3] (int64)
    pos_ptr,
    # Freq tables: ft [ft_rows, ft_cols], fy [fy_rows, fy_cols], fx [fx_rows, fx_cols]
    # All stored as complex64 (interleaved real/imag float32 pairs)
    ft_ptr,
    fy_ptr,
    fx_ptr,
    # Table dimensions
    ft_rows: tl.constexpr,
    fy_rows: tl.constexpr,
    fx_rows: tl.constexpr,
    ft_cols: tl.constexpr,
    fy_cols: tl.constexpr,
    fx_cols: tl.constexpr,
    # K dimensions
    N,  # number of tokens
    C_half,  # head_dim // 2  (number of complex pairs)
    k_stride_row,  # stride of k_ptr per row (in elements, not bytes)
    out_stride_row,
    pos_stride_row,
    # Output dtype: 0=float32, 1=bfloat16, 2=float16
    out_dtype_code: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Each program handles one token row (C_half complex pairs)."""
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C_half

    # 1. Load 3D position indices
    pos_base = row * pos_stride_row
    t_idx = tl.load(pos_ptr + pos_base + 0).to(tl.int32)
    y_idx = tl.load(pos_ptr + pos_base + 1).to(tl.int32)
    x_idx = tl.load(pos_ptr + pos_base + 2).to(tl.int32)

    # Clamp indices to valid range
    t_idx = tl.minimum(tl.maximum(t_idx, 0), ft_rows - 1)
    y_idx = tl.minimum(tl.maximum(y_idx, 0), fy_rows - 1)
    x_idx = tl.minimum(tl.maximum(x_idx, 0), fx_rows - 1)

    # 2. Gather frequency values and build combined freq_re, freq_im
    # The freq tables are complex64, stored as pairs of float32 (re, im).
    # Layout: table[row_idx, col_idx] real part at [row_idx * cols * 2 + col_idx * 2]
    #                                  imag part at [row_idx * cols * 2 + col_idx * 2 + 1]

    # Initialize combined freq as (1, 0) — identity for complex multiply
    freq_re = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    freq_im = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ft segment: cols [0, ft_cols)
    ft_mask = cols < ft_cols
    combined_mask_ft = mask & ft_mask
    ft_base = t_idx * ft_cols * 2
    ft_re = tl.load(ft_ptr + ft_base + cols * 2, mask=combined_mask_ft, other=0.0)
    ft_im = tl.load(ft_ptr + ft_base + cols * 2 + 1, mask=combined_mask_ft, other=0.0)
    freq_re = tl.where(combined_mask_ft, ft_re, freq_re)
    freq_im = tl.where(combined_mask_ft, ft_im, freq_im)

    # fy segment: cols [ft_cols, ft_cols + fy_cols)
    fy_local = cols - ft_cols
    fy_mask = (cols >= ft_cols) & (fy_local < fy_cols)
    combined_mask_fy = mask & fy_mask
    fy_base = y_idx * fy_cols * 2
    fy_re = tl.load(fy_ptr + fy_base + fy_local * 2, mask=combined_mask_fy, other=0.0)
    fy_im = tl.load(fy_ptr + fy_base + fy_local * 2 + 1, mask=combined_mask_fy, other=0.0)
    freq_re = tl.where(combined_mask_fy, fy_re, freq_re)
    freq_im = tl.where(combined_mask_fy, fy_im, freq_im)

    # fx segment: cols [ft_cols + fy_cols, ft_cols + fy_cols + fx_cols)
    fx_local = cols - ft_cols - fy_cols
    fx_mask = (cols >= ft_cols + fy_cols) & (fx_local < fx_cols)
    combined_mask_fx = mask & fx_mask
    fx_base = x_idx * fx_cols * 2
    fx_re = tl.load(fx_ptr + fx_base + fx_local * 2, mask=combined_mask_fx, other=0.0)
    fx_im = tl.load(fx_ptr + fx_base + fx_local * 2 + 1, mask=combined_mask_fx, other=0.0)
    freq_re = tl.where(combined_mask_fx, fx_re, freq_re)
    freq_im = tl.where(combined_mask_fx, fx_im, freq_im)

    # 3. Load k pairs (real, imag) and do complex multiply
    k_base = row * k_stride_row
    k_re = tl.load(k_ptr + k_base + cols * 2, mask=mask, other=0.0).to(tl.float32)
    k_im = tl.load(k_ptr + k_base + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    # Complex multiply: (k_re + j*k_im) * (freq_re + j*freq_im)
    out_re = k_re * freq_re - k_im * freq_im
    out_im = k_re * freq_im + k_im * freq_re

    # 4. Store output with dtype cast
    out_base = row * out_stride_row
    if out_dtype_code == 1:
        tl.store(out_ptr + out_base + cols * 2, out_re.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im.to(tl.bfloat16), mask=mask)
    elif out_dtype_code == 2:
        tl.store(out_ptr + out_base + cols * 2, out_re.to(tl.float16), mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + out_base + cols * 2, out_re, mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im, mask=mask)


@triton.jit
def _temporal_rope_delta_kernel(
    k_ptr,
    out_ptr,
    old_t_ptr,
    new_t_ptr,
    ft_ptr,
    ft_rows: tl.constexpr,
    ft_cols: tl.constexpr,
    N,
    k_stride_row,
    out_stride_row,
    old_t_stride,
    new_t_stride,
    out_dtype_code: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < ft_cols

    old_idx = tl.load(old_t_ptr + row * old_t_stride).to(tl.int32)
    new_idx = tl.load(new_t_ptr + row * new_t_stride).to(tl.int32)
    old_idx = tl.minimum(tl.maximum(old_idx, 0), ft_rows - 1)
    new_idx = tl.minimum(tl.maximum(new_idx, 0), ft_rows - 1)

    ft_old_base = old_idx * ft_cols * 2
    ft_new_base = new_idx * ft_cols * 2

    old_re = tl.load(ft_ptr + ft_old_base + cols * 2, mask=mask, other=1.0)
    old_im = tl.load(ft_ptr + ft_old_base + cols * 2 + 1, mask=mask, other=0.0)
    new_re = tl.load(ft_ptr + ft_new_base + cols * 2, mask=mask, other=1.0)
    new_im = tl.load(ft_ptr + ft_new_base + cols * 2 + 1, mask=mask, other=0.0)

    delta_re = new_re * old_re + new_im * old_im
    delta_im = new_im * old_re - new_re * old_im

    k_base = row * k_stride_row
    out_base = row * out_stride_row
    k_re = tl.load(k_ptr + k_base + cols * 2, mask=mask, other=0.0).to(tl.float32)
    k_im = tl.load(k_ptr + k_base + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)

    out_re = k_re * delta_re - k_im * delta_im
    out_im = k_re * delta_im + k_im * delta_re

    if out_dtype_code == 1:
        tl.store(out_ptr + out_base + cols * 2, out_re.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im.to(tl.bfloat16), mask=mask)
    elif out_dtype_code == 2:
        tl.store(out_ptr + out_base + cols * 2, out_re.to(tl.float16), mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + out_base + cols * 2, out_re, mask=mask)
        tl.store(out_ptr + out_base + cols * 2 + 1, out_im, mask=mask)

def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def triton_apply_rope_to_flat_k(
    k_flat: torch.Tensor,
    pos_3d: torch.Tensor,
    freqs: torch.Tensor,
    freq_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for apply_rope_to_flat_k using a fused Triton kernel.

    Args:
        k_flat: [N, head_dim] tensor (bfloat16/float16/float32)
        pos_3d: [N, 3] tensor of (t, y, x) indices (long)
        freqs: [max_pos, C_half] complex64 frequency table
        freq_parts: optional pre-split (ft, fy, fx) each [max_pos_i, cols_i] complex64
        out: optional pre-allocated [N, head_dim] output tensor (same dtype as k_flat)

    Returns:
        [N, head_dim] tensor with RoPE applied, same dtype as k_flat
    """
    if k_flat.numel() == 0:
        return out if out is not None else k_flat

    N, head_dim = k_flat.shape
    assert head_dim % 2 == 0, f"Head dim must be even, got {head_dim}"

    C_half = head_dim // 2
    device = k_flat.device

    # Prepare freq parts
    if freq_parts is not None:
        ft, fy, fx = freq_parts
    else:
        split = [C_half - 2 * (C_half // 3), C_half // 3, C_half // 3]
        ft, fy, fx = freqs.split(split, dim=1)

    ft = ft.to(device=device)
    fy = fy.to(device=device)
    fx = fx.to(device=device)

    # Convert freq tables to contiguous float32 interleaved (re, im) format
    # freq tables are complex64 -> view as float32 pairs
    ft_f32 = torch.view_as_real(ft.to(torch.complex64)).contiguous()  # [rows, cols, 2]
    fy_f32 = torch.view_as_real(fy.to(torch.complex64)).contiguous()
    fx_f32 = torch.view_as_real(fx.to(torch.complex64)).contiguous()

    ft_flat = ft_f32.reshape(ft_f32.shape[0], -1)  # [rows, cols*2]
    fy_flat = fy_f32.reshape(fy_f32.shape[0], -1)
    fx_flat = fx_f32.reshape(fx_f32.shape[0], -1)

    # Ensure pos is contiguous int64
    pos_3d = pos_3d.to(device=device, dtype=torch.long).contiguous()

    # Determine output dtype code
    out_dtype = k_flat.dtype
    if out_dtype == torch.bfloat16:
        dtype_code = 1
    elif out_dtype == torch.float16:
        dtype_code = 2
    else:
        dtype_code = 0

    # Prepare k as contiguous float32 for computation
    # We view k_flat as [N, C_half, 2] pairs
    k_f32 = k_flat.to(torch.float32).contiguous()

    # Use pre-allocated output or allocate new
    if out is None:
        out = torch.empty_like(k_flat)
    out.copy_(k_flat)

    # Compute BLOCK_SIZE (next power of 2 >= C_half)
    BLOCK_SIZE = _next_power_of_2(C_half)

    ft_cols = ft.shape[1] if ft.shape[1] > 0 else 0
    fy_cols = fy.shape[1] if fy.shape[1] > 0 else 0
    fx_cols = fx.shape[1] if fx.shape[1] > 0 else 0

    grid = (N,)
    _fused_rope_3d_kernel[grid](
        k_f32,
        out,
        pos_3d,
        ft_flat,
        fy_flat,
        fx_flat,
        ft_rows=ft.shape[0],
        fy_rows=fy.shape[0],
        fx_rows=fx.shape[0],
        ft_cols=ft_cols,
        fy_cols=fy_cols,
        fx_cols=fx_cols,
        N=N,
        C_half=C_half,
        k_stride_row=k_f32.stride(0),
        out_stride_row=out.stride(0),
        pos_stride_row=pos_3d.stride(0),
        out_dtype_code=dtype_code,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def triton_apply_temporal_rope_delta(
    k_flat: torch.Tensor,
    old_t: torch.Tensor | int,
    new_t: torch.Tensor | int,
    ft: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if k_flat.numel() == 0:
        return out if out is not None else k_flat

    N, head_dim = k_flat.shape
    assert head_dim % 2 == 0, f"Head dim must be even, got {head_dim}"
    if out is None:
        out = torch.empty_like(k_flat)

    device = k_flat.device
    ft = ft.to(device=device)
    ft_f32 = torch.view_as_real(ft.to(torch.complex64)).contiguous()
    ft_flat = ft_f32.reshape(ft_f32.shape[0], -1)

    if isinstance(old_t, torch.Tensor):
        old_t_tensor = old_t.to(device=device, dtype=torch.long).contiguous()
    else:
        old_t_tensor = torch.full((N,), int(old_t), device=device, dtype=torch.long)
    if isinstance(new_t, torch.Tensor):
        new_t_tensor = new_t.to(device=device, dtype=torch.long).contiguous()
    else:
        new_t_tensor = torch.full((N,), int(new_t), device=device, dtype=torch.long)

    out_dtype = k_flat.dtype
    if out_dtype == torch.bfloat16:
        dtype_code = 1
    elif out_dtype == torch.float16:
        dtype_code = 2
    else:
        dtype_code = 0

    k_f32 = k_flat.to(torch.float32).contiguous()
    ft_cols = int(ft.shape[1])
    if ft_cols <= 0:
        out.copy_(k_flat)
        return out

    block_size = _next_power_of_2(ft_cols)
    grid = (N,)
    _temporal_rope_delta_kernel[grid](
        k_f32,
        out,
        old_t_tensor,
        new_t_tensor,
        ft_flat,
        ft_rows=ft.shape[0],
        ft_cols=ft_cols,
        N=N,
        k_stride_row=k_f32.stride(0),
        out_stride_row=out.stride(0),
        old_t_stride=old_t_tensor.stride(0),
        new_t_stride=new_t_tensor.stride(0),
        out_dtype_code=dtype_code,
        BLOCK_SIZE=block_size,
    )
    return out
