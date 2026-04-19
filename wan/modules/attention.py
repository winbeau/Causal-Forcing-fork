# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'headkv_attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def headkv_attention(
    q,
    k,
    v,
    kv_cache,
    current_start=None,
    grid_sizes=None,
    freqs=None,
    start_frame=0,
    prompt_v=None,
    cache_update_mode="default",
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=True,
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    Per-head adaptive KV-cache attention via FlashAttention varlen.

    kv_cache: HeadKVCache (or AdaptiveKVCache) holding per-head ragged KV.
    cache_update_mode: "default" | "noisy" | "clean". Drives whether the update
        commits to the clean slot, a noisy scratch slot, or is read-only.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, h, d = q.shape
    out_dtype = q.dtype

    drop_head_mask = None
    raw_drop_mask = getattr(kv_cache, "drop_head_mask", None)
    if raw_drop_mask is not None:
        drop_head_mask = torch.as_tensor(raw_drop_mask, dtype=torch.bool, device=q.device)
        if drop_head_mask.numel() != h:
            drop_head_mask = None

    frame_seqlen = None
    if grid_sizes is not None:
        frame_tokens = (grid_sizes[:, 1] * grid_sizes[:, 2]).to(torch.long)
        if torch.any(frame_tokens <= 0):
            raise ValueError(f"Invalid frame token sizes: {frame_tokens.tolist()}")
        if torch.unique(frame_tokens).numel() != 1:
            raise ValueError(
                f"Mixed frame token sizes in batch are not supported: {frame_tokens.tolist()}"
            )
        frame_seqlen = int(frame_tokens[0].item())

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    kv_cache.update(
        k,
        v,
        current_start=current_start,
        grid_sizes=grid_sizes,
        freqs=freqs,
        start_frame=start_frame,
        prompt_v=prompt_v,
        cache_update_mode=cache_update_mode,
    )

    if fa_version is not None and fa_version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 is not available, use flash attention 2 instead.')

    def run_varlen(
        q_chunk,
        k_flat_chunk,
        v_flat_chunk,
        cu_seqlens_k_chunk,
        max_seqlen_k_chunk,
        cu_seqlens_q_override=None,
    ):
        lq_chunk = q_chunk.shape[1]
        q_flat_chunk = q_chunk.transpose(1, 2).reshape(b * h * lq_chunk, d)
        q_flat_chunk = half(q_flat_chunk).unsqueeze(1)
        k_flat_chunk = half(k_flat_chunk).unsqueeze(1)
        v_flat_chunk = half(v_flat_chunk).unsqueeze(1)

        if q_scale is not None:
            q_flat_chunk = q_flat_chunk * q_scale

        q_flat_chunk = q_flat_chunk.to(v_flat_chunk.dtype)
        k_flat_chunk = k_flat_chunk.to(v_flat_chunk.dtype)

        if cu_seqlens_q_override is not None:
            cu_seqlens_q_chunk = cu_seqlens_q_override
        else:
            cu_seqlens_q_chunk = torch.arange(
                0, (b * h + 1) * lq_chunk, step=lq_chunk,
                dtype=torch.int32, device=q.device,
            )

        if (fa_version is None or fa_version == 3) and FLASH_ATTN_3_AVAILABLE:
            out_chunk = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0]
        else:
            assert FLASH_ATTN_2_AVAILABLE
            out_chunk = flash_attn.flash_attn_varlen_func(
                q=q_flat_chunk,
                k=k_flat_chunk,
                v=v_flat_chunk,
                cu_seqlens_q=cu_seqlens_q_chunk,
                cu_seqlens_k=cu_seqlens_k_chunk,
                max_seqlen_q=lq_chunk,
                max_seqlen_k=max_seqlen_k_chunk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                deterministic=deterministic,
            )

        out = out_chunk.squeeze(1).reshape(b, h, lq_chunk, d).transpose(1, 2)
        if drop_head_mask is not None and torch.any(drop_head_mask):
            out[:, :, drop_head_mask, :] = 0
        return out

    use_decoupled = (
        getattr(kv_cache, "post_prune_rope", False)
        and getattr(kv_cache, "sink_grid_decoupling", False)
        and hasattr(kv_cache, "get_decoupled_flat_kv")
    )
    if use_decoupled:
        if freqs is None:
            raise ValueError("freqs is required when sink_grid_decoupling=True")
        if grid_sizes is None:
            raise ValueError("grid_sizes is required when sink_grid_decoupling=True")
        if frame_seqlen is None:
            raise ValueError("frame_seqlen is required when sink_grid_decoupling=True")
        if lq % frame_seqlen != 0:
            raise ValueError(f"q length {lq} must be divisible by frame_seqlen {frame_seqlen}.")

        out_buf = torch.empty(b, lq, h, d, device=q.device, dtype=out_dtype)
        base_start = int(current_start or 0)
        cu_seqlens_q_fixed = torch.arange(
            0, (b * h + 1) * frame_seqlen, step=frame_seqlen,
            dtype=torch.int32, device=q.device,
        )
        for offset in range(0, lq, frame_seqlen):
            q_chunk = q[:, offset:offset + frame_seqlen]
            if hasattr(kv_cache, "get_decoupled_flat_kv_and_frames"):
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k, _k_frame_ids_flat = \
                    kv_cache.get_decoupled_flat_kv_and_frames(
                        current_start=base_start + offset,
                        grid_sizes=grid_sizes,
                        freqs=freqs,
                    )
            else:
                k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_decoupled_flat_kv(
                    current_start=base_start + offset,
                    grid_sizes=grid_sizes,
                    freqs=freqs,
                )
            out_buf[:, offset:offset + frame_seqlen] = run_varlen(
                q_chunk, k_flat, v_flat, cu_seqlens_k, max_seqlen_k,
                cu_seqlens_q_override=cu_seqlens_q_fixed,
            )
        return out_buf

    if getattr(kv_cache, "post_prune_rope", False):
        if not hasattr(kv_cache, "get_flat_kv_and_pos"):
            raise ValueError(
                "kv_cache must provide get_flat_kv_and_pos or get_decoupled_flat_kv for post-prune RoPE."
            )
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k, pos_ids = kv_cache.get_flat_kv_and_pos()
        if freqs is None:
            raise ValueError("freqs is required when post_prune_rope=True")
        if not hasattr(kv_cache, "apply_rope_to_flat_k"):
            raise ValueError("kv_cache must provide apply_rope_to_flat_k for post-prune RoPE.")
        k_flat = kv_cache.apply_rope_to_flat_k(k_flat, pos_ids, freqs=freqs)
    else:
        k_flat, v_flat, cu_seqlens_k, max_seqlen_k = kv_cache.get_flat_kv()

    out = run_varlen(q, k_flat, v_flat, cu_seqlens_k, max_seqlen_k)
    return out.type(out_dtype)
