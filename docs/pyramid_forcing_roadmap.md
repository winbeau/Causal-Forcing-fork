# Pyramid-Forcing on Causal-Forcing-fork — Implementation Roadmap

> **Living document.** Updated after every completed step. Last updated: 2026-04-19.

---

## 1. Overall plan

### Goal
Port AdaHead's **Pyramid-Forcing** per-head adaptive KV caching method (`AdaptiveKVCache`, cyclic + stride + merge + dynamic RoPE) onto Causal-Forcing-fork, as an **inference-time** optimization.

### Scope (approved)
| Aspect | Decision |
|---|---|
| Variant | **chunkwise only** (`num_frame_per_block=3`) |
| Components | full G6: cyclic + stride + merge + dynamic RoPE + readout/prompt-v caches |
| Source | copy `AdaHead/headkv/` wholesale |
| Training | **no changes** — inference path only |
| Long-video | **out of scope** (`long_video/` subdir untouched) |

### Final deliverable
1. A new config `configs/causal_forcing_dmd_pyramid.yaml` that enables per-head adaptive cache.
2. Running `python inference.py --config_path configs/causal_forcing_dmd_pyramid.yaml --checkpoint_path checkpoints/chunkwise/causal_forcing.pt --data_path prompts/demos.txt --output_folder output/pyramid` produces videos comparable in quality to the baseline chunkwise run with ~92% KV memory reduction.
3. All existing configs (`_framewise.yaml`, `_chunkwise.yaml`) behave **bit-identically** (HeadKV gated by `use_headkv: false` default).

### High-level architecture diff

```
Before:                                    After:
┌──────────────┐                          ┌──────────────┐
│  inference   │                          │  inference   │
│  pipeline    │                          │  pipeline    │
│              │  kv_cache = dict list    │              │  kv_cache = list[dict] OR list[AdaptiveKVCache]
└──────┬───────┘                          └──────┬───────┘
       ▼                                         ▼
┌──────────────┐                          ┌──────────────┐
│ WanWrapper   │                          │ WanWrapper   │  + cache_update_mode, skip_x0, prompt_v
└──────┬───────┘                          └──────┬───────┘
       ▼                                         ▼
┌──────────────┐                          ┌──────────────┐
│ CausalWan    │                          │ CausalWan    │  isinstance(kv_cache, HeadKVCache)?
│ SelfAttn     │──> attention(q,k,v)      │ SelfAttn     │──┬── headkv_attention(...)  [new HeadKV path]
└──────────────┘                          └──────────────┘  └── attention(q,k,v)       [legacy dict path]
```

---

## 2. Known / anticipated difficulties

| # | Issue | Mitigation |
|---|-------|------------|
| D1 | AdaHead `wan/modules/attention/` is a **package** (core.py etc.); Causal-Forcing-fork has a **flat** `wan/modules/attention.py` module. | Port `headkv_attention` as a single function into the flat file. Adjust relative imports of helpers (`flash_attn_varlen_func`, etc.). |
| D2 | `CausalWanModel.forward` in Causal-Forcing-fork has extra branches (`classify_mode`, `clean_x` TF path) absent in AdaHead. | Thread `cache_update_mode` / `prompt_v` only through the `kv_cache is not None` path. Training / TF / classify branches get default values. |
| D3 | Current pipeline writes the dict cache on every forward. HeadKV needs `noisy` on denoising passes and `clean` on the commit pass (double-pass semantics). | Add `cache_update_mode` kwarg to all generator calls in `pipeline/causal_inference.py`. Default `"default"` = read-only; explicit `"noisy"` / `"clean"` at call sites. |
| D4 | `context_noise` is used on the post-denoise commit pass but isn't in existing Causal-Forcing configs. | Add `context_noise: 0` to `configs/default_config.yaml` and to the new pyramid config. |
| D5 | `headkv/triton_rope.py` imports Triton at module load — may fail in envs without Triton. | Guard the import with try/except inside `headkv/__init__.py`; fall back to the Python RoPE path (already the AdaHead behavior). |
| D6 | FSDP `hybrid_full` sharding wraps nn.Modules; `AdaptiveKVCache` is a plain Python object, not a module. | It's held on the pipeline (`self.kv_cache1`), not on the model, so FSDP ignores it. No change needed — but verify empirically. |
| D7 | `pyproject.toml` / `uv.lock` doesn't include any headkv-adjacent deps. | `flash-attn` is already a dep (used by `attention.py`). `pandas` may be needed for CSV parsing in `factory.py` — check and add if missing. |
| D8 | `CausalWanModel.forward` in Causal-Forcing-fork has some signature shape we haven't fully diffed yet. | Read the full forward before threading kwargs; keep existing positional order intact. |
| D9 | AdaHead's `headkv/factory.py` expects a CSV with specific row/col layout (30×12). | Runtime fix: `_initialize_kv_cache` now reads `num_layers` / `num_heads` from the loaded generator, and `use_headkv=true` hard-fails when a matrix CSV shape does not match. |
| D10 | Configs still carry `real_name: Wan2.1-T2V-14B`, but the current causal inference path instantiates `WanDiffusionWrapper` from `model_kwargs` and otherwise falls back to `Wan2.1-T2V-1.3B`. | Decision for this batch: keep the existing 1.3B inference runtime and make it explicit in `configs/causal_forcing_dmd_pyramid.yaml` via `model_kwargs.model_name: Wan2.1-T2V-1.3B`. A true 14B inference-path cleanup remains separate future work. |

### Actual difficulties encountered

- **[Step 2]** AdaHead's `headkv_attention` in `wan/modules/attention/core.py` has two orthogonal debug systems baked in (`FRAME_ATTENTION_CAPTURE` for frame-level attention capture, `soft_ablate_*` for head-level ablation). Both are gated by flags that default to off at runtime. Decision: drop both from the Causal-Forcing port — they aren't part of pyramid-forcing inference, and pulling them in would require also copying `wan/modules/attention/capture.py` (466 lines). This keeps the flat `attention.py` layout. If ablation/capture tooling is ever needed, port `capture.py` separately.

---

## 3. Per-step plan

### Step 0 — This document (**DONE**)
- **Objective**: create the living roadmap that tracks the port.
- **Files**: `docs/pyramid_forcing_roadmap.md` (this file).
- **Status**: ✅ done.

### Step 1 — Copy `headkv/` package + headkv_config.py + classification CSV
- **Objective**: bring all HeadKV implementation code into Causal-Forcing-fork as a self-contained, importable package.
- **Touched files**:
  - new `headkv/` directory (14 files from `AdaHead/headkv/`)
  - new `pipeline/headkv_config.py`
  - new `configs/head_configs/classification_results.csv`
- **Key moves**:
  - `cp -r AdaHead/headkv Causal-Forcing-fork/headkv`
  - `cp AdaHead/pipeline/headkv_config.py Causal-Forcing-fork/pipeline/headkv_config.py`
  - `cp AdaHead/configs/head_configs/classification_results.csv Causal-Forcing-fork/configs/head_configs/`
- **Invariants**: package is self-contained; only internal imports (`from .base import ...`). No edits to package code.
- **Verification**: `python -c "from headkv import AdaptiveKVCache, HeadKVCache, HeadKVConfig; print('ok')"` from inside `Causal-Forcing-fork/`.

### Step 2 — Port `headkv_attention()` into flat `wan/modules/attention.py`
- **Objective**: expose the attention dispatch that `AdaptiveKVCache` can plug into.
- **Touched files**: `wan/modules/attention.py`.
- **Source**: `AdaHead/wan/modules/attention/core.py` (the `headkv_attention` function, ~line 228 onward).
- **Key moves**: copy the function body verbatim; rewrite relative imports to match the flat module layout; keep existing `attention()` / `flash_attention()` untouched.
- **Invariants**: baseline `attention()` behavior bit-identical.
- **Verification**: `python -c "from wan.modules.attention import attention, headkv_attention; print('ok')"`.

### Step 3 — Dispatch in `CausalWanSelfAttention.forward` + thread through model
- **Objective**: when the caller passes an `AdaptiveKVCache`/`HeadKVCache` instead of a dict, route to `headkv_attention`.
- **Touched files**: `wan/modules/causal_model.py`.
- **Key moves**:
  - import `headkv_attention` and `HeadKVCache`.
  - Add `cache_update_mode="default"`, `prompt_v=None` to `CausalWanSelfAttention.forward`.
  - Insert `if isinstance(kv_cache, HeadKVCache):` branch inside the `else` path (lines 197-239) that calls `headkv_attention`.
  - Thread the kwargs through `CausalWanAttentionBlock.forward` and `CausalWanModel.forward`.
- **Invariants**: dict-cache path unchanged; TF branch unchanged.
- **Verification**: chunkwise inference with `use_headkv: false` still produces bit-identical videos vs. before changes.

### Step 4 — Add `cache_update_mode` / `skip_x0` to `WanDiffusionWrapper.forward`
- **Objective**: expose the two kwargs the pipeline will use to drive the double-pass.
- **Touched files**: `utils/wan_wrapper.py`.
- **Key moves**: add `cache_update_mode: str = "default"`, `skip_x0: bool = False` kwargs; forward them into `self.model(...)` in the `kv_cache is not None` branch.
- **Invariants**: TF path, classify-mode path untouched.

### Step 5 — Pipeline wiring (`_initialize_kv_cache`, mode-per-call, reset)
- **Objective**: build `AdaptiveKVCache`s when `use_headkv=True`, emit correct `cache_update_mode` at each call site.
- **Touched files**: `pipeline/causal_inference.py`.
- **Key moves**:
  - `__init__`: read `use_headkv`; instantiate `HeadKVPipelineConfig`.
  - `_initialize_kv_cache`: mirror AdaHead's setup (`HeadKVConfig`, `build_compositions`, `AdaptiveKVCache` per layer).
  - Model-shape cleanup: derive layer/head counts from `self.generator.model` instead of hardcoding `30/12`; legacy KV and cross-attn caches now use runtime `num_heads`, `head_dim`, and `text_len`.
  - Safety check: when `use_headkv=true`, matrix CSVs used by HeadKV (`headkv_config_path`, `headkv_policy_csv_path`) must exactly match the loaded model shape or raise a clear error.
  - Reset logic: call `cache.reset()` for HeadKV, keep dict-reset for legacy.
  - Denoising loop: `cache_update_mode="noisy"` on both generator calls.
  - Commit pass: `cache_update_mode="clean"`, `skip_x0=True`.
  - Initial-latent seeding: `cache_update_mode="clean"`.
- **Invariants**: when `use_headkv=False`, pipeline behavior bit-identical.

### Step 6 — New `configs/causal_forcing_dmd_pyramid.yaml`
- **Objective**: ship a ready-to-run pyramid config.
- **Template**: `causal_forcing_dmd_chunkwise.yaml` + pyramid block from `AdaHead/configs/pyramid-forcing.yaml:54-139`.
- **Key moves**: make the current runtime target explicit with `model_kwargs.model_name: Wan2.1-T2V-1.3B`; add `context_noise: 0`; enable the G6 pyramid HeadKV block.
- **Verification**: run `python inference.py --config_path configs/causal_forcing_dmd_pyramid.yaml ...` and inspect output.

### Step 7 — `configs/default_config.yaml` touch-up
- **Objective**: keep OmegaConf merges clean for non-pyramid configs.
- **Touched files**: `configs/default_config.yaml`.
- **Key moves**: add `use_headkv: false` and `context_noise: 0`. Other HeadKV keys read via `getattr(args, ..., default)` so they need no default.

---

## 4. Status

| Step | State | Notes |
|---|---|---|
| 0. Roadmap doc | ✅ done | this file |
| 1. Copy `headkv/` | ✅ done | package + `pipeline/headkv_config.py` + `configs/head_configs/classification_results.csv` copied; `from headkv import ...` verified under `.venv/bin/python` |
| 2. `headkv_attention` | ✅ done | ported to flat `wan/modules/attention.py`; debug hooks (`FRAME_ATTENTION_CAPTURE`, `soft_ablate_*`) intentionally dropped — not required for pyramid-forcing inference. Verified `from wan.modules.attention import headkv_attention`. |
| 3. Dispatch in model | ✅ done | `CausalWanSelfAttention.forward` gains `prompt_v=None, cache_update_mode="default"`; HeadKV branch dispatches to `headkv_attention` honoring `kv_cache.post_prune_rope`. `CausalWanAttentionBlock.forward` extracts `prompt_v` from `cross_attn.v(context)` (respecting `prompt_value_cache_enabled`) and threads kwargs down. `_forward_inference` forwards `cache_update_mode` into every block call (both gradient-checkpointed and plain paths). `from wan.modules.causal_model import CausalWanModel, CausalWanAttentionBlock, CausalWanSelfAttention` verified. |
| 4. Wrapper kwargs | ✅ done | `WanDiffusionWrapper.forward` gains `cache_update_mode: str = "default"` and `skip_x0: bool = False`. The `kv_cache is not None` branch forwards `cache_update_mode` to `self.model(...)`. When `skip_x0=True`, the wrapper returns `(flow_pred, None)` and skips the `_convert_flow_pred_to_x0` call — used by the clean commit pass. Teacher-forcing and plain-diffusion branches untouched. |
| 5. Pipeline wiring | ✅ done | `pipeline/causal_inference.py` now builds HeadKV caches, resets them correctly, emits `cache_update_mode`/`skip_x0` at the right call sites, uses runtime model dimensions for both legacy and HeadKV caches, and hard-fails on matrix CSV/model shape mismatch. |
| 6. Pyramid config | ✅ done | added `configs/causal_forcing_dmd_pyramid.yaml` with the G6 HeadKV block and explicit `model_kwargs.model_name: Wan2.1-T2V-1.3B` to match the current inference runtime. |
| 7. Default-config touch | ✅ done | `configs/default_config.yaml` now includes `use_headkv: false` in addition to `context_noise: 0`, keeping legacy configs gated and merge-clean. |

---

## 5. Verification plan (end-to-end)

### Regression (ensure no behavior change for legacy path)
```bash
cd Causal-Forcing-fork
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --output_folder output/regression
```
Compare against a pre-port baseline run (same seed, same prompts).

### Pyramid path
```bash
python inference.py \
  --config_path configs/causal_forcing_dmd_pyramid.yaml \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --output_folder output/pyramid
```
Expected: finite output, no NaNs, KV cache memory usage significantly lower (log on first block).

### Sanity checks inside pipeline
- After first block commit: `kv_cache1[0].global_end_index == num_frame_per_block * frame_seqlen == 3 * 1560 == 4680`.
- `isinstance(kv_cache1[0], AdaptiveKVCache) == True` when `use_headkv=true`.

### Ablations (optional, mirrors AdaHead ablation scripts)
- `cyclic_enabled: false` only
- `stride_enabled: false` only
- `merge_enabled: false` only
- `use_adaptive_headkv: false` (uniform HeadKV baseline)

Each should still produce output; compare visual quality vs. full pyramid.
