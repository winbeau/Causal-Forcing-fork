# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Causal Forcing is a research project for real-time streaming video generation built on Wan2.1 text-to-video models. It implements autoregressive diffusion distillation for high-quality interactive video generation, capable of real-time inference on a single RTX 4090.

## Setup

```bash
uv sync
```

Dependencies are managed in `pyproject.toml`. CLIP (from GitHub) and flash-attn (with `--no-build-isolation`) are handled automatically via `[tool.uv.sources]` and `[tool.uv]` config. Run commands with `uv run` (e.g. `uv run python inference.py ...`).

## Commands

### Inference
```bash
# Frame-wise (higher dynamics)
python inference.py \
  --config_path configs/causal_forcing_dmd_framewise.yaml \
  --checkpoint_path checkpoints/framewise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --output_folder output/framewise --use_ema

# Chunk-wise (more stable)
python inference.py \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --data_path prompts/demos.txt \
  --output_folder output/chunkwise
```

### Training (distributed, 8 GPUs)
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR train.py \
  --config_path configs/<config>.yaml --logdir logs/<name>
```

### Demo (Flask web UI)
```bash
python demo.py --port 5001 --checkpoint_path checkpoints/framewise/causal_forcing.pt
```

### Long video generation
```bash
python long_video/inference.py \
  --config_path long_video/configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/chunkwise/longvideo.pt \
  --num_output_frames 252
```

## Architecture

### Three-Stage Training Pipeline

The system trains in three stages, each with its own trainer, model, and config:

1. **Stage 1 - AR Diffusion** (`trainer: "diffusion"`): Trains base autoregressive diffusion on video data. Uses `DiffusionTrainer` + `CausalDiffusion` model. Configs: `ar_diffusion_tf_*.yaml`

2. **Stage 2 - ODE Initialization** (`trainer: "ode"`): Distills the AR model into an ODE solver using paired trajectory data. Uses `ODETrainer` + `ODERegression` model. Configs: `causal_ode_*.yaml`. Alternative: `ConsistencyDistillationTrainer` (`trainer: "consistency_distillation"`) skips ODE data generation.

3. **Stage 3 - DMD** (`trainer: "score_distillation"`): Final distillation via Distribution Matching Distillation. Uses `ScoreDistillationTrainer` + `DMD` model. Configs: `causal_forcing_dmd_*.yaml`

Each stage has **frame-wise** and **chunk-wise** variants. Frame-wise produces more dynamic video; chunk-wise is more stable.

### Model Hierarchy

`BaseModel` (model/base.py) initializes three Wan2.1 diffusion wrappers (generator, real_score, fake_score), a text encoder, a VAE, and a scheduler. `SelfForcingModel` extends it. `DMD` and other distillation models inherit from `SelfForcingModel`.

### Inference Pipelines

- `CausalInferencePipeline` (pipeline/causal_inference.py): Few-step inference with KV-cache. Selected when config has `denoising_step_list`.
- `CausalDiffusionInferencePipeline` (pipeline/causal_diffusion_inference.py): Multi-step diffusion inference. Used during training eval.

### Key Wrappers (utils/wan_wrapper.py)

- `WanDiffusionWrapper`: Wraps the Wan2.1 diffusion model, supports `is_causal` mode
- `WanTextEncoder`: Wraps UMT5-XXL text encoder
- `WanVAEWrapper`: Video VAE for latent space encode/decode

### Configuration

Configs use OmegaConf YAML. `train.py` and `inference.py` both merge `configs/default_config.yaml` with the specified config. The `trainer` field in config selects which Trainer class is instantiated. CLI args override config values.

### Distributed Training

Uses PyTorch FSDP with NCCL backend. Launched via `torchrun`. Key settings in config: `sharding_strategy: hybrid_full`, mixed precision with bfloat16. EMA weights tracked with `ema_weight` parameter.

### Data

- `TextDataset`: Simple text prompt list (for inference and Stage 3)
- `LatentLMDBDataset`: LMDB-backed video latents (Stage 1)
- `ODERegressionLMDBDataset`: ODE paired trajectories (Stage 2)
- `ShardingLMDBDataset`: Multi-shard LMDB for large-scale distributed data

Video latent shape: `(B, T, C=16, H=60, W=104)`. Default resolution: 480x832, 81 frames.

### Long Video (long_video/)

Self-contained module for minute-level generation using Rolling Forcing. Has its own model/, pipeline/, trainer/, wan/ subdirectories mirroring the main structure.

## Key Implementation Details

- Causal masking in `wan/modules/causal_model.py` is the core innovation enabling autoregressive generation
- `num_frame_per_block` controls streaming granularity (speed vs quality tradeoff)
- Low-VRAM mode (<40GB) automatically enables CPU-GPU memory swapping via `DynamicSwapInstaller`
- Checkpoints store both `generator` and `generator_ema` weights; use `--use_ema` for frame-wise inference
- Experiment tracking via Weights & Biases (configure `wandb_host`, `wandb_key`, `wandb_entity`, `wandb_project` in config)
