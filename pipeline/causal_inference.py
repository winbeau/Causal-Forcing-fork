from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from headkv import AdaptiveKVCache, HeadKVCache, HeadKVConfig, build_compositions
from headkv.config import validate_headkv_matrix_csv_shape
from pipeline.headkv_config import HeadKVPipelineConfig

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
import tqdm

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.use_headkv = getattr(args, "use_headkv", False)
        self.headkv_config = (
            HeadKVPipelineConfig.from_args(args, frame_seq_length=self.frame_seq_length)
            if self.use_headkv else None
        )

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _validate_headkv_shapes(self) -> None:
        if not self.use_headkv:
            return

        hc = self.headkv_config
        num_layers = self.generator.model.num_layers
        num_heads = self.generator.model.num_heads
        checked_paths = set()
        for label, csv_path in (
            ("config", hc.headkv_config_path),
            ("policy", hc.headkv_policy_csv_path),
        ):
            if not csv_path or csv_path in checked_paths:
                continue
            validate_headkv_matrix_csv_shape(
                csv_path,
                num_layers,
                num_heads,
                label=label,
            )
            checked_paths.add(csv_path)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf = False 
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            context_len = 0
            if self.use_headkv and self.headkv_config.headkv_is_i2v and initial_latent is not None:
                context_len = self.headkv_config.headkv_context_len
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                context_len=context_len,
                max_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
                if self.use_headkv:
                    self.crossattn_cache[block_index]["prompt_v"] = None
            # reset kv cache — distinguish HeadKV caches from legacy dict caches
            if self.kv_cache1 and isinstance(self.kv_cache1[0], HeadKVCache):
                for cache in self.kv_cache1:
                    cache.reset()
            else:
                for block_index in range(len(self.kv_cache1)):
                    self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_update_mode="clean",
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm.tqdm(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_update_mode="noisy",
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_update_mode="noisy",
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                cache_update_mode="clean",
                skip_x0=True,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if rectified_tf: 
            mean = torch.load('laboratory/mean.pt').to(output.device) 
            std = torch.load('laboratory/std.pt').to(output.device) 
            noise = torch.randn_like(output).to(output.device) 
            output -= mean 
        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, context_len=0, max_frames=None):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        When ``self.use_headkv`` is True, build a per-layer ``AdaptiveKVCache`` (or
        plain ``HeadKVCache`` for the uniform baseline) using the pyramid config.
        Otherwise fall back to the legacy dict-based sliding-window cache.
        """
        if self.use_headkv:
            hc = self.headkv_config
            num_layers = self.generator.model.num_layers
            num_heads = self.generator.model.num_heads
            head_dim = self.generator.model.dim // num_heads
            self._validate_headkv_shapes()
            if self.local_attn_size != -1:
                base_capacity_tokens = self.local_attn_size * self.frame_seq_length
            else:
                base_capacity_tokens = 32760
                if max_frames is not None:
                    base_capacity_tokens = max(base_capacity_tokens, max_frames * self.frame_seq_length)

            default_capacity = hc.headkv_default_capacity or base_capacity_tokens
            config = HeadKVConfig(
                hc.headkv_config_path,
                num_layers=num_layers,
                num_heads=num_heads,
                default_capacity=default_capacity,
                strategy_reduction_factor=hc.headkv_strategy_factor,
                code_map=hc.headkv_code_map,
                head_type_csv_path=hc.headkv_policy_csv_path,
                drop_heads_csv_path=hc.headkv_drop_heads_csv_path,
                soft_ablate_heads_csv_path=hc.headkv_soft_ablate_csv_path,
                af_policy_enabled=hc.headkv_af_policy_enabled,
                af_csv_path=hc.headkv_af_csv_path,
                af_group_dir=hc.headkv_af_group_dir,
                af_manifest_path=hc.headkv_af_manifest_path,
                frame_seq_length=hc.headkv_frame_seq_length,
            )
            if hc.use_adaptive_headkv and hc.headkv_policy_csv_path:
                compositions = build_compositions(
                    num_layers=num_layers,
                    num_heads=num_heads,
                    capacities=config.capacity_map,
                    csv_path=hc.headkv_policy_csv_path,
                    cyclic_enabled=hc.cyclic_enabled,
                    cyclic_period=hc.cyclic_period,
                    cyclic_bucket_cap=hc.cyclic_bucket_cap,
                    cyclic_dynamic_rope=hc.cyclic_dynamic_rope,
                    cyclic_osc_only=hc.cyclic_osc_only,
                    lag_enabled=hc.lag_enabled,
                    lag_offsets=hc.headkv_lag_offsets,
                    lag_history=hc.headkv_lag_history,
                    lag_dynamic_rope=hc.lag_dynamic_rope,
                    stride_enabled=hc.stride_enabled,
                    stride_interval=hc.stride_interval,
                    stride_capacity=hc.stride_capacity,
                    stride_dynamic_rope=hc.stride_dynamic_rope,
                    merge_enabled=hc.merge_enabled,
                    merge_patch_size=hc.merge_patch_size,
                    merge_capacity=hc.merge_capacity,
                    merge_dynamic_rope=hc.merge_dynamic_rope,
                    osc_sink_frames=hc.headkv_osc_sink_frames,
                    stable_sink_frames=hc.headkv_stable_sink_frames,
                    recent_frames=hc.headkv_recent_frames,
                    stable_recent_frames=hc.headkv_stable_recent_frames,
                    label_sink_frames_map=hc.headkv_label_sink_frames_map,
                    label_recent_frames_map=hc.headkv_label_recent_frames_map,
                    label_stride_enabled_map=hc.headkv_label_stride_enabled_map,
                    label_stride_interval_map=hc.headkv_label_stride_interval_map,
                    label_phase_bucket_map=hc.headkv_label_phase_bucket_map,
                    label_lag_offsets_map=hc.headkv_label_lag_offsets_map,
                    label_merge_enabled_map=hc.headkv_label_merge_enabled_map,
                    label_merge_patch_size_map=hc.headkv_label_merge_patch_size_map,
                    label_merge_capacity_map=hc.headkv_label_merge_capacity_map,
                )
                config.compositions = compositions
                config.policies = compositions
            self.kv_cache1 = [
                (
                    AdaptiveKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        sink_len=hc.headkv_sink_tokens,
                        tail_len=hc.headkv_dynamic_capacity,
                        ivc_ratio=hc.ivc_ratio,
                        semantic_ratio=hc.semantic_ratio,
                        trajectory_ratio=hc.trajectory_ratio,
                        trajectory_weight=hc.trajectory_weight,
                        history_frame_quota=hc.history_frame_quota,
                        history_quota_ivc_ratio=hc.history_quota_ivc_ratio,
                        post_train_stabilize_t=hc.post_train_stabilize_t,
                        post_train_trajectory_scale=hc.post_train_trajectory_scale,
                        post_train_history_ivc_ratio=hc.post_train_history_ivc_ratio,
                        update_interval=hc.update_interval,
                        seed_ratio=hc.semantic_seed_ratio,
                        sink_grid_decoupling=hc.sink_grid_decoupling,
                        decoupled_sink_tokens=hc.decoupled_sink_tokens,
                        decoupled_sink_time_lag=hc.decoupled_sink_time_lag,
                        sink_time_mapping_mode=hc.headkv_dynamic_rope_mode,
                        sink_time_clamp_min=hc.sink_time_clamp_min,
                        sink_time_clamp_max=hc.sink_time_clamp_max,
                        history_time_mapping_mode=hc.history_time_mapping_mode,
                        history_relative_t_max=hc.history_relative_t_max,
                        history_time_soft_factor=hc.history_time_soft_factor,
                        use_osc_frame_mode=hc.cyclic_enabled,
                        phase_period=hc.cyclic_period,
                        phase_bucket_capacity_frames=hc.cyclic_bucket_cap,
                        local_tail_frames=hc.headkv_recent_frames,
                        phase_sink_for_osc_only=hc.cyclic_osc_only,
                        phase_sink_dynamic_rope=hc.cyclic_dynamic_rope,
                        use_osc_lag_mode=hc.lag_enabled,
                        osc_lag_offsets_frames=hc.headkv_lag_offsets,
                        osc_lag_history_frames=hc.headkv_lag_history,
                        osc_lag_dynamic_rope=hc.lag_dynamic_rope,
                        disable_first_sink_for_osc_heads=hc.headkv_disable_osc_sink,
                        use_stable_head_policies=hc.headkv_stable_policy_enabled,
                        stable_sink_frames=hc.headkv_stable_sink_frames,
                        osc_sink_frames=hc.headkv_osc_sink_frames,
                        stable_recent_frames=hc.headkv_stable_recent_frames,
                        use_af_head_policies=hc.headkv_af_policy_enabled,
                        af_recent_frames_map=hc.headkv_af_recent_frames_map,
                        af_phase_bucket_map=hc.headkv_af_phase_bucket_map,
                        af_lag_offsets_map=hc.headkv_af_lag_offsets_map,
                        af_sink_frames_map=hc.headkv_af_sink_frames_map,
                        af_stride_enabled_map=hc.headkv_af_stride_enabled_map,
                        label_recent_frames_map=hc.headkv_label_recent_frames_map,
                        label_phase_bucket_map=hc.headkv_label_phase_bucket_map,
                        label_lag_offsets_map=hc.headkv_label_lag_offsets_map,
                        label_sink_frames_map=hc.headkv_label_sink_frames_map,
                        label_stride_enabled_map=hc.headkv_label_stride_enabled_map,
                        capture_frame_id_mode=hc.headkv_capture_frame_id_mode,
                        readout_cache_enabled=hc.headkv_readout_cache_enabled,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                    if hc.use_adaptive_headkv else
                    HeadKVCache(
                        config=config,
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        layer_idx=layer_idx,
                        is_i2v=hc.headkv_is_i2v,
                        context_len=context_len,
                        frame_seq_length=hc.headkv_frame_seq_length,
                        prompt_value_cache_enabled=hc.headkv_prompt_v_cache_enabled,
                    )
                )
                for layer_idx in range(num_layers)
            ]
            for cache in self.kv_cache1:
                cache.soft_ablate_region = str(hc.headkv_soft_ablate_region)
                cache.soft_ablate_scale = float(hc.headkv_soft_ablate_scale)
            return

        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
            if max_frames is not None:
                kv_cache_size = max(kv_cache_size, max_frames * self.frame_seq_length)

        for _ in range(self.num_transformer_blocks):
            num_heads = self.generator.model.num_heads
            head_dim = self.generator.model.dim // num_heads
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads
        text_len = self.generator.model.text_len

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, text_len, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
                "prompt_v": None,
            })
        self.crossattn_cache = crossattn_cache
