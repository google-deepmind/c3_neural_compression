# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Config for UVG video experiment."""

from ml_collections import config_dict
import numpy as np

from c3_neural_compression.configs import base


def get_config() -> config_dict.ConfigDict:
  """Return config object for training."""

  config = base.get_config()
  exp = config.experiment_kwargs.config

  # Dataset config
  exp.dataset.name = 'uvg'
  # Make sure root_dir matches the directory where data files are stored.
  exp.dataset.root_dir = '/tmp/uvg'
  exp.dataset.num_frames = 30
  # Optionally have data loader return patches of size
  # (num_frames, *spatial_patch_size) for each datum.
  exp.dataset.spatial_patch_size = (180, 240)
  # Set video_idx to only run on a single UVG video. video_idx=5 corresponds
  # to ShakeNDry. If video_idx=None, then run on all videos.
  exp.dataset.video_idx = 5
  # Allow each worker to only train on a subset of data, by skipping
  # `skip_examples` data points and then only training on the next
  # `num_examples` data points. If wanting to train on the whole data,
  # set both values to None.
  exp.dataset.skip_examples = 0
  exp.dataset.num_examples = 1

  # In the case where each data point is a spatiotemporal patch of video,
  # suitable values can be computed for given values of num_frames,
  # spatial_patch_size, num_videos, num_workers and worker_idx as below:
  # exp.dataset.skip_examples, exp.dataset.num_examples = (
  #     worker_start_patch_idx_and_num_patches(
  #         num_frames=exp.dataset.num_frames,
  #         spatial_ps=exp.dataset.spatial_patch_size,
  #         video_indices=exp.dataset.video_idx,
  #         num_workers=1,
  #         worker_idx=0,
  #     )
  # )

  # Optimizer config. This optimizer is used to optimize a COOL-CHIC model for a
  # given image within the `step` method.
  exp.opt.grad_norm_clip = 1e-2
  exp.opt.num_noise_steps = 100_000
  exp.opt.max_num_ste_steps = 10_000
  # Fraction of iterations after which to switch from noise quantization to
  # straight through estimator
  exp.opt.cosine_decay_schedule = True
  exp.opt.cosine_decay_schedule_kwargs = config_dict.ConfigDict()
  exp.opt.cosine_decay_schedule_kwargs.init_value = 1e-2
  # `alpha` refers to the ratio of the final learning rate over the initial
  # learning rate, i.e. it is `end_value / init_value`.
  exp.opt.cosine_decay_schedule_kwargs.alpha = 0.0
  exp.opt.cosine_decay_schedule_kwargs.decay_steps = (
      exp.opt.num_noise_steps
  )  # to keep same schedule as previously
  # Optimization in the STE regime can optionally use a schedule that is
  # determined automatically by tracking the loss
  exp.opt.ste_uses_cosine_decay = False  # whether to continue w/ cosine decay.
  # If `ste_uses_cosine_decay` is `False`, we use adam with constant learning
  # rate that is dropped according to the following parameters
  exp.opt.ste_reset_params_at_lr_decay = False
  exp.opt.ste_num_steps_not_improved = 20
  exp.opt.ste_lr_decay_factor = 0.8
  exp.opt.ste_init_lr = 1e-4
  exp.opt.ste_break_at_lr = 1e-8
  # Frequency with which results are logged during optimization
  exp.opt.noise_log_every = 100
  exp.opt.ste_log_every = 50
  exp.opt.learn_mask_log_every = 100

  # Quantization
  # Noise regime
  exp.quant.noise_quant_type = 'soft_round'
  exp.quant.soft_round_temp_start = 0.3
  exp.quant.soft_round_temp_end = 0.1
  # Which type of noise to use. Defaults to `Uniform(0, 1)` but can be replaced
  # with the Kumaraswamy distribution (with mode at 0.5).
  exp.quant.use_kumaraswamy_noise = True
  exp.quant.kumaraswamy_init_value = 1.75
  exp.quant.kumaraswamy_end_value = 1.0
  exp.quant.kumaraswamy_decay_steps = exp.opt.num_noise_steps
  # STE regime
  exp.quant.ste_quant_type = 'ste_soft_round'
  # Temperature for `ste_quant_type = ste_soft_round`.
  exp.quant.ste_soft_round_temp = 1e-4

  # Loss config
  # Rate-distortion weight used in loss (corresponds to lambda in paper)
  exp.loss.rd_weight = 1e-3
  # Use rd_weight warmup for the noise steps. 0 means no warmup is used.
  exp.loss.rd_weight_warmup_steps = 0

  # Synthesis
  exp.model.synthesis.layers = (12, 12)
  exp.model.synthesis.kernel_shape = 1
  exp.model.synthesis.add_layer_norm = False
  # Range at which we clip output of synthesis network
  exp.model.synthesis.clip_range = (0.0, 1.0)
  exp.model.synthesis.num_residual_layers = 2
  exp.model.synthesis.residual_kernel_shape = 3
  exp.model.synthesis.activation_fn = 'gelu'
  exp.model.synthesis.per_frame_conv = False
  # If `True` the mean RGB values of input video patch are used to initialise
  # the bias of the last layer in the synthesis network.
  exp.model.synthesis.b_last_init_input_mean = False

  # Latents
  exp.model.latents.add_gains = True
  exp.model.latents.learnable_gains = False
  # Options to either set the gains directly (`gain_values`) or modify the
  #   default `gain_factor` (`2**i` for grid `i` in cool-chic).
  #   `gain_values` has to be a list of length `num_grids`.
  # Note: If you want to sweep the `gain_factor`, you have to set it to `0.`
  #   first as otherwise the sweep cannot overwrite them
  exp.model.latents.gain_values = None  # use cool-chic default
  exp.model.latents.gain_factor = None  # use cool-chic default
  exp.model.latents.num_grids = 5
  exp.model.latents.q_step = 0.2
  exp.model.latents.downsampling_factor = (2.0, 2.0, 2.0)
  # Controls how often each grid is downsampled by a factor of
  # `downsampling_factor`, relative to the input resolution.
  # For example, if `downsampling_factor` is 2 and the exponents are (0, 1, 2),
  # the latent grids will have shapes (T // 2**0, H // 2**0, W // 2**0),
  # (T // 2**1, H // 2**1, W // 2**1), and (T // 2**2, H // 2**2, W // 2**2) for
  # a video of shape (T, H, W, 3). A value of None defaults to
  # range(exp.model.latents.num_grids).
  exp.model.latents.downsampling_exponents = None

  # Entropy model
  exp.model.entropy.layers = (12, 12)
  # Defining the below as a tuple allows to sweep tuple values.
  exp.model.entropy.context_num_rows_cols = (2, 2, 2)
  exp.model.entropy.activation_fn = 'gelu'
  exp.model.entropy.scale_range = (1e-3, 150)
  exp.model.entropy.shift_log_scale = 8.0
  exp.model.entropy.clip_like_cool_chic = True
  exp.model.entropy.use_linear_w_init = True
  # Settings related to condition the network on the latent grid in some way. At
  # the moment only `per_grid` is supported, in which a separate entropy model
  # is learned per latent grid.
  exp.model.entropy.conditional_spec = config_dict.ConfigDict()
  exp.model.entropy.conditional_spec.use_conditioning = False
  exp.model.entropy.conditional_spec.type = 'per_grid'

  exp.model.entropy.mask_config = config_dict.ConfigDict()
  # Whether to use the full causal mask for the context (false) or to have a
  # custom mask with a smaller context (true).
  exp.model.entropy.mask_config.use_custom_masking = False
  # When use_custom_masking=True, define the width(=height) of the mask for the
  # context corresponding to the current latent frame.
  exp.model.entropy.mask_config.current_frame_mask_size = 7
  # When use_custom_masking=True, we have the option of learning the contiguous
  # mask for the context corresponding to the previous latent frame.
  exp.model.entropy.mask_config.learn_prev_frame_mask = False
  # When learn_prev_frame_mask=True, define the number of training iterations
  # for which the previous frame mask is learned.
  exp.model.entropy.mask_config.learn_prev_frame_mask_iter = 1000
  # When use_custom_masking=True, define the shape of the contiguous mask that
  # we'll use for the previous frame. This mask can either be used as the
  # learned mask (when learn_prev_frame_mask=True) or a fixed mask defined by
  # prev_frame_mask_top_lefts below (when learn_prev_frame_mask=False)
  exp.model.entropy.mask_config.prev_frame_contiguous_mask_shape = (4, 4)
  # Only include the previous latent frame in the context for the latent grids
  # with indices below. For the other grids, only have the current latent frame
  # (with mask size determined by current_frame_mask_size) as context.
  exp.model.entropy.mask_config.prev_frame_mask_grids = (0, 1, 2)
  # When use_custom_masking=True but learn_prev_frame_mask=False, the below arg
  # defines the top left indices for each grid to be used by a fixed custom
  # rectangular mask of the previous latent frame. Its length should match the
  # length of `prev_frame_mask_grids`. Note these indices are relative to
  # the top left entry of the previous latent frame. e.g., a value of (1, 2)
  # would mean that the mask starts 1 latent pixel below and 2 latent pixels to
  # the right of the top left entry of the previous latent frame context. Note
  # that when learn_prev_frame_mask=True, these values have no effect.
  exp.model.entropy.mask_config.prev_frame_mask_top_lefts = (
      (29, 11),
      (31, 30),
      (60, 1),
  )

  # Upsampling model
  # Only valid option is 'image_resize'
  exp.model.upsampling.type = 'image_resize'
  exp.model.upsampling.kwargs = config_dict.ConfigDict()
  # Choose the interpolation method for 'image_resize'.
  exp.model.upsampling.kwargs.interpolation_method = 'bilinear'

  # Model quantization
  # Range of quantization steps for weights and biases over which to search
  # during post-training quantization step
  # Note COOL-CHIC uses the following
  # POSSIBLE_Q_STEP_ARM_NN = 2. ** torch.linspace(-7, 0, 8, device='cpu')
  # POSSIBLE_Q_STEP_SYN_NN = 2. ** torch.linspace(-16, 0, 17, device='cpu')
  # However, we found experimentally that the used steps are always in the range
  # 1e-5, 1e-2, so no need to sweep over 30 orders of magnitudes as COOL-CHIC
  # does. We currently use the following list, but can experiment with different
  # parameters in sweeps.
  exp.model.quant.q_steps_weight = [5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 6e-3, 1e-2]
  exp.model.quant.q_steps_bias = [5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 6e-3, 1e-2]

  # Prevents accidentally setting keys that aren't recognized (e.g. in tests).
  config.lock()

  return config


def worker_start_patch_idx_and_num_patches(
    num_frames, spatial_ps, video_indices, num_workers, worker_idx
):
  """Compute start patch index and number of patches for given worker.

  Args:
    num_frames: Number of frames in each datum.
    spatial_ps: The spatial dimensions of each datum.
    video_indices: Indices of videos being trained on.
    num_workers: Number of workers to use for training once on the dataset of
      videos specified by `video_indices`. Note that this isn't necessarily
      equal to the number of workers for the whole sweep, since a single sweep
      could also sweep over hyperparams unrelated to the dataset e.g. model
      hyperparams.
    worker_idx: The worker index between 0 and `num_workers-1`.

  Returns:
    worker_start_patch_idx: The patch index to start at for the `worker_idx`th
      worker. Since we use 0-indexing, this is the same as the number of patches
      to skip starting from the first patch.
    worker_num_patches: Number of patches that the `worker_idx`th worker is
      trained on.

  Notes:
    For example, if num_frames = 30, spatial_ps = (180, 240), video_indices=[5],
    num_workers=100, then there are 300*1080*1920/(30*180*240) = 480 patches.
    So each worker gets either 4 or 5 patches. In particular worker_idx=0 would
    have worker_start_patch_idx=0 and worker_num_patches=5 whereas worker_idx=99
    would have worker_start_patch_idx=467 and worker_num_patches=4.
  """
  assert num_workers > 0
  assert 0 <= worker_idx and worker_idx < num_workers
  # Check that spatial_ps are valid by checking they divide the video H and W.
  if spatial_ps:
    assert 1080 % spatial_ps[0] == 0 and 1920 % spatial_ps[1] == 0
    num_spatial_patches = 1080 * 1920 // (spatial_ps[0] * spatial_ps[1])
  else:
    num_spatial_patches = 1
  # Compute total number of frames
  total_num_frames = 0
  for video_idx in video_indices:
    assert video_idx >= 0 and video_idx <= 6
    if video_idx == 5:
      total_num_frames += 300
    else:
      total_num_frames += 600
  assert total_num_frames % num_frames == 0
  num_temporal_patches = total_num_frames // num_frames
  # Compute total number of patches
  num_total_patches = num_spatial_patches * num_temporal_patches
  # Compute all patch indices of worker. Note np.array_split allows cases where
  # `num_total_patches` is not exactly divisible by `num_workers`.
  worker_patch_indices = np.array_split(
      np.arange(num_total_patches), num_workers
  )[worker_idx]
  worker_start_patch_idx = int(worker_patch_indices[0])
  worker_num_patches = worker_patch_indices.size
  return worker_start_patch_idx, worker_num_patches

# Below is pseudocode for the sweep that is used to produce the final results
# in the paper. The sweep covers a single video index and rd weight.

# def c3_sweep(rd_weight, video_index):
#   # Get num_frames and spatial_ps according to rd_weight
#   if rd_weight > 1e-3:  # low bpp regime
#     num_frames = 75
#     spatial_ps = (270, 320)
#   elif rd_weight > 2e-4:  # mid bpp regime
#     num_frames = 60
#     spatial_ps = (180, 240)
#   else:  # rd_weight <= 2e-4: high bpp regime
#     num_frames = 30
#     spatial_ps = (180, 240)
#   assert (
#       300 % num_frames == 0
#   ), f'300 is not divisible by num_frames: {num_frames}.'
#   # Check video_index is valid and obtain num_patches accordingly
#   assert video_index in [0, 1, 2, 3, 4, 5, 6]
#   # Note that video_index=5 (SHAKENDRY) has 300 frames and the rest has 600
#   # frames, where each frame has shape (1080, 1920).
#   total_frames_list = data_loading.DATASET_ATTRIBUTES[
#       'uvg/1080x1920']['frames']
#   num_total_frames = total_frames_list[video_index]
#   num_spatial_patches = 1080 * 1920 // (spatial_ps[0] * spatial_ps[1])
#   num_patches = num_spatial_patches * num_total_frames // num_frames
#   base = 'config.experiment_kwargs.config.'
#   # number of workers to use for fitting a single copy of the dataset.
#   # pylint:disable=g-complex-comprehension
#   tuples_list = [
#       worker_start_patch_idx_and_num_patches(
#           num_frames=num_frames,
#           spatial_ps=spatial_ps,
#           video_indices=(video_index,),
#           num_workers=num_patches,
#           worker_idx=i,
#       )
#       for i in range(num_patches)
#   ]
#   skip_examples, num_examples = zip(*tuples_list)
#   skip_examples = list(skip_examples)
#   num_examples = list(num_examples)
#   # The default sweep uses 3 different settings for conditioning/masking:
#   # 1. no conditioning
#   # 2. per_grid conditioning (separate entropy model per latent grid)
#   # 3. learned contiguous masking with per_grid conditioning.
#   no_cond_sweep = product([
#       sweep(base + 'model.entropy.context_num_rows_cols', [(1, 4, 4)]),
#       sweep(
#           base + 'model.entropy.conditional_spec.use_conditioning', [False]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.use_custom_masking', [False]
#       ),
#       sweep(base + 'model.entropy.layers', [(16, 16)]),
#       sweep(base + 'model.latents.num_grids', [6]),
#   ])
#   cond_sweep = product([
#       sweep(base + 'model.entropy.context_num_rows_cols', [(1, 4, 4)]),
#       sweep(
#           base + 'model.entropy.conditional_spec.use_conditioning', [True]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.use_custom_masking', [False]
#       ),
#       sweep(base + 'model.entropy.layers', [(2, 2)]),
#       sweep(base + 'model.latents.num_grids', [5]),
#   ])
#   learned_mask_cond_sweep = product([
#       sweep(base + 'model.entropy.context_num_rows_cols', [(1, 32, 32)]),
#       sweep(
#           base + 'model.entropy.conditional_spec.use_conditioning', [True]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.use_custom_masking', [True]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.current_frame_mask_size', [7]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.learn_prev_frame_mask', [True]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.learn_prev_frame_mask_iter',
#           [1000]
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.prev_frame_contiguous_mask_shape',
#           [(4, 4)],
#       ),
#       sweep(
#           base + 'model.entropy.mask_config.prev_frame_mask_grids',
#           [(0, 1, 2)]
#       ),
#       sweep(base + 'model.entropy.layers', [(8, 8)]),
#       sweep(base + 'model.latents.num_grids', [6]),
#   ])
#   return product([
#       sweep('config.random_seed', [4]),
#       sweep(base + 'model.synthesis.layers', [(32, 32)]),
#       fixed(base + 'quant.noise_quant_type', 'soft_round'),
#       sweep(base + 'opt.cosine_decay_schedule_kwargs.init_value', [1e-2]),
#       sweep(
#           base + 'opt.grad_norm_clip', [1e-2 if video_index == 0 else 3e-2]
#       ),
#       fixed(base + 'loss.rd_weight_warmup_steps', 0),
#       fixed(base + 'model.synthesis.num_residual_layers', 2),
#       # Note that q_step has not yet been tuned!
#       fixed(base + 'model.latents.q_step', 0.3),
#       sweep(base + 'loss.rd_weight', [rd_weight]),
#       chainit([no_cond_sweep, cond_sweep, learned_mask_cond_sweep]),
#       zipit([
#           sweep(
#               base + 'model.synthesis.per_frame_conv', [True, True, False]
#           ),
#           sweep(
#               base + 'model.synthesis.b_last_init_input_mean',
#               [True, False, True],
#           ),
#       ]),
#       fixed(base + 'dataset.spatial_patch_size', spatial_ps),
#       fixed(base + 'dataset.video_idx', (video_index,)),
#       fixed(base + 'dataset.num_frames', num_frames),
#       zipit([
#           sweep(base + 'dataset.skip_examples', skip_examples),
#           sweep(base + 'dataset.num_examples', num_examples),
#       ]),
#   ])
