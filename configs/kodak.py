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

"""Config for KODAK experiment."""

from ml_collections import config_dict

from c3_neural_compression.configs import base


def get_config() -> config_dict.ConfigDict:
  """Return config object for training."""

  config = base.get_config()
  exp = config.experiment_kwargs.config

  # Dataset config
  exp.dataset.name = 'kodak'
  # Make sure root_dir matches the directory where data files are stored.
  exp.dataset.root_dir = '/tmp/kodak'
  exp.dataset.skip_examples = 0
  exp.dataset.num_examples = 1  # Set this to None to train on whole dataset.
  exp.dataset.num_frames = None
  exp.dataset.spatial_patch_size = None
  exp.dataset.video_idx = None

  # Optimizer config. This optimizer is used to optimize a COOL-CHIC model for a
  # given image within the `step` method.
  exp.opt.grad_norm_clip = 1e-1
  exp.opt.num_noise_steps = 100_000
  exp.opt.max_num_ste_steps = 10_000
  # Optimization in the noise quantization regime uses a cosine decay learning
  # rate schedule
  exp.opt.cosine_decay_schedule = True
  exp.opt.cosine_decay_schedule_kwargs = config_dict.ConfigDict()
  exp.opt.cosine_decay_schedule_kwargs.init_value = 1e-2
  # `alpha` refers to the ratio of the final learning rate over the initial
  # learning rate, i.e., it is `end_value / init_value`.
  exp.opt.cosine_decay_schedule_kwargs.alpha = 0.0
  exp.opt.cosine_decay_schedule_kwargs.decay_steps = (
      exp.opt.num_noise_steps
  )  # to keep same schedule as previously
  # Optimization in the STE regime can optionally use a schedule that is
  # determined automatically by tracking the loss
  exp.opt.ste_uses_cosine_decay = False  # whether to continue w/ cosine decay.
  # If `ste_uses_cosine_decay` is `False`, we use adam with a learning rate
  # that is dropped after a few steps without improvement, according to the
  # following configs:
  # Whether parameters and state are reset to the previous best when the
  # learning rate is decayed.
  exp.opt.ste_reset_params_at_lr_decay = True
  # Number of steps without improvement before learning rate is dropped.
  exp.opt.ste_num_steps_not_improved = 20
  exp.opt.ste_lr_decay_factor = 0.8
  exp.opt.ste_init_lr = 1e-4
  exp.opt.ste_break_at_lr = 1e-8
  # Frequency with which results are logged during optimization
  exp.opt.noise_log_every = 100
  exp.opt.ste_log_every = 50

  # Quantization
  # Noise regime
  exp.quant.noise_quant_type = 'soft_round'
  exp.quant.soft_round_temp_start = 0.3
  exp.quant.soft_round_temp_end = 0.1
  # Which type of noise to use. Defaults to the Kumaraswamy distribution (with
  # mode at 0.5), and changes to `Uniform(0, 1)` when
  # `use_kumaraswamy_noise=False`.
  exp.quant.use_kumaraswamy_noise = True
  exp.quant.kumaraswamy_init_value = 2.0
  exp.quant.kumaraswamy_end_value = 1.0
  exp.quant.kumaraswamy_decay_steps = exp.opt.num_noise_steps
  # STE regime
  exp.quant.ste_quant_type = 'ste_soft_round'
  # Temperature for `ste_quant_type = ste_soft_round`.
  exp.quant.ste_soft_round_temp = 1e-4

  # Loss config
  # Rate-distortion weight used in loss (corresponds to lambda in paper)
  exp.loss.rd_weight = 0.001
  # Use rd_weight warmup for the noise steps. 0 means no warmup is used.
  exp.loss.rd_weight_warmup_steps = 0

  # Synthesis
  exp.model.synthesis.layers = (18, 18)
  exp.model.synthesis.kernel_shape = 1
  exp.model.synthesis.add_layer_norm = False
  # Range at which we clip output of synthesis network
  exp.model.synthesis.clip_range = (0.0, 1.0)
  exp.model.synthesis.num_residual_layers = 2
  exp.model.synthesis.residual_kernel_shape = 3
  exp.model.synthesis.activation_fn = 'gelu'
  # If True adds a nonlinearity between linear and residual layers in synthesis
  exp.model.synthesis.add_activation_before_residual = False
  # If True the mean RGB values of input image are used to initialise the bias
  # of the last layer in the synthesis network.
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
  exp.model.latents.num_grids = 7
  exp.model.latents.q_step = 0.4
  exp.model.latents.downsampling_factor = 2.  # Use same factor for h & w.
  # Controls how often each grid is downsampled by a factor of
  # `downsampling_factor`, relative to the input resolution.
  # For example, if `downsampling_factor` is 2 and the exponents are (0, 1, 2),
  # the latent grids will have shapes (H // 2**0, W // 2**0),
  # (H // 2**1, W // 2**1), and (H // 2**2, W // 2**2) for an image of shape
  # (H, W, 3). A value of None defaults to range(exp.model.latents.num_grids).
  exp.model.latents.downsampling_exponents = tuple(
      range(exp.model.latents.num_grids)
  )

  # Entropy model
  exp.model.entropy.layers = (18, 18)
  exp.model.entropy.context_num_rows_cols = (3, 3)
  exp.model.entropy.activation_fn = 'gelu'
  exp.model.entropy.scale_range = (1e-3, 150)
  exp.model.entropy.shift_log_scale = 8.
  exp.model.entropy.clip_like_cool_chic = True
  exp.model.entropy.use_linear_w_init = True
  # Settings related to condition the network on the latent grid in some way. At
  # the moment only `use_prev_grid` is supported.
  exp.model.entropy.conditional_spec = config_dict.ConfigDict()
  exp.model.entropy.conditional_spec.use_conditioning = False
  # Whether to condition the entropy model on the previous grid. If this is
  # `True`, the parameter `conditional_spec.prev_kernel_shape` should be set.
  exp.model.entropy.conditional_spec.use_prev_grid = False
  exp.model.entropy.conditional_spec.interpolation = 'bilinear'
  exp.model.entropy.conditional_spec.prev_kernel_shape = (3, 3)

  # Upsampling model
  # Only valid option is 'image_resize'
  exp.model.upsampling.type = 'image_resize'
  exp.model.upsampling.kwargs = config_dict.ConfigDict()
  # Choose the interpolation method for 'image_resize'. Currently only
  # 'bilinear' is supported because we only define MACs for this case.
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

# Below is pseudocode for the sweeps that are used to produce the final results
# in the paper.

# def c3_sweep():
#   base = 'config.experiment_kwargs.config.'
#   return product([
#       sweep(base + 'dataset.skip_examples', list(range(24))),
#       sweep(base + 'dataset.num_examples', [1]),
#       sweep(base + 'loss.rd_weight',
#           [
#               0.0001,
#               0.0002,
#               0.0003,
#               0.0004,
#               0.0005,
#               0.0008,
#               0.001,
#               0.002,
#               0.003,
#               0.004,
#               0.005,
#               0.008,
#               0.01,
#               0.02,
#           ],
#       ),
#   ])


# def c3_adaptive_sweep():
#   base = 'config.experiment_kwargs.config.'
#   return hyper.product([
#       sweep(
#           base + 'model.entropy.context_num_rows_cols', [(2, 2), (3, 3)]
#       ),
#       sweep(
#           base + 'model.latents.downsampling_exponents',
#           [
#               (0, 1, 2, 3, 4, 5, 6),  # default: highest grid res = image res
#               (1, 2, 3, 4, 5, 6, 7),  # all grids are downsampled once more
#           ],
#       ),
#       zipit([
#           sweep(
#               base + 'model.entropy.layers', [(12, 12), (18, 18), (24, 24)]
#           ),
#           sweep(
#               base + 'model.synthesis.layers', [(12, 12), (18, 18), (24, 24)]
#           ),
#       ]),
#       sweep(base + 'dataset.skip_examples', list(range(24))),
#       sweep(base + 'dataset.num_examples', [1]),
#       sweep(
#           base + 'loss.rd_weight',
#           [
#               0.0001,
#               0.0002,
#               0.0003,
#               0.0004,
#               0.0005,
#               0.0008,
#               0.001,
#               0.002,
#               0.003,
#               0.004,
#               0.005,
#               0.008,
#               0.01,
#               0.02,
#           ],
#       ),
#   ])
