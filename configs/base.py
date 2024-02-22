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

"""Base config for C3 experiments."""

from jaxline import base_config
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Return config object for training."""

  # Several config settings are defined internally, use this function to extract
  # them
  config = base_config.get_base_config()

  # Note that we only have training jobs.
  config.eval_modes = ()

  config.binary_args = [
      ('--define cudnn_embed_so', 1),
      ('--define=cuda_compress', 1),
  ]

  # Training loop config
  config.interval_type = 'steps'  # Use steps instead of default seconds
  # The below is the number of times we call the `step` method in
  # `experiment_compression.py`. Think of the `step` method as the effective
  # `main` block of the experiment, where we loop over all train & test images
  # and optimize for each one sequentially.
  config.training_steps = 1
  config.log_train_data_interval = 1
  config.log_tensors_interval = 1
  config.save_checkpoint_interval = -1
  config.checkpoint_dir = '/tmp/training/'
  config.eval_specific_checkpoint_dir = ''

  config.random_seed = 0

  # Create config dict hierarchy.
  config.experiment_kwargs = config_dict.ConfigDict()
  exp = config.experiment_kwargs.config = config_dict.ConfigDict()
  exp.dataset = config_dict.ConfigDict()
  exp.opt = config_dict.ConfigDict()
  exp.loss = config_dict.ConfigDict()
  exp.quant = config_dict.ConfigDict()
  exp.eval = config_dict.ConfigDict()
  exp.model = config_dict.ConfigDict()
  exp.model.synthesis = config_dict.ConfigDict()
  exp.model.latents = config_dict.ConfigDict()
  exp.model.entropy = config_dict.ConfigDict()
  exp.model.upsampling = config_dict.ConfigDict()
  exp.model.quant = config_dict.ConfigDict()

  # Whether to log per-datum metrics.
  exp.log_per_datum_metrics = True
  # Log gradient norms for different sets of params
  exp.log_gradient_norms = False

  return config
