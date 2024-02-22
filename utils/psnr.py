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

"""Helper functions for PSNR computations."""

import jax
import jax.numpy as jnp


mse_fn = lambda x, y: jnp.mean((x - y) ** 2)
mse_fn_jitted = jax.jit(mse_fn)
psnr_fn = lambda mse: -10 * jnp.log10(mse)
psnr_fn_jitted = jax.jit(psnr_fn)
inverse_psnr_fn = lambda psnr: jnp.exp(-psnr * jnp.log(10) / 10)
inverse_psnr_fn_jitted = jax.jit(inverse_psnr_fn)
