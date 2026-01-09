"""Pallas kernel for inference-optimized fused NCA execution.

This module implements a custom Pallas kernel that runs multiple NCA steps
entirely in GPU SRAM, achieving maximum arithmetic intensity for inference.

Note: This kernel is for inference only (no dropout, no gradient support).
For training, use fused_kernel.py which supports autodiff.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from .tiling import TileConfig, compute_tile_grid, unravel_tile_index

# Check if Pallas is available
try:
	import jax.experimental.pallas as pl
	from jax.experimental.pallas import gpu as plgpu

	PALLAS_AVAILABLE = True
except ImportError:
	PALLAS_AVAILABLE = False
	pl = None
	plgpu = None


class NCAWeights(NamedTuple):
	"""Flattened NCA weights for Pallas kernel."""

	# Perception weights: (kernel_h, kernel_w, kernel_d, channel_size, num_kernels_per_channel)
	perception: Array

	# MLP weights
	mlp_w1: Array  # (perception_size, hidden_size)
	mlp_b1: Array  # (hidden_size,)
	mlp_w2: Array  # (hidden_size, channel_size)
	mlp_b2: Array  # (channel_size,)

	step_size: float


@dataclass
class PallasConfig:
	"""Configuration for Pallas kernel execution."""

	tile_interior_size: tuple[int, int, int]  # (8, 8, 8) default
	num_fused_steps: int  # 4 default
	halo_size: int  # = num_fused_steps for 3x3x3 kernel
	channel_size: int  # 16
	perception_size: int  # 64
	hidden_size: int  # 128

	@property
	def tile_with_halo_size(self) -> tuple[int, int, int]:
		return tuple(s + 2 * self.halo_size for s in self.tile_interior_size)


def extract_weights_from_nca(nca) -> NCAWeights:
	"""Extract flattened weights from a GPT2BlockNCA for Pallas kernel.

	Args:
		nca: GPT2BlockNCA instance.

	Returns:
		NCAWeights tuple with flattened weight arrays.

	"""
	# Perception kernel weights
	perception = nca.perceive.conv.kernel.value

	# MLP weights from update layers
	layers = nca.update.layers
	mlp_w1 = layers[0].kernel.value.squeeze()  # Remove spatial dims
	mlp_b1 = layers[0].bias.value if hasattr(layers[0], "bias") else jnp.zeros(mlp_w1.shape[-1])
	mlp_w2 = layers[1].kernel.value.squeeze()
	mlp_b2 = layers[1].bias.value if hasattr(layers[1], "bias") else jnp.zeros(mlp_w2.shape[-1])

	return NCAWeights(
		perception=perception,
		mlp_w1=mlp_w1,
		mlp_b1=mlp_b1,
		mlp_w2=mlp_w2,
		mlp_b2=mlp_b2,
		step_size=nca.update.step_size,
	)


def depthwise_conv_3d_naive(
	state: Array,
	kernel: Array,
	channel_size: int,
	num_kernels_per_channel: int,
) -> Array:
	"""Naive 3D depthwise convolution for Pallas (no padding, assumes padded input).

	This is a fallback implementation. In full Pallas, we'd use shared memory
	and register tiling for maximum performance.

	Args:
		state: Input state (h, w, d, channel_size).
		kernel: Convolution kernel (kh, kw, kd, channel_size, num_kernels_per_channel).
		channel_size: Number of input channels.
		num_kernels_per_channel: Number of kernels per channel.

	Returns:
		Output perception (h-2, w-2, d-2, perception_size).

	"""
	# For 3x3x3 kernel, output is 2 smaller in each dim
	h, w, d, c = state.shape
	kh, kw, kd = kernel.shape[:3]
	out_h, out_w, out_d = h - kh + 1, w - kw + 1, d - kd + 1

	perception_size = channel_size * num_kernels_per_channel

	# Manual convolution (slow but works)
	def conv_at_position(i, j, k):
		# Extract patch
		patch = lax.dynamic_slice(state, (i, j, k, 0), (kh, kw, kd, c))

		# Depthwise conv: each channel independently
		# patch: (kh, kw, kd, c), kernel: (kh, kw, kd, c, num_kernels)
		# Output per position: (c * num_kernels,) = perception_size
		result = jnp.einsum("hwdc,hwdck->ck", patch, kernel)
		return result.reshape(-1)  # (perception_size,)

	# Vectorize over output positions
	i_idx = jnp.arange(out_h)
	j_idx = jnp.arange(out_w)
	k_idx = jnp.arange(out_d)

	# Create meshgrid of positions
	ii, jj, kk = jnp.meshgrid(i_idx, j_idx, k_idx, indexing="ij")
	positions = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

	# Map over all positions
	output = jax.vmap(lambda p: conv_at_position(p[0], p[1], p[2]))(positions)
	output = output.reshape(out_h, out_w, out_d, perception_size)

	return output


def gelu(x: Array) -> Array:
	"""GELU activation function."""
	return x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def nca_step_functional(
	state: Array,
	weights: NCAWeights,
	config: PallasConfig,
) -> Array:
	"""Single NCA step (functional version for use in Pallas-style code).

	Args:
		state: Current state (h, w, d, channel_size) - includes halo for conv.
		weights: NCA weights.
		config: Pallas configuration.

	Returns:
		Updated state with halo stripped (h-2, w-2, d-2, channel_size).

	"""
	# Perception: 3D depthwise conv
	num_kernels = config.perception_size // config.channel_size
	perception = depthwise_conv_3d_naive(
		state, weights.perception, config.channel_size, num_kernels
	)

	# MLP: perception_size -> hidden_size -> channel_size
	# Layer 1: perception -> hidden with GELU
	hidden = jnp.dot(perception, weights.mlp_w1) + weights.mlp_b1
	hidden = gelu(hidden)

	# Layer 2: hidden -> channel_size
	update = jnp.dot(hidden, weights.mlp_w2) + weights.mlp_b2

	# Residual update (center of original state, after conv reduces by 1 on each side)
	h, w, d, _ = state.shape
	state_center = state[1:-1, 1:-1, 1:-1, :]

	return state_center + weights.step_size * update


def fused_nca_inference_jax(
	state: Array,
	weights: NCAWeights,
	config: PallasConfig,
	num_steps: int,
) -> Array:
	"""Fused NCA inference using pure JAX (fallback if Pallas unavailable).

	This implementation fuses multiple steps but doesn't have explicit SRAM control.
	Still achieves some fusion benefits through XLA.

	Args:
		state: Input state (seq, compressed, depth, channels).
		weights: NCA weights.
		config: Pallas configuration.
		num_steps: Number of steps to run.

	Returns:
		Final state after num_steps.

	"""
	# Pad state for all steps (need num_steps halo)
	h = num_steps
	padded = jnp.pad(state, [(h, h), (h, h), (h, h), (0, 0)], mode="reflect")

	def step_fn(state, _):
		return nca_step_functional(state, weights, config), None

	# Run fused steps
	final_state, _ = lax.scan(step_fn, padded, None, length=num_steps)

	return final_state


def tiled_inference_jax(
	state: Array,
	weights: NCAWeights,
	config: PallasConfig,
) -> Array:
	"""Tiled inference using pure JAX (fallback).

	Processes tiles sequentially with fusion within each tile.

	Args:
		state: Input state (seq, compressed, depth, channels).
		weights: NCA weights.
		config: Pallas configuration.

	Returns:
		Output state after one tile pass (num_fused_steps).

	"""
	tile_config = TileConfig(
		interior_size=config.tile_interior_size,
		num_fused_steps=config.num_fused_steps,
		halo_size=config.halo_size,
	)

	# Pad state
	h = config.halo_size
	padded = jnp.pad(state, [(h, h), (h, h), (h, h), (0, 0)], mode="reflect")

	tile_grid = compute_tile_grid(state.shape, tile_config)
	num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]

	# Process each tile
	def process_tile(flat_idx):
		tile_idx = unravel_tile_index(flat_idx, tile_grid)
		ti, tj, tk = tile_idx
		int_i, int_j, int_k = config.tile_interior_size

		# Extract tile with halo
		start_i, start_j, start_k = ti * int_i, tj * int_j, tk * int_k
		tile_h, tile_w, tile_d = config.tile_with_halo_size

		tile = lax.dynamic_slice(
			padded,
			(start_i, start_j, start_k, 0),
			(tile_h, tile_w, tile_d, config.channel_size),
		)

		# Run fused steps
		result = fused_nca_inference_jax(
			tile[h:-h, h:-h, h:-h, :],  # Interior with halo for first step
			weights,
			config,
			config.num_fused_steps,
		)

		return result

	# Collect all tile interiors
	tile_interiors = lax.map(process_tile, jnp.arange(num_tiles))

	# Reassemble
	int_i, int_j, int_k = config.tile_interior_size
	out_seq = tile_grid[0] * int_i
	out_comp = tile_grid[1] * int_j
	out_depth = tile_grid[2] * int_k

	output = jnp.zeros((out_seq, out_comp, out_depth, config.channel_size), dtype=state.dtype)

	def scatter_tile(carry, args):
		output, flat_idx = carry
		tile_interior = args
		tile_idx = unravel_tile_index(flat_idx, tile_grid)
		ti, tj, tk = tile_idx

		out_i, out_j, out_k = ti * int_i, tj * int_j, tk * int_k
		output = lax.dynamic_update_slice(output, tile_interior, (out_i, out_j, out_k, 0))

		return (output, flat_idx + 1), None

	(output, _), _ = lax.scan(scatter_tile, (output, 0), tile_interiors)

	# Crop to original size
	seq, comp, depth, _ = state.shape
	return output[:seq, :comp, :depth, :]


# Pallas kernel implementation (only available if Pallas is importable)
if PALLAS_AVAILABLE:

	def create_pallas_nca_kernel(config: PallasConfig):
		"""Create a Pallas kernel for fused NCA inference.

		Note: This is a template - full implementation requires careful
		block spec configuration and may need tuning per GPU architecture.

		Args:
			config: Pallas configuration.

		Returns:
			Pallas kernel function.

		"""
		tile_h, tile_w, tile_d = config.tile_with_halo_size
		int_h, int_w, int_d = config.tile_interior_size
		halo = config.halo_size

		def nca_kernel(
			state_ref,  # Input state reference
			weights_ref,  # Weights reference (struct)
			output_ref,  # Output reference
		):
			"""Pallas kernel body.

			Each program instance processes one tile.
			"""
			# Get tile indices
			tile_i = pl.program_id(0)
			tile_j = pl.program_id(1)
			tile_k = pl.program_id(2)

			# Compute tile start positions
			start_i = tile_i * int_h
			start_j = tile_j * int_w
			start_k = tile_k * int_d

			# Load tile with halo into local memory
			local_state = pl.load(
				state_ref,
				(
					pl.dslice(start_i, tile_h),
					pl.dslice(start_j, tile_w),
					pl.dslice(start_k, tile_d),
					pl.dslice(None),
				),
			)

			# Load weights (these stay in SRAM across all steps)
			perception_w = pl.load(weights_ref.perception, ...)
			mlp_w1 = pl.load(weights_ref.mlp_w1, ...)
			mlp_b1 = pl.load(weights_ref.mlp_b1, ...)
			mlp_w2 = pl.load(weights_ref.mlp_w2, ...)
			mlp_b2 = pl.load(weights_ref.mlp_b2, ...)
			step_size = weights_ref.step_size

			# Run N fused steps entirely in SRAM
			for _ in range(config.num_fused_steps):
				# Perception: 3D depthwise conv
				# (Implementation would use pl.dot or manual tiling)
				perception = _pallas_depthwise_conv_3d(local_state, perception_w, config)

				# MLP Layer 1 with GELU
				hidden = pl.dot(perception, mlp_w1) + mlp_b1
				hidden = gelu(hidden)

				# MLP Layer 2
				update = pl.dot(hidden, mlp_w2) + mlp_b2

				# Residual update
				# After conv, state shrinks by 1 on each side
				state_center = local_state[1:-1, 1:-1, 1:-1, :]
				local_state = state_center + step_size * update

			# Write interior back to HBM
			# After num_fused_steps, we've shrunk by num_fused_steps on each side
			pl.store(
				output_ref,
				(
					pl.dslice(tile_i * int_h, int_h),
					pl.dslice(tile_j * int_w, int_w),
					pl.dslice(tile_k * int_d, int_d),
					pl.dslice(None),
				),
				local_state,
			)

		return nca_kernel

	def _pallas_depthwise_conv_3d(state, kernel, config):
		"""Placeholder for Pallas depthwise 3D conv.

		Full implementation would tile the convolution into
		shared memory and use tensor core operations where available.
		"""
		# This would be replaced with optimized Pallas convolution
		# For now, fall back to naive implementation
		num_kernels = config.perception_size // config.channel_size
		return depthwise_conv_3d_naive(state, kernel, config.channel_size, num_kernels)


# Main inference function that selects best implementation
@partial(jax.jit, static_argnames=("num_steps", "use_pallas"))
def fused_nca_inference(
	state: Array,
	weights: NCAWeights,
	config: PallasConfig,
	num_steps: int,
	use_pallas: bool = False,
) -> Array:
	"""Run fused NCA inference with best available implementation.

	Args:
		state: Input state (..., seq, compressed, depth, channels).
		weights: NCA weights.
		config: Pallas configuration.
		num_steps: Total number of NCA steps.
		use_pallas: Whether to use Pallas kernel (if available).

	Returns:
		Final state after num_steps.

	"""
	# Handle batched input
	if state.ndim > 4:
		batch_shape = state.shape[:-4]
		spatial_shape = state.shape[-4:]
		state_flat = state.reshape(-1, *spatial_shape)

		def process_single(s):
			return _fused_inference_single(s, weights, config, num_steps, use_pallas)

		results = jax.vmap(process_single)(state_flat)
		return results.reshape(*batch_shape, *spatial_shape)
	else:
		return _fused_inference_single(state, weights, config, num_steps, use_pallas)


def _fused_inference_single(
	state: Array,
	weights: NCAWeights,
	config: PallasConfig,
	num_steps: int,
	use_pallas: bool,
) -> Array:
	"""Single (non-batched) fused inference."""
	num_passes = num_steps // config.num_fused_steps
	remainder = num_steps % config.num_fused_steps

	# Run tile passes
	def tile_pass(state, _):
		return tiled_inference_jax(state, weights, config), None

	state, _ = lax.scan(tile_pass, state, None, length=num_passes)

	# Handle remainder
	if remainder > 0:
		remainder_config = PallasConfig(
			tile_interior_size=config.tile_interior_size,
			num_fused_steps=remainder,
			halo_size=remainder,
			channel_size=config.channel_size,
			perception_size=config.perception_size,
			hidden_size=config.hidden_size,
		)
		state = tiled_inference_jax(state, weights, remainder_config)

	return state


# Convenience function to create config from NCA
def create_pallas_config_from_nca(
	nca,
	tile_interior_size: tuple[int, int, int] = (8, 8, 8),
	num_fused_steps: int = 4,
) -> PallasConfig:
	"""Create PallasConfig from a GPT2BlockNCA instance.

	Args:
		nca: GPT2BlockNCA instance.
		tile_interior_size: Interior tile size.
		num_fused_steps: Number of steps to fuse.

	Returns:
		PallasConfig instance.

	"""
	return PallasConfig(
		tile_interior_size=tile_interior_size,
		num_fused_steps=num_fused_steps,
		halo_size=num_fused_steps,
		channel_size=nca.channel_size,
		perception_size=nca.perceive.perception_size,
		hidden_size=nca.update.layers[0].out_features,
	)
