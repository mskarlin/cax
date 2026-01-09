"""Pure JAX fused NCA kernel for training.

This module implements multi-step NCA fusion using JAX primitives (lax.scan)
with gradient checkpointing support. Suitable for training with autodiff.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax

from .tiling import (
	TileConfig,
	compute_tile_grid,
	extract_tile_with_halo,
	pad_state_for_tiling,
	reassemble_tiles,
	unravel_tile_index,
)


def fused_nca_steps(
	perceive_fn: Callable[[Array], Array],
	update_fn: Callable[[Array, Array, Array | None], Array],
	state: Array,
	num_steps: int,
	*,
	dropout_key: Array | None = None,
	use_checkpointing: bool = True,
) -> Array:
	"""Run multiple NCA steps fused together with lax.scan.

	This keeps intermediate states in registers/cache rather than writing
	back to HBM after each step, significantly improving arithmetic intensity.

	Args:
		perceive_fn: Perception function (state -> perception).
		update_fn: Update function (state, perception, key) -> next_state.
		state: Current state array.
		num_steps: Number of NCA steps to fuse.
		dropout_key: Optional RNG key for dropout (splits per step).
		use_checkpointing: Whether to use gradient checkpointing.

	Returns:
		Final state after num_steps.

	"""

	def single_step(carry: tuple[Array, Array | None], _: Any) -> tuple[tuple[Array, Array | None], None]:
		state, key = carry

		# Split key for this step
		if key is not None:
			key, subkey = jax.random.split(key)
		else:
			subkey = None

		# Perceive and update
		perception = perceive_fn(state)
		next_state = update_fn(state, perception, subkey)

		return (next_state, key), None

	# Apply checkpointing if requested (trades compute for memory in backward pass)
	if use_checkpointing:
		single_step = jax.checkpoint(single_step)

	# Run fused steps
	(final_state, _), _ = lax.scan(
		single_step,
		(state, dropout_key),
		None,
		length=num_steps,
	)

	return final_state


def process_tile(
	perceive_fn: Callable[[Array], Array],
	update_fn: Callable[[Array, Array, Array | None], Array],
	tile_with_halo: Array,
	config: TileConfig,
	dropout_key: Array | None = None,
	use_checkpointing: bool = True,
) -> Array:
	"""Process a single tile through num_fused_steps NCA steps.

	Args:
		perceive_fn: Perception function.
		update_fn: Update function.
		tile_with_halo: Tile including halo region.
		config: Tile configuration.
		dropout_key: Optional RNG key for dropout.
		use_checkpointing: Whether to use gradient checkpointing.

	Returns:
		Tile interior after processing (halo stripped).

	"""
	# Run fused steps on tile
	result = fused_nca_steps(
		perceive_fn,
		update_fn,
		tile_with_halo,
		config.num_fused_steps,
		dropout_key=dropout_key,
		use_checkpointing=use_checkpointing,
	)

	# Strip halo to get interior
	h = config.halo_size
	interior = result[h:-h, h:-h, h:-h, :]

	return interior


def tiled_fused_forward(
	state: Array,
	perceive_fn: Callable[[Array], Array],
	update_fn: Callable[[Array, Array, Array | None], Array],
	config: TileConfig,
	dropout_key: Array | None = None,
	use_checkpointing: bool = True,
	pad_mode: str = "reflect",
) -> Array:
	"""Full forward pass with tiled fusion.

	Processes all tiles through num_fused_steps, then reassembles.
	This is one "tile pass" - for total_steps NCA steps, call this
	(total_steps // num_fused_steps) times.

	Args:
		state: Input state (..., seq, compressed, depth, channels).
		perceive_fn: Perception function (local convolution).
		update_fn: Update function (MLP + residual).
		config: Tile configuration.
		dropout_key: Optional RNG key for dropout.
		use_checkpointing: Whether to use gradient checkpointing.
		pad_mode: Boundary padding mode.

	Returns:
		Output state after one tile pass (num_fused_steps total).

	"""
	# Handle batched input
	if state.ndim > 4:
		# Vectorize over batch dimensions
		batch_shape = state.shape[:-4]
		spatial_shape = state.shape[-4:]

		# Flatten batch dims
		state_flat = state.reshape(-1, *spatial_shape)
		batch_size = state_flat.shape[0]

		# Process each batch element
		if dropout_key is not None:
			keys = jax.random.split(dropout_key, batch_size)
		else:
			keys = [None] * batch_size

		def process_batch_elem(s, k):
			return _tiled_fused_forward_single(
				s, perceive_fn, update_fn, config, k, use_checkpointing, pad_mode
			)

		# Use vmap for efficiency
		if dropout_key is not None:
			results = jax.vmap(
				lambda s, k: process_batch_elem(s, k),
				in_axes=(0, 0),
			)(state_flat, jnp.stack(keys))
		else:
			results = jax.vmap(
				lambda s: process_batch_elem(s, None),
				in_axes=(0,),
			)(state_flat)

		# Reshape back to original batch shape
		return results.reshape(*batch_shape, *spatial_shape)
	else:
		return _tiled_fused_forward_single(
			state, perceive_fn, update_fn, config, dropout_key, use_checkpointing, pad_mode
		)


def _tiled_fused_forward_single(
	state: Array,
	perceive_fn: Callable[[Array], Array],
	update_fn: Callable[[Array, Array, Array | None], Array],
	config: TileConfig,
	dropout_key: Array | None,
	use_checkpointing: bool,
	pad_mode: str,
) -> Array:
	"""Process a single (non-batched) state with tiled fusion."""
	# Pad state for halo access
	padded = pad_state_for_tiling(state, config, mode=pad_mode)

	# Compute tile grid
	tile_grid = compute_tile_grid(state.shape, config)
	num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]

	# Split keys for each tile
	if dropout_key is not None:
		tile_keys = jax.random.split(dropout_key, num_tiles)
	else:
		tile_keys = None

	def process_one_tile(flat_idx: int, key: Array | None) -> Array:
		"""Process a single tile."""
		tile_idx = unravel_tile_index(flat_idx, tile_grid)
		tile = extract_tile_with_halo(padded, tile_idx, config)
		return process_tile(
			perceive_fn, update_fn, tile, config, key, use_checkpointing
		)

	# Process all tiles
	# Use lax.map for memory efficiency (sequential processing)
	# Alternative: vmap for speed if memory allows
	if tile_keys is not None:
		tile_interiors = lax.map(
			lambda args: process_one_tile(args[0], args[1]),
			(jnp.arange(num_tiles), tile_keys),
		)
	else:
		tile_interiors = lax.map(
			lambda idx: process_one_tile(idx, None),
			jnp.arange(num_tiles),
		)

	# Reassemble tiles
	output = reassemble_tiles(tile_interiors, state.shape, config)

	return output


def create_update_fn_with_dropout(
	mlp_fn: Callable[[Array], Array],
	step_size: float,
	dropout_rate: float,
) -> Callable[[Array, Array, Array | None], Array]:
	"""Create an update function with dropout support for fused kernel.

	This replaces the nnx.Dropout which requires module state, with a
	functional dropout that takes an explicit key.

	Args:
		mlp_fn: MLP function (perception -> update delta).
		step_size: Residual step size.
		dropout_rate: Cell dropout rate.

	Returns:
		Update function compatible with fused kernel.

	"""

	def update_fn(state: Array, perception: Array, key: Array | None) -> Array:
		# Compute update through MLP
		update = mlp_fn(perception)

		# Apply cell dropout if key provided and rate > 0
		if key is not None and dropout_rate > 0:
			keep_prob = 1.0 - dropout_rate
			# Dropout mask broadcast across channels (same mask for all channels of a cell)
			mask_shape = update.shape[:-1]  # All dims except channels
			mask = jax.random.bernoulli(key, keep_prob, shape=mask_shape)
			mask = mask[..., None]  # Add channel dim for broadcasting
			update = update * mask / keep_prob  # Scale to maintain expected value

		# Residual update
		return state + step_size * update

	return update_fn


@partial(jax.jit, static_argnames=("num_steps", "use_checkpointing", "pad_mode"))
def multi_tile_pass_forward(
	state: Array,
	perceive_fn: Callable[[Array], Array],
	update_fn: Callable[[Array, Array, Array | None], Array],
	config: TileConfig,
	num_steps: int,
	dropout_key: Array | None = None,
	use_checkpointing: bool = True,
	pad_mode: str = "reflect",
) -> Array:
	"""Run multiple tile passes to achieve total num_steps.

	Args:
		state: Input state.
		perceive_fn: Perception function.
		update_fn: Update function.
		config: Tile configuration.
		num_steps: Total number of NCA steps.
		dropout_key: Optional RNG key.
		use_checkpointing: Whether to checkpoint.
		pad_mode: Boundary padding mode.

	Returns:
		Final state after num_steps.

	"""
	num_passes = num_steps // config.num_fused_steps
	remainder = num_steps % config.num_fused_steps

	# Split keys for each pass
	if dropout_key is not None:
		pass_keys = jax.random.split(dropout_key, num_passes + (1 if remainder > 0 else 0))
	else:
		pass_keys = [None] * (num_passes + 1)

	# Run main tile passes
	def tile_pass_step(carry, pass_key):
		state = carry
		state = tiled_fused_forward(
			state, perceive_fn, update_fn, config, pass_key, use_checkpointing, pad_mode
		)
		return state, None

	state, _ = lax.scan(
		tile_pass_step,
		state,
		jnp.stack(pass_keys[:num_passes]) if dropout_key is not None else None,
		length=num_passes,
	)

	# Handle remainder steps (if num_steps not divisible by num_fused_steps)
	if remainder > 0:
		# Create temporary config with fewer fused steps
		remainder_config = TileConfig(
			interior_size=config.interior_size,
			num_fused_steps=remainder,
			halo_size=remainder,  # Halo = steps for 3x3x3 kernel
		)
		state = tiled_fused_forward(
			state,
			perceive_fn,
			update_fn,
			remainder_config,
			pass_keys[-1] if dropout_key is not None else None,
			use_checkpointing,
			pad_mode,
		)

	return state
