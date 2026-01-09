"""Tiling utilities for fused NCA kernel with halo support.

This module provides utilities for spatial tiling of 3D NCA states,
enabling multi-step fusion by processing tiles that fit in on-chip memory.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class TileConfig:
	"""Configuration for spatial tiling with halo.

	Attributes:
		interior_size: Size of tile interior (seq, compressed, depth).
		num_fused_steps: Number of NCA steps to fuse per tile pass.
		halo_size: Size of halo around each tile (typically = num_fused_steps for 3x3x3 kernel).

	"""

	interior_size: tuple[int, int, int]
	num_fused_steps: int
	halo_size: int

	@property
	def tile_with_halo_size(self) -> tuple[int, int, int]:
		"""Size of tile including halo on all sides."""
		return tuple(s + 2 * self.halo_size for s in self.interior_size)

	@property
	def total_tile_elements(self) -> int:
		"""Total elements in tile with halo (excluding channels)."""
		h, w, d = self.tile_with_halo_size
		return h * w * d


def compute_tile_grid(
	state_shape: tuple[int, ...],
	config: TileConfig,
) -> tuple[int, int, int]:
	"""Compute number of tiles needed in each spatial dimension.

	Args:
		state_shape: Shape of state (..., seq_len, compressed_dim, state_depth, channel_size).
		config: Tile configuration.

	Returns:
		Tuple of (num_tiles_seq, num_tiles_compressed, num_tiles_depth).

	"""
	# Extract spatial dimensions (last 4 dims are seq, compressed, depth, channels)
	seq_len = state_shape[-4]
	compressed_dim = state_shape[-3]
	state_depth = state_shape[-2]

	# Compute number of tiles needed (ceiling division)
	num_tiles_seq = (seq_len + config.interior_size[0] - 1) // config.interior_size[0]
	num_tiles_compressed = (compressed_dim + config.interior_size[1] - 1) // config.interior_size[1]
	num_tiles_depth = (state_depth + config.interior_size[2] - 1) // config.interior_size[2]

	return (num_tiles_seq, num_tiles_compressed, num_tiles_depth)


def pad_state_for_tiling(
	state: Array,
	config: TileConfig,
	mode: str = "reflect",
) -> Array:
	"""Pad state with boundary conditions for halo access.

	Pads the spatial dimensions (seq, compressed, depth) with the halo size.
	The channel dimension is not padded.

	Args:
		state: State array with shape (..., seq_len, compressed_dim, state_depth, channel_size).
		config: Tile configuration.
		mode: Padding mode ('reflect', 'edge', or 'constant').

	Returns:
		Padded state with shape (..., seq_len+2*halo, compressed_dim+2*halo,
		state_depth+2*halo, channel_size).

	"""
	h = config.halo_size

	# Compute padding for each dimension
	# Leading batch dimensions get no padding, spatial dims get halo, channels get none
	ndim = state.ndim
	pad_width = [(0, 0)] * (ndim - 4) + [(h, h), (h, h), (h, h), (0, 0)]

	if mode == "constant":
		return jnp.pad(state, pad_width, mode="constant", constant_values=0)
	elif mode == "edge":
		return jnp.pad(state, pad_width, mode="edge")
	elif mode == "reflect":
		return jnp.pad(state, pad_width, mode="reflect")
	else:
		raise ValueError(f"Unknown padding mode: {mode}")


def compute_tile_bounds(
	tile_idx: tuple[int, int, int],
	config: TileConfig,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	"""Compute start and end indices for a tile with halo.

	Args:
		tile_idx: Tile index (i, j, k).
		config: Tile configuration.

	Returns:
		Tuple of (start_indices, sizes) for the tile with halo.
		Start indices are relative to the padded state.

	"""
	ti, tj, tk = tile_idx
	int_i, int_j, int_k = config.interior_size
	h = config.halo_size

	# Start indices in padded state
	# The padded state has halo added, so tile 0's interior starts at index halo
	# But we want to include the halo, so we start at (tile_idx * interior_size)
	start_i = ti * int_i
	start_j = tj * int_j
	start_k = tk * int_k

	# Sizes include halo on both sides
	size_i, size_j, size_k = config.tile_with_halo_size

	return (start_i, start_j, start_k), (size_i, size_j, size_k)


def extract_tile_with_halo(
	padded_state: Array,
	tile_idx: tuple[int, int, int],
	config: TileConfig,
) -> Array:
	"""Extract a single tile with halo from padded state.

	Args:
		padded_state: Padded state array (seq+2h, compressed+2h, depth+2h, channels).
		tile_idx: Tile index (i, j, k).
		config: Tile configuration.

	Returns:
		Tile with halo, shape (interior+2*halo)^3 x channels.

	"""
	starts, sizes = compute_tile_bounds(tile_idx, config)
	si, sj, sk = starts
	ni, nj, nk = sizes
	channels = padded_state.shape[-1]

	# Use dynamic_slice for JIT compatibility
	tile = jax.lax.dynamic_slice(
		padded_state,
		(si, sj, sk, 0),
		(ni, nj, nk, channels),
	)

	return tile


def scatter_tile_interior(
	output: Array,
	tile_interior: Array,
	tile_idx: tuple[int, int, int],
	config: TileConfig,
) -> Array:
	"""Scatter tile interior back to output array.

	Args:
		output: Output array to scatter into (seq, compressed, depth, channels).
		tile_interior: Tile interior to scatter (interior_size x channels).
		tile_idx: Tile index (i, j, k).
		config: Tile configuration.

	Returns:
		Updated output array.

	"""
	ti, tj, tk = tile_idx
	int_i, int_j, int_k = config.interior_size

	# Compute output position
	out_i = ti * int_i
	out_j = tj * int_j
	out_k = tk * int_k

	# Use dynamic_update_slice for JIT compatibility
	output = jax.lax.dynamic_update_slice(
		output,
		tile_interior,
		(out_i, out_j, out_k, 0),
	)

	return output


def unravel_tile_index(
	flat_idx: int,
	tile_grid: tuple[int, int, int],
) -> tuple[int, int, int]:
	"""Convert flat tile index to 3D tile coordinates.

	Args:
		flat_idx: Flat tile index.
		tile_grid: Number of tiles in each dimension (ni, nj, nk).

	Returns:
		Tuple of (i, j, k) tile coordinates.

	"""
	ni, nj, nk = tile_grid
	i = flat_idx // (nj * nk)
	remainder = flat_idx % (nj * nk)
	j = remainder // nk
	k = remainder % nk
	return (i, j, k)


def extract_all_tiles(
	state: Array,
	config: TileConfig,
) -> Array:
	"""Extract all tiles from state (for testing/debugging).

	Args:
		state: State array (seq, compressed, depth, channels).
		config: Tile configuration.

	Returns:
		Array of tiles with shape (num_tiles, tile_h, tile_w, tile_d, channels).

	"""
	padded = pad_state_for_tiling(state, config)
	tile_grid = compute_tile_grid(state.shape, config)
	num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]

	def extract_one(flat_idx):
		tile_idx = unravel_tile_index(flat_idx, tile_grid)
		return extract_tile_with_halo(padded, tile_idx, config)

	tiles = jax.vmap(extract_one)(jnp.arange(num_tiles))
	return tiles


def reassemble_tiles(
	tile_interiors: Array,
	original_shape: tuple[int, ...],
	config: TileConfig,
) -> Array:
	"""Reassemble tile interiors back into full state.

	Args:
		tile_interiors: Array of tile interiors (num_tiles, int_i, int_j, int_k, channels).
		original_shape: Original state shape (..., seq, compressed, depth, channels).
		config: Tile configuration.

	Returns:
		Reassembled state with original shape.

	"""
	# Extract spatial dimensions
	seq_len = original_shape[-4]
	compressed_dim = original_shape[-3]
	state_depth = original_shape[-2]
	channels = original_shape[-1]

	tile_grid = compute_tile_grid(original_shape, config)
	num_tiles = tile_grid[0] * tile_grid[1] * tile_grid[2]

	# Initialize output (may be larger than original if not evenly divisible)
	int_i, int_j, int_k = config.interior_size
	out_seq = tile_grid[0] * int_i
	out_comp = tile_grid[1] * int_j
	out_depth = tile_grid[2] * int_k

	output = jnp.zeros((out_seq, out_comp, out_depth, channels), dtype=tile_interiors.dtype)

	# Scatter each tile
	def scatter_one(carry, tile_data):
		output, flat_idx = carry
		tile_interior = tile_data
		tile_idx = unravel_tile_index(flat_idx, tile_grid)
		output = scatter_tile_interior(output, tile_interior, tile_idx, config)
		return (output, flat_idx + 1), None

	(output, _), _ = jax.lax.scan(scatter_one, (output, 0), tile_interiors)

	# Crop to original size
	output = output[:seq_len, :compressed_dim, :state_depth, :]

	return output


def estimate_tile_memory(
	config: TileConfig,
	channels: int,
	dtype_bytes: int = 4,
) -> dict[str, int]:
	"""Estimate memory usage for a single tile.

	Args:
		config: Tile configuration.
		channels: Number of channels.
		dtype_bytes: Bytes per element (4 for FP32, 2 for FP16/BF16).

	Returns:
		Dictionary with memory estimates in bytes.

	"""
	h, w, d = config.tile_with_halo_size
	int_h, int_w, int_d = config.interior_size

	tile_with_halo_bytes = h * w * d * channels * dtype_bytes
	tile_interior_bytes = int_h * int_w * int_d * channels * dtype_bytes

	return {
		"tile_with_halo": tile_with_halo_bytes,
		"tile_interior": tile_interior_bytes,
		"halo_overhead": tile_with_halo_bytes - tile_interior_bytes,
		"halo_overhead_ratio": (tile_with_halo_bytes / tile_interior_bytes) - 1,
	}
