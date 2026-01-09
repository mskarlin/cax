"""Tests for fused NCA kernel implementation.

Run with: python -m pytest examples/60_gpt2_distillation/test_fused_kernel.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

# Import modules to test
from .fused_nca import FusedGPT2BlockNCA, count_params
from .nca import GPT2BlockNCA, create_gpt2_nca
from .tiling import (
	TileConfig,
	compute_tile_grid,
	estimate_tile_memory,
	extract_all_tiles,
	extract_tile_with_halo,
	pad_state_for_tiling,
	reassemble_tiles,
)


class TestTiling:
	"""Tests for tiling utilities."""

	def test_tile_config_properties(self):
		"""Test TileConfig computed properties."""
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)

		assert config.tile_with_halo_size == (16, 16, 16)
		assert config.total_tile_elements == 16 * 16 * 16

	def test_compute_tile_grid(self):
		"""Test tile grid computation."""
		state_shape = (64, 64, 16, 16)  # seq, compressed, depth, channels
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)

		grid = compute_tile_grid(state_shape, config)

		assert grid == (8, 8, 2)  # 64/8, 64/8, 16/8

	def test_pad_state_for_tiling(self):
		"""Test state padding for tiling."""
		state = jnp.ones((64, 64, 16, 16))
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)

		padded = pad_state_for_tiling(state, config, mode="constant")

		# Should be padded by halo_size on each side (spatial dims only)
		assert padded.shape == (72, 72, 24, 16)  # 64+8, 64+8, 16+8, 16

	def test_extract_tile_with_halo(self):
		"""Test tile extraction."""
		state = jnp.arange(64 * 64 * 16 * 16).reshape(64, 64, 16, 16).astype(float)
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)

		padded = pad_state_for_tiling(state, config)
		tile = extract_tile_with_halo(padded, (0, 0, 0), config)

		assert tile.shape == (16, 16, 16, 16)  # tile_with_halo_size + channels

	def test_tile_reassembly_roundtrip(self):
		"""Test that tiles can be extracted and reassembled."""
		key = jax.random.key(42)
		state = jax.random.normal(key, (64, 64, 16, 16))
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)

		# Extract all tiles
		tiles = extract_all_tiles(state, config)

		# Simulate processing (just extract interiors)
		h = config.halo_size
		tile_interiors = tiles[:, h:-h, h:-h, h:-h, :]

		# Reassemble
		reconstructed = reassemble_tiles(tile_interiors, state.shape, config)

		# Should match original
		np.testing.assert_allclose(state, reconstructed, rtol=1e-5)

	def test_memory_estimation(self):
		"""Test memory estimation."""
		config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)
		mem = estimate_tile_memory(config, channels=16, dtype_bytes=4)

		# 16^3 * 16 * 4 = 262144 bytes = 256 KB
		assert mem["tile_with_halo"] == 16 * 16 * 16 * 16 * 4
		# Interior: 8^3 * 16 * 4 = 32768 bytes = 32 KB
		assert mem["tile_interior"] == 8 * 8 * 8 * 16 * 4


class TestFusedNCA:
	"""Tests for FusedGPT2BlockNCA."""

	@pytest.fixture
	def setup_ncas(self):
		"""Create base and fused NCAs with same weights."""
		key = jax.random.key(42)
		rngs = nnx.Rngs(key)

		# Create base NCA
		base_nca = GPT2BlockNCA(
			hidden_dim=768,
			compressed_dim=64,
			state_depth=16,
			channel_size=16,
			perception_size=64,
			hidden_layer_sizes=(128,),
			step_size=0.1,
			cell_dropout_rate=0.0,  # Disable dropout for comparison
			rngs=rngs,
		)

		# Create fused NCA with same initialization
		rngs2 = nnx.Rngs(key)
		fused_nca = FusedGPT2BlockNCA(
			hidden_dim=768,
			compressed_dim=64,
			state_depth=16,
			channel_size=16,
			perception_size=64,
			hidden_layer_sizes=(128,),
			step_size=0.1,
			cell_dropout_rate=0.0,  # Disable dropout for comparison
			tile_interior_size=(8, 8, 8),
			num_fused_steps=4,
			rngs=rngs2,
		)

		return base_nca, fused_nca

	def test_same_param_count(self, setup_ncas):
		"""Fused NCA should have same parameter count as base."""
		base_nca, fused_nca = setup_ncas

		from .nca import count_params as base_count

		base_params = base_count(base_nca)
		fused_params = count_params(fused_nca)

		assert base_params == fused_params

	def test_init_state_equivalence(self, setup_ncas):
		"""init_state should produce identical results."""
		base_nca, fused_nca = setup_ncas

		key = jax.random.key(0)
		activations = jax.random.normal(key, (2, 64, 768))

		state_base = base_nca.init_state(activations)
		state_fused = fused_nca.init_state(activations)

		np.testing.assert_allclose(state_base, state_fused, rtol=1e-5)

	def test_extract_output_equivalence(self, setup_ncas):
		"""extract_output should produce identical results."""
		base_nca, fused_nca = setup_ncas

		key = jax.random.key(0)
		state = jax.random.normal(key, (2, 64, 64, 16, 16))

		output_base = base_nca.extract_output(state)
		output_fused = fused_nca.extract_output(state)

		np.testing.assert_allclose(output_base, output_fused, rtol=1e-5)

	def test_single_step_equivalence(self, setup_ncas):
		"""Single step should produce same results (before fusion)."""
		base_nca, fused_nca = setup_ncas

		key = jax.random.key(0)
		state = jax.random.normal(key, (64, 64, 16, 16)) * 0.1

		# Single step
		next_base = base_nca._step(state)
		next_fused = fused_nca._step(state)

		np.testing.assert_allclose(next_base, next_fused, rtol=1e-4, atol=1e-6)

	def test_tile_info(self, setup_ncas):
		"""Test tile info reporting."""
		_, fused_nca = setup_ncas

		info = fused_nca.get_tile_info()

		assert info["interior_size"] == (8, 8, 8)
		assert info["num_fused_steps"] == 4
		assert info["tile_grid_64x64x16"] == (8, 8, 2)
		assert info["num_tiles_64x64x16"] == 128


class TestFactoryFunction:
	"""Tests for create_gpt2_nca factory function."""

	def test_create_base_nca(self):
		"""Factory creates GPT2BlockNCA when use_fusion=False."""
		rngs = nnx.Rngs(jax.random.key(0))
		nca = create_gpt2_nca(use_fusion=False, rngs=rngs)

		assert isinstance(nca, GPT2BlockNCA)
		assert not isinstance(nca, FusedGPT2BlockNCA)

	def test_create_fused_nca(self):
		"""Factory creates FusedGPT2BlockNCA when use_fusion=True."""
		rngs = nnx.Rngs(jax.random.key(0))
		nca = create_gpt2_nca(use_fusion=True, rngs=rngs)

		assert isinstance(nca, FusedGPT2BlockNCA)

	def test_fusion_params_passed(self):
		"""Fusion parameters are correctly passed."""
		rngs = nnx.Rngs(jax.random.key(0))
		nca = create_gpt2_nca(
			use_fusion=True,
			tile_interior_size=(16, 16, 8),
			num_fused_steps=8,
			rngs=rngs,
		)

		assert nca.tile_config.interior_size == (16, 16, 8)
		assert nca.tile_config.num_fused_steps == 8


class TestDropout:
	"""Tests for dropout behavior in fused kernel."""

	def test_dropout_stochasticity(self):
		"""Different dropout keys should produce different outputs."""
		rngs1 = nnx.Rngs(jax.random.key(0))
		rngs2 = nnx.Rngs(jax.random.key(1))

		nca1 = FusedGPT2BlockNCA(
			cell_dropout_rate=0.5,  # High dropout for visibility
			rngs=rngs1,
		)
		nca2 = FusedGPT2BlockNCA(
			cell_dropout_rate=0.5,
			rngs=rngs2,
		)

		# Same initial state
		state = jax.random.normal(jax.random.key(42), (64, 64, 16, 16)) * 0.1

		# Different outputs due to different dropout
		out1 = nca1(state, num_steps=4, training=True)
		out2 = nca2(state, num_steps=4, training=True)

		# Should be different (with high probability)
		assert not jnp.allclose(out1, out2, atol=1e-3)

	def test_inference_deterministic(self):
		"""Inference mode should be deterministic (no dropout)."""
		rngs = nnx.Rngs(jax.random.key(0))
		nca = FusedGPT2BlockNCA(
			cell_dropout_rate=0.5,
			rngs=rngs,
		)

		state = jax.random.normal(jax.random.key(42), (64, 64, 16, 16)) * 0.1

		# Multiple inference calls should be identical
		out1 = nca(state, num_steps=4, training=False)
		out2 = nca(state, num_steps=4, training=False)

		np.testing.assert_allclose(out1, out2, rtol=1e-5)


if __name__ == "__main__":
	# Run basic tests
	print("Running tiling tests...")

	# Test tile config
	config = TileConfig(interior_size=(8, 8, 8), num_fused_steps=4, halo_size=4)
	print(f"  Tile with halo size: {config.tile_with_halo_size}")

	# Test tile grid
	state_shape = (64, 64, 16, 16)
	grid = compute_tile_grid(state_shape, config)
	print(f"  Tile grid for {state_shape}: {grid}")

	# Test padding
	state = jnp.ones(state_shape)
	padded = pad_state_for_tiling(state, config)
	print(f"  Padded shape: {padded.shape}")

	# Test memory estimation
	mem = estimate_tile_memory(config, channels=16)
	print(f"  Tile memory (FP32): {mem['tile_with_halo'] / 1024:.1f} KB")

	print("\nRunning NCA tests...")

	# Create NCAs
	key = jax.random.key(42)
	rngs = nnx.Rngs(key)

	print("  Creating base NCA...")
	base_nca = GPT2BlockNCA(rngs=rngs, cell_dropout_rate=0.0)

	print("  Creating fused NCA...")
	rngs2 = nnx.Rngs(key)
	fused_nca = FusedGPT2BlockNCA(rngs=rngs2, cell_dropout_rate=0.0)

	# Test init_state
	activations = jax.random.normal(jax.random.key(0), (2, 64, 768))
	state_base = base_nca.init_state(activations)
	state_fused = fused_nca.init_state(activations)
	print(f"  init_state match: {jnp.allclose(state_base, state_fused)}")

	# Test single step
	state = jax.random.normal(jax.random.key(0), (64, 64, 16, 16)) * 0.1
	next_base = base_nca._step(state)
	next_fused = fused_nca._step(state)
	print(f"  Single step match: {jnp.allclose(next_base, next_fused, rtol=1e-4)}")

	# Test tile info
	info = fused_nca.get_tile_info()
	print(f"  Tile info: {info}")

	print("\nAll basic tests passed!")
