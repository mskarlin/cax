"""Fused GPT-2 Block NCA with tiled execution for high arithmetic intensity.

This module provides FusedGPT2BlockNCA, a drop-in replacement for GPT2BlockNCA
that uses spatial tiling and multi-step fusion to achieve ~235+ FLOPs/byte
arithmetic intensity (vs ~17 FLOPs/byte naive).

Two execution paths:
- Training: Pure JAX with lax.scan and gradient checkpointing (fused_kernel.py)
- Inference: Optimized functional path with optional Pallas (pallas_kernel.py)
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State

from .fused_kernel import (
	TileConfig,
	create_update_fn_with_dropout,
	tiled_fused_forward,
)
from .nca import GPT2BlockNCA
from .pallas_kernel import (
	NCAWeights,
	PallasConfig,
	create_pallas_config_from_nca,
	extract_weights_from_nca,
	fused_nca_inference,
)


class FusedGPT2BlockNCA(ComplexSystem):
	"""GPT-2 Block NCA with fused tiled execution.

	This class wraps a standard GPT2BlockNCA and provides optimized execution
	through spatial tiling and multi-step fusion. The key benefits are:

	1. Higher arithmetic intensity (~235+ FLOPs/byte vs ~17 naive)
	2. Compute-bound execution on modern GPUs
	3. Full training support with gradient checkpointing
	4. Optimized inference path (optional Pallas kernel)

	The API is identical to GPT2BlockNCA for easy drop-in replacement.

	"""

	def __init__(
		self,
		hidden_dim: int = 768,
		compressed_dim: int = 64,
		state_depth: int = 16,
		channel_size: int = 16,
		perception_size: int = 64,
		hidden_layer_sizes: tuple[int, ...] = (128,),
		*,
		kernel_size: tuple[int, int, int] = (3, 3, 3),
		step_size: float = 0.1,
		cell_dropout_rate: float = 0.1,
		# Fusion configuration
		tile_interior_size: tuple[int, int, int] = (8, 8, 8),
		num_fused_steps: int = 4,
		use_gradient_checkpointing: bool = True,
		pad_mode: str = "reflect",
		rngs: nnx.Rngs,
	):
		"""Initialize the Fused GPT-2 Block NCA.

		Args:
			hidden_dim: Original hidden dimension of transformer (768 for GPT-2).
			compressed_dim: Compressed dimension for NCA grid (e.g., 64).
			state_depth: Depth of the 3D state grid (e.g., 16).
			channel_size: Number of channels in the NCA state.
			perception_size: Size of the perception vector.
			hidden_layer_sizes: Sizes of hidden layers in the update MLP.
			kernel_size: Size of the 3D convolution kernel.
			step_size: Step size for residual updates.
			cell_dropout_rate: Dropout rate for cell updates.
			tile_interior_size: Interior size of each tile (seq, compressed, depth).
			num_fused_steps: Number of NCA steps to fuse per tile pass.
			use_gradient_checkpointing: Whether to checkpoint for memory efficiency.
			pad_mode: Boundary padding mode ('reflect', 'edge', 'constant').
			rngs: RNG key for initialization.

		"""
		# Create base NCA (contains all learnable parameters)
		self.base_nca = GPT2BlockNCA(
			hidden_dim=hidden_dim,
			compressed_dim=compressed_dim,
			state_depth=state_depth,
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			kernel_size=kernel_size,
			step_size=step_size,
			cell_dropout_rate=cell_dropout_rate,
			rngs=rngs,
		)

		# Fusion configuration
		self.tile_config = TileConfig(
			interior_size=tile_interior_size,
			num_fused_steps=num_fused_steps,
			halo_size=num_fused_steps,  # 1 cell per step for 3x3x3 kernel
		)

		self.use_gradient_checkpointing = use_gradient_checkpointing
		self.pad_mode = pad_mode
		self.cell_dropout_rate = cell_dropout_rate
		self.step_size = step_size

		# Store rngs for dropout
		self._rngs = rngs

		# Cache for inference weights (lazy initialization)
		self._inference_weights: NCAWeights | None = None
		self._pallas_config: PallasConfig | None = None

	# Delegate properties to base NCA
	@property
	def hidden_dim(self) -> int:
		return self.base_nca.hidden_dim

	@property
	def compressed_dim(self) -> int:
		return self.base_nca.compressed_dim

	@property
	def state_depth(self) -> int:
		return self.base_nca.state_depth

	@property
	def channel_size(self) -> int:
		return self.base_nca.channel_size

	@property
	def perceive(self):
		return self.base_nca.perceive

	@property
	def update(self):
		return self.base_nca.update

	@property
	def project_in(self):
		return self.base_nca.project_in

	@property
	def project_out(self):
		return self.base_nca.project_out

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Single NCA step (delegates to base NCA for non-fused execution).

		Note: For fused execution, use __call__ directly instead of _step.
		This is provided for API compatibility but doesn't benefit from fusion.

		"""
		return self.base_nca._step(state, input, sow=sow)

	def _create_perceive_fn(self) -> callable:
		"""Create a functional perceive function for the fused kernel."""

		def perceive_fn(state: Array) -> Array:
			return self.base_nca.perceive(state)

		return perceive_fn

	def _create_update_fn(self, training: bool = True) -> callable:
		"""Create a functional update function for the fused kernel.

		For training, creates a function that accepts an RNG key for dropout.
		For inference, creates a function with no dropout.

		"""
		# Get MLP function from update module
		def mlp_fn(perception: Array) -> Array:
			# Apply MLP layers without dropout (dropout handled separately)
			x = perception
			for layer in self.base_nca.update.layers[:-1]:
				x = self.base_nca.update.activation_fn(layer(x))
			return self.base_nca.update.layers[-1](x)

		if training:
			return create_update_fn_with_dropout(
				mlp_fn, self.step_size, self.cell_dropout_rate
			)
		else:
			# No dropout for inference
			def update_fn(state: Array, perception: Array, key: Array | None) -> Array:
				update = mlp_fn(perception)
				return state + self.step_size * update

			return update_fn

	@partial(nnx.jit, static_argnames=("num_steps", "training", "use_pallas"))
	def __call__(
		self,
		state: State,
		input: Input | None = None,
		*,
		num_steps: int = 32,
		training: bool = True,
		use_pallas: bool = False,
		sow: bool = False,
	) -> State:
		"""Execute NCA with fused tiled execution.

		Args:
			state: Current NCA state (..., seq_len, compressed_dim, state_depth, channel_size).
			input: Optional input (not used in standard distillation).
			num_steps: Total number of NCA steps to execute.
			training: Whether to use training mode (with dropout and checkpointing).
			use_pallas: Whether to use Pallas kernel for inference (GPU only).
			sow: Whether to sow intermediate values (not supported with fusion).

		Returns:
			Final state after num_steps.

		"""
		if sow:
			# Fall back to base NCA for intermediate collection
			return self.base_nca(state, input, num_steps=num_steps, sow=True)

		if training:
			return self._forward_training(state, num_steps)
		else:
			return self._forward_inference(state, num_steps, use_pallas)

	def _forward_training(self, state: State, num_steps: int) -> State:
		"""Training forward pass with fused tiled execution.

		Uses pure JAX implementation with gradient checkpointing for
		memory-efficient training.

		"""
		perceive_fn = self._create_perceive_fn()
		update_fn = self._create_update_fn(training=True)

		# Get dropout key
		key = self._rngs.dropout() if hasattr(self._rngs, "dropout") else None

		num_passes = num_steps // self.tile_config.num_fused_steps
		remainder = num_steps % self.tile_config.num_fused_steps

		# Split keys for each pass
		if key is not None:
			pass_keys = jax.random.split(key, num_passes + (1 if remainder > 0 else 0))
		else:
			pass_keys = [None] * (num_passes + 1)

		# Run main tile passes
		for i in range(num_passes):
			state = tiled_fused_forward(
				state,
				perceive_fn,
				update_fn,
				self.tile_config,
				dropout_key=pass_keys[i],
				use_checkpointing=self.use_gradient_checkpointing,
				pad_mode=self.pad_mode,
			)

		# Handle remainder steps
		if remainder > 0:
			remainder_config = TileConfig(
				interior_size=self.tile_config.interior_size,
				num_fused_steps=remainder,
				halo_size=remainder,
			)
			state = tiled_fused_forward(
				state,
				perceive_fn,
				update_fn,
				remainder_config,
				dropout_key=pass_keys[-1] if key is not None else None,
				use_checkpointing=self.use_gradient_checkpointing,
				pad_mode=self.pad_mode,
			)

		return state

	def _forward_inference(
		self, state: State, num_steps: int, use_pallas: bool = False
	) -> State:
		"""Inference forward pass with optimized execution.

		Uses functional implementation without dropout, optionally with
		Pallas kernel for maximum performance.

		"""
		# Extract/cache weights for inference
		if self._inference_weights is None:
			self._inference_weights = extract_weights_from_nca(self.base_nca)
			self._pallas_config = create_pallas_config_from_nca(
				self.base_nca,
				tile_interior_size=self.tile_config.interior_size,
				num_fused_steps=self.tile_config.num_fused_steps,
			)

		return fused_nca_inference(
			state,
			self._inference_weights,
			self._pallas_config,
			num_steps,
			use_pallas=use_pallas,
		)

	def init_state(self, activations: Array) -> State:
		"""Initialize NCA state from transformer input activations.

		Delegates to base NCA.

		Args:
			activations: Input activations with shape (..., seq_len, hidden_dim).

		Returns:
			NCA state with shape (..., seq_len, compressed_dim, state_depth, channel_size).

		"""
		return self.base_nca.init_state(activations)

	def extract_output(self, state: State) -> Array:
		"""Extract transformer output activations from NCA state.

		Delegates to base NCA.

		Args:
			state: NCA state.

		Returns:
			Output activations with shape (..., seq_len, hidden_dim).

		"""
		return self.base_nca.extract_output(state)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render NCA state to RGB visualization.

		Delegates to base NCA.

		Args:
			state: NCA state.

		Returns:
			RGB image with shape (..., seq_len, compressed_dim, 3).

		"""
		return self.base_nca.render(state)

	def invalidate_inference_cache(self):
		"""Invalidate cached inference weights.

		Call this after updating model weights (e.g., during training)
		if you plan to switch between training and inference modes.

		"""
		self._inference_weights = None
		self._pallas_config = None

	def get_tile_info(self) -> dict:
		"""Get information about the tiling configuration.

		Returns:
			Dictionary with tile configuration details.

		"""
		from .tiling import compute_tile_grid, estimate_tile_memory

		# Example state shape
		example_shape = (64, self.compressed_dim, self.state_depth, self.channel_size)
		tile_grid = compute_tile_grid(example_shape, self.tile_config)
		mem_info = estimate_tile_memory(self.tile_config, self.channel_size)

		return {
			"interior_size": self.tile_config.interior_size,
			"tile_with_halo_size": self.tile_config.tile_with_halo_size,
			"num_fused_steps": self.tile_config.num_fused_steps,
			"halo_size": self.tile_config.halo_size,
			"tile_grid_64x64x16": tile_grid,
			"num_tiles_64x64x16": tile_grid[0] * tile_grid[1] * tile_grid[2],
			"tile_memory_fp32": mem_info,
		}


def count_params(nca: FusedGPT2BlockNCA) -> int:
	"""Count the number of parameters in the fused NCA.

	Args:
		nca: FusedGPT2BlockNCA instance.

	Returns:
		Total number of parameters.

	"""
	from .nca import count_params as base_count_params

	return base_count_params(nca.base_nca)
