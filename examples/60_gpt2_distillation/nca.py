"""GPT-2 Block Neural Cellular Automata for transformer distillation."""

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem, Input, State
from cax.utils import clip_and_uint8

from .perceive import TransformerDistillPerceive
from .update import TransformerDistillUpdate


class GPT2BlockNCA(ComplexSystem):
	"""Neural Cellular Automata for distilling a GPT-2 transformer block.

	The NCA operates on a 2D grid where:
	- Axis 0 (rows): Sequence positions
	- Axis 1 (cols): Compressed hidden dimension (projected from full hidden_dim)

	To make the computation tractable, we project the 768-dim hidden state
	down to a smaller `compressed_dim` (e.g., 64), run the NCA on this
	compressed space, then project back up.

	The goal is to learn NCA rules that, when run for N steps, transform
	input activations into output activations matching those of a GPT-2
	transformer block.

	"""

	def __init__(
		self,
		hidden_dim: int = 768,
		compressed_dim: int = 64,
		channel_size: int = 16,
		perception_size: int = 64,
		hidden_layer_sizes: tuple[int, ...] = (128,),
		*,
		local_kernel_size: tuple[int, int] = (3, 3),
		row_kernel_size: int = 5,
		col_kernel_size: int = 5,
		step_size: float = 0.1,
		cell_dropout_rate: float = 0.1,
		rngs: nnx.Rngs,
	):
		"""Initialize the GPT-2 Block NCA.

		Args:
			hidden_dim: Original hidden dimension of transformer (768 for GPT-2).
			compressed_dim: Compressed dimension for NCA grid (e.g., 64).
			channel_size: Number of channels in the NCA state.
			perception_size: Size of the perception vector.
			hidden_layer_sizes: Sizes of hidden layers in the update MLP.
			local_kernel_size: Size of the local convolution kernel.
			row_kernel_size: Size of the row-wise (feature mixing) kernel.
			col_kernel_size: Size of the column-wise (sequence mixing) kernel.
			step_size: Step size for residual updates.
			cell_dropout_rate: Dropout rate for cell updates.
			rngs: RNG key for initialization.

		"""
		self.hidden_dim = hidden_dim
		self.compressed_dim = compressed_dim
		self.channel_size = channel_size

		# Projection layers: 768 -> compressed_dim and back
		self.project_in = nnx.Linear(hidden_dim, compressed_dim, rngs=rngs)
		self.project_out = nnx.Linear(compressed_dim, hidden_dim, rngs=rngs)

		self.perceive = TransformerDistillPerceive(
			channel_size=channel_size,
			perception_size=perception_size,
			local_kernel_size=local_kernel_size,
			row_kernel_size=row_kernel_size,
			col_kernel_size=col_kernel_size,
			rngs=rngs,
		)

		self.update = TransformerDistillUpdate(
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			step_size=step_size,
			cell_dropout_rate=cell_dropout_rate,
			rngs=rngs,
		)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Perform a single NCA step.

		Args:
			state: Current NCA state (..., seq_len, hidden_dim, channel_size).
			input: Optional input (not used in standard distillation).
			sow: Whether to sow intermediate values for training.

		Returns:
			Next state with same shape.

		"""
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	def init_state(self, activations: Array) -> State:
		"""Initialize NCA state from transformer input activations.

		Projects the input from hidden_dim to compressed_dim, then
		places it in channel 0 of the NCA state.

		Args:
			activations: Input activations with shape (..., seq_len, hidden_dim).

		Returns:
			NCA state with shape (..., seq_len, compressed_dim, channel_size).

		"""
		# Project to compressed space: (..., seq_len, hidden_dim) -> (..., seq_len, compressed_dim)
		compressed = self.project_in(activations)

		# Create state with all channels
		state_shape = compressed.shape + (self.channel_size,)
		state = jnp.zeros(state_shape, dtype=compressed.dtype)

		# Place compressed activations in channel 0
		state = state.at[..., 0].set(compressed)

		return state

	def extract_output(self, state: State) -> Array:
		"""Extract transformer output activations from NCA state.

		Extracts channel 0 and projects back to full hidden_dim.

		Args:
			state: NCA state with shape (..., seq_len, compressed_dim, channel_size).

		Returns:
			Output activations with shape (..., seq_len, hidden_dim).

		"""
		# Extract compressed output from channel 0
		compressed_output = state[..., 0]

		# Project back to full hidden dimension
		return self.project_out(compressed_output)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render NCA state to RGB visualization.

		Creates a heatmap visualization of the activation channel.

		Args:
			state: NCA state (..., seq_len, compressed_dim, channel_size).

		Returns:
			RGB image with shape (..., seq_len, compressed_dim, 3).

		"""
		# Extract activation channel
		activation = state[..., 0]

		# Normalize to [0, 1]
		min_val = jnp.min(activation)
		max_val = jnp.max(activation)
		normalized = (activation - min_val) / (max_val - min_val + 1e-8)

		# Create simple grayscale visualization
		rgb = jnp.stack([normalized, normalized, normalized], axis=-1)

		return clip_and_uint8(rgb)


def count_params(nca: GPT2BlockNCA) -> int:
	"""Count the number of parameters in the NCA.

	Args:
		nca: GPT2BlockNCA instance.

	Returns:
		Total number of parameters.

	"""
	import jax

	params = nnx.state(nca, nnx.Param)
	return sum(x.size for x in jax.tree.leaves(params))
