"""Custom update module for transformer distillation."""

from collections.abc import Callable

from flax import nnx

from cax.core import Input, State
from cax.core.perceive import Perception
from cax.core.update import ResidualUpdate


class TransformerDistillUpdate(ResidualUpdate):
	"""3D update module for transformer distillation.

	Extends ResidualUpdate with GELU activation (matching GPT-2) and
	smaller step size for stable training. Operates on a 3D spatial grid
	(seq_len, compressed_dim, state_depth).

	Does not use alive masking since all positions in transformer
	activations are active.

	"""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		hidden_layer_sizes: tuple[int, ...] = (256,),
		*,
		activation_fn: Callable = nnx.gelu,
		step_size: float = 0.1,
		cell_dropout_rate: float = 0.1,
		zeros_init: bool = True,
		rngs: nnx.Rngs,
	):
		"""Initialize the 3D update module.

		Args:
			channel_size: Number of channels in the NCA state.
			perception_size: Size of the perception input.
			hidden_layer_sizes: Sizes of hidden layers in the MLP.
			activation_fn: Activation function (GELU to match GPT-2).
			step_size: Step size for residual update (smaller for stability).
			cell_dropout_rate: Dropout rate for cell updates.
			zeros_init: Whether to use zeros initialization for last layer.
			rngs: RNG key for initialization.

		"""
		super().__init__(
			num_spatial_dims=3,  # 3D grid (seq_len x compressed_dim x state_depth)
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layer_sizes,
			activation_fn=activation_fn,
			step_size=step_size,
			cell_dropout_rate=cell_dropout_rate,
			zeros_init=zeros_init,
			rngs=rngs,
		)

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Process the current state and perception to produce a new state.

		Uses residual update: state += step_size * dropout(mlp(perception))

		Args:
			state: Current NCA state (..., seq_len, compressed_dim, state_depth, channel_size).
			perception: Current perception (..., seq_len, compressed_dim, state_depth, perception_size).
			input: Optional input (not used in standard distillation).

		Returns:
			Next state with same shape as input state.

		"""
		return super().__call__(state, perception, input)
