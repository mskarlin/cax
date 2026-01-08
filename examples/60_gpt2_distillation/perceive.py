"""Custom perceive module for transformer distillation."""

import jax.numpy as jnp
from flax import nnx

from cax.core import State
from cax.core.perceive import Perceive, Perception


class TransformerDistillPerceive(Perceive):
	"""3D perceive module for transformer distillation.

	Uses 3D convolutions on a (seq_len, compressed_dim, state_depth) grid
	to enable local information flow in all three dimensions.

	The 3D structure allows:
	- Sequence mixing (dim 0): Information flow across token positions
	- Feature mixing (dim 1): Information flow across hidden dimensions
	- Depth mixing (dim 2): Information flow through state depth (like NCA steps)

	"""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		*,
		kernel_size: tuple[int, int, int] = (3, 3, 3),
		rngs: nnx.Rngs,
	):
		"""Initialize the 3D perceive module.

		Args:
			channel_size: Number of input channels in the NCA state.
			perception_size: Size of the output perception vector.
			kernel_size: Size of the 3D convolution kernel (seq, hidden, depth).
			rngs: RNG key for initialization.

		"""
		self.channel_size = channel_size
		self.perception_size = perception_size

		# Number of perception kernels per channel (for depthwise conv)
		# perception_size should be divisible by channel_size
		num_kernels = perception_size // channel_size

		# 3D depthwise convolution
		# Each channel gets num_kernels independent 3D filters
		self.conv = nnx.Conv(
			in_features=channel_size,
			out_features=perception_size,
			kernel_size=kernel_size,
			padding="SAME",
			feature_group_count=channel_size,  # Depthwise
			use_bias=False,
			rngs=rngs,
		)

	def __call__(self, state: State) -> Perception:
		"""Apply 3D perception to the state.

		Args:
			state: NCA state with shape (..., seq_len, compressed_dim, state_depth, channel_size).

		Returns:
			Perception with shape (..., seq_len, compressed_dim, state_depth, perception_size).

		"""
		return self.conv(state)
