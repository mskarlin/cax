"""Custom perceive module for transformer distillation."""

import jax.numpy as jnp
from flax import nnx

from cax.core import State
from cax.core.perceive import Perceive, Perception


class TransformerDistillPerceive(Perceive):
	"""Multi-scale perceive module for transformer distillation.

	Uses multiple convolution scales to enable both local and extended-range
	information flow across the 2D (sequence x hidden_dim) grid.

	Combines:
	- Local 3x3 convolution for neighborhood perception
	- Row-wise 1xK convolution for feature mixing (across hidden_dim)
	- Column-wise Kx1 convolution for sequence mixing (across positions)

	"""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		*,
		local_kernel_size: tuple[int, int] = (3, 3),
		row_kernel_size: int = 7,
		col_kernel_size: int = 7,
		rngs: nnx.Rngs,
	):
		"""Initialize the perceive module.

		Args:
			channel_size: Number of input channels in the NCA state.
			perception_size: Size of the output perception vector.
			local_kernel_size: Size of the local convolution kernel.
			row_kernel_size: Size of the row-wise (feature mixing) kernel.
			col_kernel_size: Size of the column-wise (sequence mixing) kernel.
			rngs: RNG key for initialization.

		"""
		self.channel_size = channel_size
		self.perception_size = perception_size

		# Local 3x3 convolution (depthwise for efficiency)
		# Captures immediate neighborhood information
		local_out = channel_size * 3  # identity, grad_x, grad_y equivalent
		self.local_conv = nnx.Conv(
			in_features=channel_size,
			out_features=local_out,
			kernel_size=local_kernel_size,
			padding="SAME",
			feature_group_count=channel_size,  # Depthwise
			use_bias=False,
			rngs=rngs,
		)

		# Row-wise 1xK convolution (feature mixing across hidden_dim)
		# Enables information flow across the hidden dimension
		row_out = channel_size * 2
		self.row_conv = nnx.Conv(
			in_features=channel_size,
			out_features=row_out,
			kernel_size=(1, row_kernel_size),
			padding="SAME",
			use_bias=False,
			rngs=rngs,
		)

		# Column-wise Kx1 convolution (sequence mixing across positions)
		# Enables information flow across the sequence dimension
		col_out = channel_size * 2
		self.col_conv = nnx.Conv(
			in_features=channel_size,
			out_features=col_out,
			kernel_size=(col_kernel_size, 1),
			padding="SAME",
			use_bias=False,
			rngs=rngs,
		)

		# Combine all perceptions into final perception size
		total_perception = local_out + row_out + col_out
		self.combine = nnx.Conv(
			in_features=total_perception,
			out_features=perception_size,
			kernel_size=(1, 1),
			padding="SAME",
			use_bias=False,
			rngs=rngs,
		)

	def __call__(self, state: State) -> Perception:
		"""Apply multi-scale perception to the state.

		Args:
			state: NCA state with shape (..., seq_len, hidden_dim, channel_size).

		Returns:
			Perception with shape (..., seq_len, hidden_dim, perception_size).

		"""
		# Apply each convolution scale
		local_perception = self.local_conv(state)
		row_perception = self.row_conv(state)
		col_perception = self.col_conv(state)

		# Concatenate along channel dimension
		combined = jnp.concatenate(
			[local_perception, row_perception, col_perception], axis=-1
		)

		# Combine into final perception
		perception = self.combine(combined)

		return perception
