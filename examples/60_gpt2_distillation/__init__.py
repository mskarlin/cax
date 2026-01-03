"""GPT-2 Transformer Block Distillation into Neural Cellular Automata.

This example demonstrates distilling a GPT-2 transformer block into a Neural
Cellular Automata (NCA), trading parameters for compute.
"""

from .data import (
	GPT2ActivationCollector,
	create_activation_dataset,
	get_or_create_activations,
	load_activations,
	save_activations,
)
from .nca import GPT2BlockNCA, count_params
from .perceive import TransformerDistillPerceive
from .update import TransformerDistillUpdate

__all__ = [
	"GPT2ActivationCollector",
	"create_activation_dataset",
	"get_or_create_activations",
	"load_activations",
	"save_activations",
	"GPT2BlockNCA",
	"count_params",
	"TransformerDistillPerceive",
	"TransformerDistillUpdate",
]
