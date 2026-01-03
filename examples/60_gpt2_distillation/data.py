"""GPT-2 activation extraction utilities for NCA distillation."""

import gc
import os
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


# Default cache directory
CACHE_DIR = Path("./activation_cache")


class GPT2ActivationCollector:
	"""Collects input/output activations from a specific GPT-2 transformer block.

	For fastest collection, use device="cuda" with larger batch sizes.
	For memory efficiency (to leave GPU for JAX), use device="cpu".
	"""

	def __init__(self, block_idx: int = 0, device: str | None = None, use_flash_attn: bool = False):
		"""Initialize the activation collector.

		Args:
			block_idx: Index of the transformer block to extract activations from (0-11).
			device: Device to run the model on. If None, auto-selects (CUDA if available).
			use_flash_attn: Whether to use flash attention (requires flash-attn package).

		"""
		if device is None:
			device = "cuda" if torch.cuda.is_available() else "cpu"

		self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
		self.tokenizer.pad_token = self.tokenizer.eos_token

		# Use flash attention if available and requested
		attn_impl = "flash_attention_2" if use_flash_attn else None

		self.model = GPT2Model.from_pretrained(
			"gpt2",
			output_hidden_states=True,
			torch_dtype=torch.float16 if device == "cuda" else torch.float32,
			attn_implementation=attn_impl,
		)
		self.model.eval()
		self.model.to(device)
		self.device = device
		self.block_idx = block_idx

		# Freeze all parameters
		for param in self.model.parameters():
			param.requires_grad = False

		print(f"GPT-2 loaded on {device} (block {block_idx})")

	def collect_activations(
		self, texts: list[str], max_length: int = 128
	) -> dict[str, np.ndarray]:
		"""Collect input/output activations for a transformer block.

		Args:
			texts: List of text strings to process.
			max_length: Maximum sequence length (will pad/truncate).

		Returns:
			Dictionary with keys:
				- 'input': Activations before block (batch, seq, hidden) as float32 numpy
				- 'output': Activations after block (batch, seq, hidden) as float32 numpy
				- 'attention_mask': Attention mask (batch, seq)

		"""
		inputs = self.tokenizer(
			texts,
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length=max_length,
		)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = self.model(**inputs)

		hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

		# hidden_states[0] is embedding output
		# hidden_states[i] is output of block i-1
		# So input to block_idx is hidden_states[block_idx]
		# Output of block_idx is hidden_states[block_idx + 1]

		# Convert to float32 numpy arrays (not JAX yet - saves memory during collection)
		result = {
			"input": hidden_states[self.block_idx].float().cpu().numpy(),
			"output": hidden_states[self.block_idx + 1].float().cpu().numpy(),
			"attention_mask": inputs["attention_mask"].cpu().numpy(),
		}

		# Clean up torch tensors
		del outputs, hidden_states
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

		return result

	def cleanup(self):
		"""Release model memory after activation collection is complete."""
		del self.model
		del self.tokenizer
		gc.collect()
		torch.cuda.empty_cache() if torch.cuda.is_available() else None


def create_activation_dataset(
	collector: GPT2ActivationCollector,
	texts: list[str],
	max_length: int = 128,
	batch_size: int = 8,
	cleanup_after: bool = True,
	show_progress: bool = True,
) -> dict[str, jnp.ndarray]:
	"""Create dataset of activation pairs from a list of texts.

	Args:
		collector: GPT2ActivationCollector instance.
		texts: List of text strings to process.
		max_length: Maximum sequence length.
		batch_size: Batch size for processing (smaller = less memory).
		cleanup_after: Whether to cleanup the collector after processing.
		show_progress: Whether to show a progress bar.

	Returns:
		Dictionary with 'input', 'output', and 'mask' arrays as JAX arrays.

	"""
	inputs_list = []
	outputs_list = []
	masks_list = []

	num_batches = (len(texts) + batch_size - 1) // batch_size

	if show_progress:
		try:
			from tqdm.auto import tqdm
			iterator = tqdm(range(0, len(texts), batch_size), desc="Collecting activations", total=num_batches)
		except ImportError:
			iterator = range(0, len(texts), batch_size)
	else:
		iterator = range(0, len(texts), batch_size)

	for i in iterator:
		batch_texts = texts[i : i + batch_size]
		activations = collector.collect_activations(batch_texts, max_length)
		inputs_list.append(activations["input"])
		outputs_list.append(activations["output"])
		masks_list.append(activations["attention_mask"])

	# Cleanup PyTorch model before creating JAX arrays
	if cleanup_after:
		collector.cleanup()

	# Concatenate numpy arrays first, then convert to JAX
	inputs_np = np.concatenate(inputs_list, axis=0)
	outputs_np = np.concatenate(outputs_list, axis=0)
	masks_np = np.concatenate(masks_list, axis=0)

	# Free the lists
	del inputs_list, outputs_list, masks_list
	gc.collect()

	return {
		"input": jnp.array(inputs_np),
		"output": jnp.array(outputs_np),
		"mask": jnp.array(masks_np),
	}


def iter_openwebtext(num_samples: int = 10000) -> Iterator[str]:
	"""Iterate over OpenWebText samples.

	Args:
		num_samples: Number of samples to yield.

	Yields:
		Text strings from the dataset.

	"""
	try:
		from datasets import load_dataset

		dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

		for i, example in enumerate(dataset):
			if i >= num_samples:
				break
			yield example["text"]
	except Exception as e:
		# Fallback: generate synthetic text if dataset unavailable
		print(f"Warning: Could not load OpenWebText ({e}). Using synthetic data.")
		import random

		words = [
			"the",
			"quick",
			"brown",
			"fox",
			"jumps",
			"over",
			"lazy",
			"dog",
			"and",
			"runs",
			"through",
			"forest",
			"while",
			"birds",
			"sing",
			"beautiful",
			"songs",
			"in",
			"morning",
			"light",
		]
		for _ in range(num_samples):
			length = random.randint(20, 100)
			yield " ".join(random.choices(words, k=length))


def create_dataset_from_openwebtext(
	collector: GPT2ActivationCollector,
	num_samples: int = 10000,
	max_length: int = 128,
	batch_size: int = 32,
) -> dict[str, jnp.ndarray]:
	"""Create activation dataset from OpenWebText.

	Args:
		collector: GPT2ActivationCollector instance.
		num_samples: Number of text samples to process.
		max_length: Maximum sequence length.
		batch_size: Batch size for processing.

	Returns:
		Dictionary with 'input', 'output', and 'mask' arrays.

	"""
	texts = list(iter_openwebtext(num_samples))
	return create_activation_dataset(collector, texts, max_length, batch_size)


def save_activations(dataset: dict[str, jnp.ndarray], path: str | Path) -> None:
	"""Save activations to disk as .npz file.

	Args:
		dataset: Dictionary with 'input', 'output', and 'mask' arrays.
		path: Path to save the .npz file.

	"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)

	np.savez_compressed(
		path,
		input=np.array(dataset["input"]),
		output=np.array(dataset["output"]),
		mask=np.array(dataset["mask"]),
	)
	print(f"Saved activations to {path}")


def load_activations(path: str | Path) -> dict[str, jnp.ndarray]:
	"""Load activations from disk.

	Args:
		path: Path to the .npz file.

	Returns:
		Dictionary with 'input', 'output', and 'mask' as JAX arrays.

	"""
	path = Path(path)
	data = np.load(path)
	result = {
		"input": jnp.array(data["input"]),
		"output": jnp.array(data["output"]),
		"mask": jnp.array(data["mask"]),
	}
	print(f"Loaded activations from {path}: {result['input'].shape[0]} samples")
	return result


def get_or_create_activations(
	block_idx: int = 0,
	num_samples: int = 1024,
	max_length: int = 64,
	batch_size: int = 64,
	cache_dir: str | Path = CACHE_DIR,
	device: str | None = None,
	force_recreate: bool = False,
) -> dict[str, jnp.ndarray]:
	"""Get activations from cache or create them.

	This is the recommended high-level function. It:
	1. Checks if cached activations exist
	2. If not, creates them using GPU (fast) and saves to disk
	3. Cleans up PyTorch before returning JAX arrays

	Args:
		block_idx: Which transformer block to extract (0-11).
		num_samples: Number of text samples.
		max_length: Maximum sequence length.
		batch_size: Batch size for GPU processing (64 is good for 24GB GPU).
		cache_dir: Directory to store cached activations.
		device: Device for GPT-2 ('cuda' recommended for speed, 'cpu' for memory).
		force_recreate: If True, recreate even if cache exists.

	Returns:
		Dictionary with 'input', 'output', and 'mask' as JAX arrays.

	"""
	cache_dir = Path(cache_dir)
	cache_file = cache_dir / f"gpt2_block{block_idx}_n{num_samples}_len{max_length}.npz"

	# Try to load from cache
	if cache_file.exists() and not force_recreate:
		print(f"Loading cached activations from {cache_file}")
		return load_activations(cache_file)

	# Create activations
	print(f"Creating activations (this only needs to be done once)...")

	# Use GPU by default for speed
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	# Adjust batch size based on device
	if device == "cuda" and batch_size < 32:
		batch_size = 64  # Use larger batches on GPU for speed
		print(f"Using batch_size={batch_size} on GPU for faster processing")

	collector = GPT2ActivationCollector(block_idx=block_idx, device=device)

	# Generate texts
	print(f"Generating {num_samples} text samples...")
	texts = list(iter_openwebtext(num_samples))

	# Collect activations
	dataset = create_activation_dataset(
		collector,
		texts,
		max_length=max_length,
		batch_size=batch_size,
		cleanup_after=True,
		show_progress=True,
	)

	# Save to cache
	save_activations(dataset, cache_file)

	return dataset
