#!/usr/bin/env python3
"""
Convert Moshi MLX/PyTorch models to GGUF format for llama.cpp.

This converter handles:
- Main Temporal Transformer (LM)
- DepFormer (depth transformer for codebook dependencies)
- Conditioners (optional voice conditioning)

Usage:
    python convert_moshi_to_gguf.py \
        --model-path /path/to/moshi_mlx \
        --output moshi-1.6b-f16.gguf \
        --outtype f16

Requirements:
    pip install gguf mlx numpy safetensors
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available, will use safetensors directly")

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    safe_open = None
    safe_load_torch = None

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False
    spm = None

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF data types
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30


# GGUF metadata value types
class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: list[tuple[str, Any, int]] = []
        self.tensors: list[tuple[str, np.ndarray, int]] = []

    def add_string(self, key: str, value: str):
        self.metadata.append((key, value, GGUFValueType.STRING))

    def add_uint32(self, key: str, value: int):
        self.metadata.append((key, value, GGUFValueType.UINT32))

    def add_int32(self, key: str, value: int):
        self.metadata.append((key, value, GGUFValueType.INT32))

    def add_float32(self, key: str, value: float):
        self.metadata.append((key, value, GGUFValueType.FLOAT32))

    def add_bool(self, key: str, value: bool):
        self.metadata.append((key, value, GGUFValueType.BOOL))

    def add_array(self, key: str, value: list, elem_type: int):
        self.metadata.append((key, (value, elem_type), GGUFValueType.ARRAY))

    def add_tensor(self, name: str, data: np.ndarray, dtype: int = GGMLType.F16):
        self.tensors.append((name, data, dtype))

    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, value: Any, vtype: int):
        f.write(struct.pack('<I', vtype))

        if vtype == GGUFValueType.STRING:
            self._write_string(f, value)
        elif vtype == GGUFValueType.UINT32:
            f.write(struct.pack('<I', value))
        elif vtype == GGUFValueType.INT32:
            f.write(struct.pack('<i', value))
        elif vtype == GGUFValueType.FLOAT32:
            f.write(struct.pack('<f', value))
        elif vtype == GGUFValueType.BOOL:
            f.write(struct.pack('<B', 1 if value else 0))
        elif vtype == GGUFValueType.ARRAY:
            arr, elem_type = value
            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(arr)))
            for elem in arr:
                if elem_type == GGUFValueType.INT32:
                    f.write(struct.pack('<i', elem))
                elif elem_type == GGUFValueType.FLOAT32:
                    f.write(struct.pack('<f', elem))
                elif elem_type == GGUFValueType.STRING:
                    self._write_string(f, elem)
        elif vtype == GGUFValueType.UINT64:
            f.write(struct.pack('<Q', value))
        elif vtype == GGUFValueType.INT64:
            f.write(struct.pack('<q', value))
        elif vtype == GGUFValueType.FLOAT64:
            f.write(struct.pack('<d', value))

    def write(self):
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Metadata
            for key, value, vtype in self.metadata:
                self._write_string(f, key)
                self._write_metadata_value(f, value, vtype)

            # Tensor info
            tensor_data_offset = 0
            tensor_infos = []

            for name, data, dtype in self.tensors:
                # Convert to target dtype
                if dtype == GGMLType.F16:
                    tensor_data = data.astype(np.float16)
                elif dtype == GGMLType.F32:
                    tensor_data = data.astype(np.float32)
                elif dtype == GGMLType.BF16:
                    # BF16 stored as uint16
                    tensor_data = data.astype(np.float32).view(np.uint32)
                    tensor_data = ((tensor_data >> 16) & 0xFFFF).astype(np.uint16)
                else:
                    tensor_data = data.astype(np.float32)

                self._write_string(f, name)
                f.write(struct.pack('<I', len(data.shape)))  # n_dims
                for dim in data.shape:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', dtype))  # type
                f.write(struct.pack('<Q', tensor_data_offset))  # offset

                tensor_infos.append((tensor_data, tensor_data_offset))
                tensor_data_offset += tensor_data.nbytes
                # Align to 32 bytes
                tensor_data_offset = (tensor_data_offset + 31) & ~31

            # Alignment padding before tensor data
            current_pos = f.tell()
            alignment = 32
            padding_needed = (alignment - (current_pos % alignment)) % alignment
            f.write(b'\x00' * padding_needed)

            # Tensor data
            for tensor_data, _ in tensor_infos:
                f.write(tensor_data.tobytes())
                # Alignment padding
                current_pos = f.tell()
                padding_needed = (alignment - (current_pos % alignment)) % alignment
                f.write(b'\x00' * padding_needed)


def load_config(model_path: Path) -> dict:
    """Load model configuration from JSON or infer from weights."""
    config_paths = [
        model_path / "config.json",
        model_path / "moshi_config.json",
        model_path.parent / "config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    # Try to infer from model structure
    print("Warning: No config.json found, will infer from weights")
    return {}


def infer_config_from_weights(weights: dict[str, np.ndarray]) -> dict:
    """Infer model configuration from weight shapes."""
    config = {}

    # Count transformer layers
    layer_indices = set()
    for name in weights.keys():
        if name.startswith("transformer.layers."):
            parts = name.split(".")
            layer_indices.add(int(parts[2]))
    if layer_indices:
        config["num_layers"] = max(layer_indices) + 1

    # Get d_model from text_emb or transformer weights
    if "text_emb.weight" in weights:
        config["text_card"], config["dim"] = weights["text_emb.weight"].shape
        config["text_card"] -= 1  # Vocab includes padding
    elif "transformer.layers.0.self_attn.in_proj_weight" in weights:
        config["dim"] = weights["transformer.layers.0.self_attn.in_proj_weight"].shape[1]

    # Infer num_heads from in_proj shape (3 * dim for QKV)
    if "transformer.layers.0.self_attn.in_proj_weight" in weights:
        in_proj = weights["transformer.layers.0.self_attn.in_proj_weight"]
        # Standard head_dim is 64 or 128
        dim = in_proj.shape[1]
        for head_dim in [64, 128, 256]:
            if dim % head_dim == 0:
                config["num_heads"] = dim // head_dim
                break

    # Count audio embeddings
    audio_emb_indices = set()
    for name in weights.keys():
        if name.startswith("emb."):
            parts = name.split(".")
            audio_emb_indices.add(int(parts[1]))
    if audio_emb_indices:
        config["n_q"] = len(audio_emb_indices)

    # Get audio vocab size from embeddings
    if "emb.0.weight" in weights:
        config["card"] = weights["emb.0.weight"].shape[0] - 1

    # Count DepFormer layers
    dep_layer_indices = set()
    for name in weights.keys():
        if name.startswith("depformer.layers."):
            parts = name.split(".")
            dep_layer_indices.add(int(parts[2]))
    if dep_layer_indices:
        config["depformer_num_layers"] = max(dep_layer_indices) + 1

    # Get DepFormer dim
    if "depformer.layers.0.self_attn.in_proj_weight" in weights:
        config["depformer_dim"] = weights["depformer.layers.0.self_attn.in_proj_weight"].shape[1]

    # Count DepFormer gating blocks per layer (dep_q)
    gating_indices = set()
    for name in weights.keys():
        if "depformer.layers.0.gating." in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "gating" and i + 1 < len(parts):
                    try:
                        gating_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
    if gating_indices:
        config["dep_q"] = len(gating_indices)

    # Get depformer num_heads
    if "depformer.layers.0.self_attn.in_proj_weight" in weights:
        dep_in_proj = weights["depformer.layers.0.self_attn.in_proj_weight"]
        dep_dim = dep_in_proj.shape[1]
        for head_dim in [64, 128, 256]:
            if dep_dim % head_dim == 0:
                config["depformer_num_heads"] = dep_dim // head_dim
                break

    return config


def load_tokenizer(tokenizer_path: Path) -> tuple[list[str], list[float], list[int]]:
    """Load SentencePiece tokenizer and return vocab, scores, and token types."""
    if not HAS_SPM:
        raise ImportError("sentencepiece is required for tokenizer loading")

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    vocab = []
    scores = []
    token_types = []

    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        vocab.append(piece)
        scores.append(score)

        # Token types: 0=normal, 1=unknown, 2=control, 3=user_defined, 4=byte
        if sp.IsUnknown(i):
            token_types.append(2)  # UNK
        elif sp.IsControl(i):
            token_types.append(3)  # CONTROL
        elif sp.IsByte(i):
            token_types.append(6)  # BYTE
        elif sp.IsUnused(i):
            token_types.append(5)  # UNUSED
        else:
            token_types.append(1)  # NORMAL

    return vocab, scores, token_types


def load_weights(model_path: Path) -> dict[str, np.ndarray]:
    """Load weights from safetensors or MLX format."""
    weights = {}

    # Try safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    # Filter out tokenizer files (Mimi codec)
    model_files = [f for f in safetensor_files if "tokenizer" not in f.name.lower()]

    if model_files and HAS_SAFETENSORS:
        # Try using torch for bfloat16 support
        try:
            import torch
            for sf_path in model_files:
                print(f"Loading {sf_path.name} (using torch for bf16 support)...")
                state_dict = safe_load_torch(str(sf_path))
                for key, tensor in state_dict.items():
                    # Convert to float32 numpy
                    weights[key] = tensor.float().numpy()
            return weights
        except ImportError:
            print("Warning: torch not available, trying manual bf16 conversion")

        # Manual loading without torch (won't work for bf16)
        for sf_path in model_files:
            print(f"Loading {sf_path.name}...")
            with safe_open(sf_path, framework="numpy") as f:
                for key in f.keys():
                    try:
                        weights[key] = f.get_tensor(key)
                    except TypeError as e:
                        if "bfloat16" in str(e):
                            print(f"  Skipping bf16 tensor (need torch): {key}")
                            continue
                        raise
        return weights

    # Try MLX format
    if HAS_MLX:
        mlx_files = list(model_path.glob("*.npz")) + list(model_path.glob("weights/*.npz"))
        for mlx_path in mlx_files:
            print(f"Loading {mlx_path.name}...")
            data = mx.load(str(mlx_path))
            for key, value in data.items():
                weights[key] = np.array(value)

    return weights


def reconstruct_low_rank_embeddings(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Reconstruct full embeddings from low-rank factorization.

    Reconstructs in PyTorch format [vocab, embd] which will later be transposed to GGML format.

    For text embeddings (text_emb, depformer_text_emb):
        Intermediate: out2 @ weight.T + out1 @ (low_rank.T @ low_rank) @ weight.T => [embd, vocab]
        Transposed: => [vocab, embd]

    For audio embeddings (depformer_emb.{i}):
        Intermediate: low_rank @ weight.T => [embd, vocab]
        Transposed: => [vocab, embd]
    """
    reconstructed = {}
    processed_keys = set()

    # Process depformer text embedding
    if 'depformer_text_emb.weight' in weights:
        weight = weights['depformer_text_emb.weight']
        low_rank = weights.get('depformer_text_emb.low_rank.weight')
        out1 = weights.get('depformer_text_emb.out1.weight')
        out2 = weights.get('depformer_text_emb.out2.weight')

        if low_rank is not None and out1 is not None and out2 is not None:
            print("Reconstructing depformer_text_emb from low-rank factorization...")
            # part1: out2 @ weight.T => [embd, vocab]
            part1 = np.matmul(out2, weight.T)
            # part2: out1 @ (low_rank.T @ low_rank) @ weight.T => [embd, vocab]
            lr_prod = np.matmul(low_rank.T, low_rank)
            part2 = np.matmul(np.matmul(out1, lr_prod), weight.T)
            # Transpose to PyTorch format [vocab, embd]
            reconstructed['depformer_text_emb.weight'] = (part1 + part2).T

            processed_keys.update(['depformer_text_emb.weight', 'depformer_text_emb.low_rank.weight',
                                   'depformer_text_emb.out1.weight', 'depformer_text_emb.out2.weight'])

    # Process depformer audio embeddings (0-30)
    for i in range(32):
        weight_key = f'depformer_emb.{i}.weight'
        low_rank_key = f'depformer_emb.{i}.low_rank.weight'

        if weight_key in weights and low_rank_key in weights:
            weight = weights[weight_key]
            low_rank = weights[low_rank_key]
            print(f"Reconstructing depformer_emb.{i} from low-rank factorization...")
            # low_rank @ weight.T => [embd, vocab], transpose to [vocab, embd]
            reconstructed[f'depformer_emb.{i}.weight'] = np.matmul(low_rank, weight.T).T
            processed_keys.update([weight_key, low_rank_key])

    # Process main text embedding if needed
    if 'text_emb.weight' in weights:
        weight = weights['text_emb.weight']
        low_rank = weights.get('text_emb.low_rank.weight')
        out1 = weights.get('text_emb.out1.weight')
        out2 = weights.get('text_emb.out2.weight')

        if low_rank is not None and out1 is not None and out2 is not None:
            print("Reconstructing text_emb from low-rank factorization...")
            # Same formula as depformer_text_emb
            part1 = np.matmul(out2, weight.T)
            lr_prod = np.matmul(low_rank.T, low_rank)
            part2 = np.matmul(np.matmul(out1, lr_prod), weight.T)
            # Transpose to PyTorch format [vocab, embd]
            reconstructed['text_emb.weight'] = (part1 + part2).T

            processed_keys.update(['text_emb.weight', 'text_emb.low_rank.weight',
                                   'text_emb.out1.weight', 'text_emb.out2.weight'])

    # Copy over all non-processed weights
    for key, value in weights.items():
        if key not in processed_keys:
            reconstructed[key] = value

    return reconstructed


def convert_pytorch_to_mlx_names(pth_weights: dict[str, np.ndarray], config: dict) -> dict[str, np.ndarray]:
    """
    Convert PyTorch weight names to MLX convention.
    Based on lm.py load_pytorch_weights() method.

    Handles multiple naming conventions:
    - Production models with gating.linear_in/linear_out and alpha norms
    - Test models with linear1/linear2 and weight/bias norms
    """
    mlx_weights = {}

    # Get depformer config
    dep_q = config.get("dep_q", 8)
    depformer_schedule = config.get("depformer_weights_per_step_schedule")

    if depformer_schedule is not None:
        depformer_chunks = max(depformer_schedule) + 1
    else:
        depformer_chunks = dep_q

    for name, tensor in pth_weights.items():
        new_name = name

        # Output norm: alpha[0,0] -> weight (production) or keep as-is (test)
        if name == "out_norm.alpha":
            tensor = tensor[0, 0] if tensor.ndim >= 2 else tensor.flatten()[0]
            new_name = "out_norm.weight"
        elif name in ["out_norm.weight", "out_norm.bias"]:
            new_name = name  # Keep as-is

        # Audio embeddings: emb.{i} -> audio_embs.{i}
        elif name.startswith("emb."):
            new_name = name.replace("emb.", "audio_embs.")

        # Transformer layers
        elif name.startswith("transformer.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            rest = ".".join(parts[3:])

            # Convert alpha to weight for norms
            if rest.endswith(".alpha"):
                tensor = tensor[0, 0] if tensor.ndim >= 2 else tensor.flatten()[0]
                rest = rest.replace(".alpha", ".weight")

            # Convert in_proj_weight to in_proj.weight
            rest = rest.replace(".in_proj_weight", ".in_proj.weight")

            # Handle linear1/linear2 (test model) vs gating (production)
            if rest == "linear1.weight":
                rest = "gating.linear_in.weight"
            elif rest == "linear2.weight":
                rest = "gating.linear_out.weight"

            new_name = f"transformer.layers.{layer_idx}.{rest}"

        # DepFormer text embedding
        elif name == "depformer_text_emb.weight":
            new_name = "depformer.slices.0.emb.weight"
        elif name.startswith("depformer_text_emb."):
            suffix = name[len("depformer_text_emb."):]
            new_name = f"depformer.slices.0.emb.{suffix}"

        # DepFormer audio embeddings
        elif name.startswith("depformer_emb."):
            parts = name.split(".")
            idx = int(parts[1])
            rest = ".".join(parts[2:])
            new_name = f"depformer.slices.{idx + 1}.emb.{rest}"

        # DepFormer input projections
        elif name.startswith("depformer_in."):
            parts = name.split(".")
            pth_idx = int(parts[1])
            new_name = f"depformer.linear_in.{pth_idx}.weight"

        # DepFormer output projections
        elif name.startswith("linears."):
            parts = name.split(".")
            slice_idx = int(parts[1])
            new_name = f"depformer.slices.{slice_idx}.linear_out.weight"

        # DepFormer transformer layers
        elif name.startswith("depformer.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            rest = ".".join(parts[3:])

            # Handle norm alpha conversion (production)
            if rest.endswith(".alpha"):
                tensor = tensor[0, 0] if tensor.ndim >= 2 else tensor.flatten()[0]
                rest = rest.replace(".alpha", ".weight")

            # Handle attention weight splitting for gating (per-slice weights)
            if "gating." in rest and rest.count(".") >= 2:
                # gating.{slice_idx}.linear_in.weight
                gating_parts = rest.split(".")
                slice_idx = int(gating_parts[1])
                linear_type = gating_parts[2]
                suffix = ".".join(gating_parts[3:]) if len(gating_parts) > 3 else "weight"
                new_name = f"depformer.transformer.layers.{layer_idx}.gating.{slice_idx}.{linear_type}.{suffix}"

            elif "self_attn.in_proj_weight" in rest:
                # Attention weights are shared, split across depformer chunks
                if tensor.shape[0] % depformer_chunks == 0:
                    chunks = np.split(tensor, depformer_chunks, axis=0)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_name = f"depformer.transformer.layers.{layer_idx}.self_attn.{chunk_idx}.in_proj.weight"
                        mlx_weights[chunk_name] = chunk
                    continue
                else:
                    # Can't split, keep as single tensor
                    new_name = f"depformer.transformer.layers.{layer_idx}.self_attn.in_proj.weight"

            elif "self_attn.out_proj.weight" in rest:
                if tensor.shape[0] % depformer_chunks == 0:
                    chunks = np.split(tensor, depformer_chunks, axis=0)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_name = f"depformer.transformer.layers.{layer_idx}.self_attn.{chunk_idx}.out_proj.weight"
                        mlx_weights[chunk_name] = chunk
                    continue
                else:
                    new_name = f"depformer.transformer.layers.{layer_idx}.self_attn.out_proj.weight"
            else:
                new_name = f"depformer.transformer.layers.{layer_idx}.{rest}"

        # Text embeddings
        elif name in ["text_emb.weight", "text_linear.weight"]:
            new_name = name

        # Keep other names as-is
        mlx_weights[new_name] = tensor

    return mlx_weights


def convert_to_gguf_names(mlx_weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Convert MLX weight names to GGUF tensor names for llama.cpp.
    Uses llama.cpp naming conventions.
    """
    gguf_weights = {}

    for name, tensor in mlx_weights.items():
        # Text embedding
        if name == "text_emb.weight":
            new_name = "token_embd.weight"
        elif name.startswith("text_emb."):
            new_name = name.replace("text_emb.", "token_embd.")

        # Audio embeddings (MLX style)
        elif name.startswith("audio_embs."):
            new_name = name.replace("audio_embs.", "audio_embd.")

        # Audio embeddings (PyTorch style - emb.{i})
        elif name.startswith("emb."):
            new_name = name.replace("emb.", "audio_embd.")

        # Output norm
        elif name in ["out_norm.weight", "out_norm.bias"]:
            new_name = name.replace("out_norm.", "output_norm.")

        # Text output linear
        elif name == "text_linear.weight":
            new_name = "output.weight"

        # Transformer layers
        elif name.startswith("transformer.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            rest = ".".join(parts[3:])

            # Map to llama.cpp conventions
            if rest == "norm1.weight":
                new_name = f"blk.{layer_idx}.attn_norm.weight"
            elif rest == "norm1.bias":
                new_name = f"blk.{layer_idx}.attn_norm.bias"
            elif rest == "norm2.weight":
                new_name = f"blk.{layer_idx}.ffn_norm.weight"
            elif rest == "norm2.bias":
                new_name = f"blk.{layer_idx}.ffn_norm.bias"
            elif rest == "self_attn.in_proj.weight":
                # Split into Q, K, V
                qkv = np.split(tensor, 3, axis=0)
                gguf_weights[f"blk.{layer_idx}.attn_q.weight"] = qkv[0]
                gguf_weights[f"blk.{layer_idx}.attn_k.weight"] = qkv[1]
                gguf_weights[f"blk.{layer_idx}.attn_v.weight"] = qkv[2]
                continue
            elif rest == "self_attn.out_proj.weight":
                new_name = f"blk.{layer_idx}.attn_output.weight"
            elif rest == "gating.linear_in.weight":
                # Gated FFN: split into gate and up
                gate_up = np.split(tensor, 2, axis=0)
                gguf_weights[f"blk.{layer_idx}.ffn_gate.weight"] = gate_up[0]
                gguf_weights[f"blk.{layer_idx}.ffn_up.weight"] = gate_up[1]
                continue
            elif rest == "gating.linear_out.weight":
                new_name = f"blk.{layer_idx}.ffn_down.weight"
            # Linear1/linear2 (test model style - non-gated FFN)
            elif rest == "linear1.weight":
                new_name = f"blk.{layer_idx}.ffn_up.weight"
            elif rest == "linear2.weight":
                new_name = f"blk.{layer_idx}.ffn_down.weight"
            # Cross-attention (if present)
            elif rest == "norm_cross.weight":
                new_name = f"blk.{layer_idx}.cross_attn_norm.weight"
            elif rest == "cross_attention.in_proj.weight":
                qkv = np.split(tensor, 3, axis=0)
                gguf_weights[f"blk.{layer_idx}.cross_attn_q.weight"] = qkv[0]
                gguf_weights[f"blk.{layer_idx}.cross_attn_k.weight"] = qkv[1]
                gguf_weights[f"blk.{layer_idx}.cross_attn_v.weight"] = qkv[2]
                continue
            elif rest == "cross_attention.out_proj.weight":
                new_name = f"blk.{layer_idx}.cross_attn_output.weight"
            else:
                new_name = f"blk.{layer_idx}.{rest}"

        # DepFormer slices (MLX style)
        elif name.startswith("depformer.slices."):
            new_name = name

        # DepFormer layers (PyTorch style - shared weights)
        elif name.startswith("depformer.layers.") or name.startswith("depformer.transformer.layers."):
            new_name = name

        # DepFormer embeddings (PyTorch style)
        elif name.startswith("depformer_text_emb."):
            new_name = name.replace("depformer_text_emb.", "depformer.slices.0.emb.")
        elif name.startswith("depformer_emb."):
            parts = name.split(".")
            idx = int(parts[1])
            rest = ".".join(parts[2:])
            new_name = f"depformer.slices.{idx + 1}.emb.{rest}"

        # DepFormer input projections
        elif name.startswith("depformer_in."):
            parts = name.split(".")
            idx = int(parts[1])
            new_name = f"depformer.slices.{idx}.linear_in.weight"

        # DepFormer output projections (linears.{i})
        elif name.startswith("linears."):
            parts = name.split(".")
            idx = int(parts[1])
            new_name = f"depformer.slices.{idx}.linear_out.weight"

        # DepFormer linear_in (MLX style)
        elif name.startswith("depformer.linear_in."):
            new_name = name

        # Condition provider
        elif name.startswith("condition_provider."):
            new_name = name

        # Extra heads
        elif name.startswith("extra_heads."):
            new_name = name

        else:
            new_name = name

        gguf_weights[new_name] = tensor

    return gguf_weights


def write_moshi_gguf(
    output_path: str,
    weights: dict[str, np.ndarray],
    config: dict,
    outtype: str = "f16",
    tokenizer_path: Path = None,
    include_depformer: bool = False
):
    """Write Moshi model to GGUF format."""

    writer = GGUFWriter(output_path)

    # Load tokenizer if provided
    vocab, scores, token_types = None, None, None
    if tokenizer_path and tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}...")
        vocab, scores, token_types = load_tokenizer(tokenizer_path)

    # Determine output dtype
    if outtype == "f32":
        ggml_type = GGMLType.F32
    elif outtype == "f16":
        ggml_type = GGMLType.F16
    elif outtype == "bf16":
        ggml_type = GGMLType.BF16
    else:
        raise ValueError(f"Unsupported output type: {outtype}")

    # === Metadata ===

    # General
    writer.add_string("general.architecture", "moshi")
    writer.add_string("general.name", config.get("name", "moshi"))
    writer.add_uint32("general.file_type", ggml_type)

    # Get dimensions from config or infer from weights
    d_model = config.get("dim", config.get("d_model", 2048))
    n_layers = config.get("num_layers", 16)
    n_heads = config.get("num_heads", 16)
    n_vocab_text_in = config.get("text_card", 48000) + 1
    n_vocab_text_out = config.get("text_card", 48000)
    n_vocab_audio = config.get("card", 2048) + 1
    n_audio_codebooks = config.get("n_q", 16)
    context_length = config.get("context", 3000)

    # Try to infer from weights if not in config
    if "token_embd.weight" in weights:
        n_vocab_text_in, d_model = weights["token_embd.weight"].shape

    # Transformer config
    writer.add_uint32("moshi.embedding_length", d_model)
    writer.add_uint32("moshi.block_count", n_layers)
    writer.add_uint32("moshi.attention.head_count", n_heads)
    writer.add_uint32("moshi.attention.head_count_kv", n_heads)  # No GQA in Moshi
    writer.add_uint32("moshi.context_length", context_length)
    writer.add_uint32("moshi.rope.dimension_count", d_model // n_heads)
    writer.add_float32("moshi.rope.freq_base", config.get("max_period", 100000))
    writer.add_float32("moshi.attention.layer_norm_rms_epsilon", 1e-8)

    # Try to infer n_ff from weights
    n_ff = config.get("hidden_size", config.get("ffn_dim", 0))
    if n_ff == 0 and "blk.0.ffn_gate.weight" in weights:
        # ffn_gate is [n_ff, n_embd] after transposition
        n_ff = weights["blk.0.ffn_gate.weight"].shape[0]
    elif n_ff == 0:
        # Default for Moshi 7B
        n_ff = 11264

    writer.add_uint32("moshi.feed_forward_length", n_ff)

    # Vocabulary
    writer.add_uint32("moshi.vocab_size_text_in", n_vocab_text_in)
    writer.add_uint32("moshi.vocab_size_text_out", n_vocab_text_out)
    writer.add_uint32("moshi.vocab_size_audio", n_vocab_audio)
    writer.add_uint32("moshi.audio_codebooks", n_audio_codebooks)

    # DepFormer config
    dep_q = config.get("dep_q", 8)
    depformer_dim = config.get("depformer_dim", 1024)
    depformer_layers = config.get("depformer_num_layers", 6)
    depformer_heads = config.get("depformer_num_heads", 16)

    writer.add_uint32("moshi.depformer.num_slices", dep_q)
    writer.add_uint32("moshi.depformer.embedding_length", depformer_dim)
    writer.add_uint32("moshi.depformer.block_count", depformer_layers)
    writer.add_uint32("moshi.depformer.attention.head_count", depformer_heads)

    # Audio delays
    delays = config.get("delays", [0] + [2] * (n_audio_codebooks - 1))
    if delays and delays[0] == 0:  # First delay is text, skip it
        audio_delays = delays[1:] if len(delays) > 1 else delays
    else:
        audio_delays = delays
    writer.add_array("moshi.audio_delays", audio_delays, GGUFValueType.INT32)

    # DepFormer weight schedule (if any)
    depformer_schedule = config.get("depformer_weights_per_step_schedule")
    if depformer_schedule:
        writer.add_array("moshi.depformer.weights_schedule", depformer_schedule, GGUFValueType.INT32)

    # Demux second stream
    writer.add_bool("moshi.demux_second_stream", config.get("demux_second_stream", False))

    # Cross attention (for voice conditioning)
    writer.add_bool("moshi.cross_attention", config.get("cross_attention", False))

    # === Tokenizer ===
    if vocab is not None:
        print(f"Writing tokenizer with {len(vocab)} tokens...")
        writer.add_string("tokenizer.ggml.model", "llama")  # SPM is compatible with llama tokenizer
        writer.add_array("tokenizer.ggml.tokens", vocab, GGUFValueType.STRING)
        writer.add_array("tokenizer.ggml.scores", scores, GGUFValueType.FLOAT32)
        writer.add_array("tokenizer.ggml.token_type", token_types, GGUFValueType.INT32)
        # Special tokens
        writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
        writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
        writer.add_uint32("tokenizer.ggml.unknown_token_id", 0)
        writer.add_bool("tokenizer.ggml.add_bos_token", False)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
    else:
        print("Warning: No tokenizer provided, model may not work correctly")

    # === Tensors ===

    # Filter to only tensors llama.cpp currently expects for Moshi
    # Base transformer + audio embeddings
    expected_prefixes = [
        "token_embd.", "output_norm.", "output.",
        "blk.",  # Transformer blocks
        "audio_embd.",  # Audio codebook embeddings (16 codebooks)
    ]

    # Include DepFormer if requested (228 additional tensors)
    if include_depformer:
        expected_prefixes.append("depformer.")
        print("Including DepFormer tensors...")

    filtered_weights = {}
    for name, tensor in weights.items():
        if any(name.startswith(p) for p in expected_prefixes):
            filtered_weights[name] = tensor

    print(f"\nWriting {len(filtered_weights)} tensors (filtered from {len(weights)})...")
    for name, tensor in sorted(filtered_weights.items()):
        if tensor.size == 0:
            print(f"  Skipping empty tensor: {name}")
            continue

        # GGML expects weights transposed compared to PyTorch
        # Embeddings: [n_vocab, n_embd] -> [n_embd, n_vocab]
        # Linear weights: [out_features, in_features] -> [in_features, out_features]
        if tensor.ndim == 2:
            tensor = tensor.T

        # Truncate token_embd to match tokenizer vocab size if needed
        if name == "token_embd.weight" and vocab is not None and tensor.shape[1] > len(vocab):
            print(f"  Truncating {name} from {tensor.shape[1]} to {len(vocab)} vocab size")
            tensor = tensor[:, :len(vocab)]

        # Ensure contiguous and correct type
        tensor = np.ascontiguousarray(tensor)

        # 1D tensors (norms, biases) should be F32 for compatibility
        tensor_type = GGMLType.F32 if tensor.ndim == 1 else ggml_type
        writer.add_tensor(name, tensor, tensor_type)

    # Update metadata with actual tensor count
    print(f"  Total tensors written: {len(filtered_weights)}")

    print(f"\nWriting GGUF to {output_path}...")
    writer.write()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Moshi model to GGUF format")
    parser.add_argument(
        "--model-path", "-m",
        type=Path,
        required=True,
        help="Path to Moshi model directory (containing .safetensors or config.json)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="moshi.gguf",
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16"],
        default="f16",
        help="Output data type"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (if not in model directory)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=Path,
        help="Path to SentencePiece tokenizer model file (.model)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--include-depformer",
        action="store_true",
        help="Include DepFormer tensors in output (228 additional tensors)"
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Load weights first (needed for config inference)
    print(f"Loading weights from {args.model_path}...")
    weights = load_weights(args.model_path)

    if not weights:
        print("Error: No weights found")
        sys.exit(1)

    print(f"Loaded {len(weights)} tensors")

    # Load config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = load_config(args.model_path)

    # Infer missing config from weights
    inferred = infer_config_from_weights(weights)
    for key, value in inferred.items():
        if key not in config:
            config[key] = value
            print(f"Inferred config: {key} = {value}")

    if args.verbose:
        print(f"Config: {json.dumps(config, indent=2)}")

    # Filter out test data tensors (not model weights)
    test_data_keys = {"codes", "logits", "mask", "text_logits", "text_mask"}
    weights = {k: v for k, v in weights.items() if k not in test_data_keys}

    # Reconstruct full embeddings from low-rank factorization if present
    if any('low_rank' in name for name in weights.keys()):
        print("Detected low-rank embeddings, reconstructing...")
        weights = reconstruct_low_rank_embeddings(weights)

    # Check if these are PyTorch weights and need name conversion
    # PyTorch weights have: emb.*, depformer_*, linears.*, out_norm.alpha, in_proj_weight
    needs_conversion = any(
        name.startswith("emb.") or
        name.startswith("depformer_") or
        name.startswith("linears.") or
        "alpha" in name or
        "in_proj_weight" in name
        for name in weights.keys()
    )

    if needs_conversion:
        print("Detected PyTorch weight format, converting names...")
        weights = convert_pytorch_to_mlx_names(weights, config)

    # Convert to GGUF tensor names
    print("Converting to GGUF tensor names...")
    weights = convert_to_gguf_names(weights)

    if args.verbose:
        print("\nTensor names:")
        for name in sorted(weights.keys()):
            print(f"  {name}: {weights[name].shape}")

    # Find tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        # Try to find tokenizer in model directory
        for name in ["tokenizer.model", "tokenizer_spm_32k_3.model"]:
            candidate = args.model_path / name
            if candidate.exists():
                tokenizer_path = candidate
                break
            # Also check parent directory
            candidate = args.model_path.parent / name
            if candidate.exists():
                tokenizer_path = candidate
                break

    # Write GGUF
    write_moshi_gguf(args.output, weights, config, args.outtype, tokenizer_path, args.include_depformer)

    print(f"\nSuccessfully converted to {args.output}")
    print(f"  Tensors: {len(weights)}")
    print(f"  Format: {args.outtype}")


if __name__ == "__main__":
    main()
