#!/usr/bin/env python3
"""
Convert Mimi codec model to GGUF format for llama.cpp.

Mimi is the neural audio codec used by Moshi. It consists of:
- SEANet encoder/decoder (convolutional)
- Transformer for encoder/decoder
- Split Residual Vector Quantizer (SRVQ)

For TTS, we only need the decoder path:
- Codebook lookup (quantizer.decode)
- Upsample
- Decoder transformer
- SEANet decoder

Usage:
    python convert_mimi_to_gguf.py --model-path kyutai/moshiko-pytorch-bf16 --output mimi.gguf
    python convert_mimi_to_gguf.py --weights-file mimi.safetensors --output mimi.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add gguf module path
LLAMA_CPP_PATH = Path(__file__).parent.parent.parent / "llama.cpp"
sys.path.insert(0, str(LLAMA_CPP_PATH / "gguf-py"))

import gguf

# Mimi config from moshi_mlx/moshi_mlx/models/mimi.py
DEFAULT_CONFIG = {
    "channels": 1,
    "sample_rate": 24000,
    "frame_rate": 12.5,
    "renormalize": True,
    "seanet": {
        "dimension": 512,
        "channels": 1,
        "causal": True,
        "nfilters": 64,
        "nresidual_layers": 1,
        "ratios": [8, 6, 5, 4],
        "ksize": 7,
        "residual_ksize": 3,
        "last_ksize": 3,
        "dilation_base": 2,
        "pad_mode": "constant",
        "true_skip": True,
        "compress": 2,
    },
    "transformer": {
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 8,
        "causal": True,
        "norm_first": True,
        "bias_ff": False,
        "bias_attn": False,
        "layer_scale": 0.01,
        "positional_embedding": "rope",
        "use_conv_bias": True,
        "gating": False,
        "norm": "layer_norm",
        "context": 250,
        "max_period": 10000,
        "max_seq_len": 8192,
        "kv_repeat": 1,
        "dim_feedforward": 2048,
        "conv_layout": True,
        "use_conv_block": False,
        "cross_attention": False,
        "conv_kernel_size": 3,
    },
    "quantizer_nq": 8,  # default, can be 16 or 32
    "quantizer_bins": 2048,
    "quantizer_dim": 256,
}


def load_model_weights(model_path: str | None, weights_file: str | None):
    """Load model weights from HuggingFace model or local file."""
    weights = {}

    if weights_file:
        # Load directly from safetensors or torch file
        path = Path(weights_file)
        if path.suffix == ".safetensors":
            # Use torch for bf16 support
            try:
                import torch
                from safetensors.torch import load_file as safe_load_torch
                state_dict = safe_load_torch(str(path))
                for key, tensor in state_dict.items():
                    weights[key] = tensor.float().numpy()
            except ImportError:
                from safetensors import safe_open
                with safe_open(path, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
        elif path.suffix in (".pt", ".pth", ".bin"):
            import torch
            state_dict = torch.load(path, map_location="cpu")
            for key, value in state_dict.items():
                weights[key] = value.numpy()
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
    elif model_path:
        # Load from HuggingFace
        from huggingface_hub import hf_hub_download

        # Try to download mimi weights
        for filename in ["tokenizer-e351c8d8-checkpoint125.safetensors", "mimi.safetensors", "model.safetensors"]:
            try:
                weights_path = hf_hub_download(repo_id=model_path, filename=filename)
                print(f"Downloaded {filename} from {model_path}")

                from safetensors import safe_open
                with safe_open(weights_path, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
                break
            except Exception as e:
                print(f"Could not load {filename}: {e}")
                continue
    else:
        raise ValueError("Must specify either --model-path or --weights-file")

    return weights


def convert_pytorch_to_mlx_names(weights: dict) -> dict:
    """Convert PyTorch tensor names to MLX-style names (matching mimi.py load_pytorch_weights)."""
    converted = {}

    for key, value in weights.items():
        k = key

        # Remove leading underscores from path components
        k = ".".join([s.removeprefix("_") for s in k.split(".")])

        # Encoder/decoder model prefix
        if k.startswith("encoder.model."):
            k = k.replace("encoder.model.", "encoder.")
        if k.startswith("decoder.model."):
            k = k.replace("decoder.model.", "decoder.")

        # In-projection weight
        if k.endswith(".in_proj_weight"):
            k = k.replace(".in_proj_weight", ".in_proj.weight")

        # Gating linear layers
        if k.endswith(".linear1.weight"):
            k = k.replace(".linear1.weight", ".gating.linear1.weight")
        if k.endswith(".linear2.weight"):
            k = k.replace(".linear2.weight", ".gating.linear2.weight")

        # Decoder layer mapping (hardcoded indices from PyTorch)
        for layer_idx, decoder_idx in enumerate([2, 5, 8, 11]):
            k = k.replace(f"decoder.{decoder_idx}.", f"decoder.layers.{layer_idx}.upsample.")
            k = k.replace(f"decoder.{decoder_idx + 1}.", f"decoder.layers.{layer_idx}.residuals.0.")

        # Encoder layer mapping
        for layer_idx, encoder_idx in enumerate([1, 4, 7, 10]):
            k = k.replace(f"encoder.{encoder_idx}.", f"encoder.layers.{layer_idx}.residuals.0.")
            k = k.replace(f"encoder.{encoder_idx + 2}.", f"encoder.layers.{layer_idx}.downsample.")

        # Initial and final conv layers
        k = k.replace("decoder.0.", "decoder.init_conv1d.")
        k = k.replace("decoder.14.", "decoder.final_conv1d.")
        k = k.replace("encoder.0.", "encoder.init_conv1d.")
        k = k.replace("encoder.14.", "encoder.final_conv1d.")

        # Block indices
        k = k.replace(".block.1.", ".block.0.")
        k = k.replace(".block.3.", ".block.1.")

        # Handle conv weight transposition (PyTorch: outC, inC, kSize -> MLX: outC, kSize, inC)
        if (k.endswith(".conv.weight") or
            k.endswith(".output_proj.weight") or
            k.endswith(".input_proj.weight")):
            value = np.swapaxes(value, -1, -2)

        # Handle conv-transposed weights (PyTorch: inC, outC, kSize -> MLX: outC, kSize, inC)
        if k.endswith(".convtr.weight"):
            value = np.transpose(value, (1, 2, 0))

        converted[k] = value

    return converted


def convert_to_gguf_names(weights: dict, include_encoder: bool = False) -> dict:
    """Convert MLX-style names to GGUF tensor names."""
    converted = {}

    for key, value in weights.items():
        k = key

        # Handle encoder tensors
        if k.startswith("encoder.") or k.startswith("encoder_transformer."):
            if not include_encoder:
                continue
            # Encoder transformer
            if k.startswith("encoder_transformer."):
                k = k.replace("encoder_transformer.", "enc_transformer.")
            # SEANet encoder
            if k.startswith("encoder."):
                k = k.replace("encoder.", "seanet_enc.")

        # Quantizer
        if k.startswith("quantizer."):
            # rvq_first and rvq_rest
            k = k.replace("quantizer.", "quantizer.")

        # Decoder transformer
        if k.startswith("decoder_transformer."):
            k = k.replace("decoder_transformer.", "dec_transformer.")

        # SEANet decoder
        if k.startswith("decoder."):
            k = k.replace("decoder.", "seanet_dec.")

        # Upsample/downsample
        if k.startswith("upsample."):
            k = k.replace("upsample.", "upsample.")
        if k.startswith("downsample."):
            if not include_encoder:
                continue
            k = k.replace("downsample.", "downsample.")

        converted[k] = value

    return converted


def filter_decoder_weights(weights: dict) -> dict:
    """Filter to only include decoder-related weights."""
    filtered = {}

    decoder_prefixes = [
        "decoder.",          # SEANet decoder
        "decoder_transformer.",  # Decoder transformer
        "quantizer.",        # Vector quantizer (need for decoding)
        "upsample.",         # Upsampling
    ]

    for key, value in weights.items():
        for prefix in decoder_prefixes:
            if key.startswith(prefix):
                filtered[key] = value
                break

    return filtered


def filter_encoder_weights(weights: dict) -> dict:
    """Filter to only include encoder-related weights."""
    filtered = {}

    encoder_prefixes = [
        "encoder.",              # SEANet encoder
        "encoder_transformer.",  # Encoder transformer
        "quantizer.",            # Vector quantizer (need for encoding)
        "downsample.",           # Downsampling
    ]

    for key, value in weights.items():
        for prefix in encoder_prefixes:
            if key.startswith(prefix):
                filtered[key] = value
                break

    return filtered


def write_mimi_gguf(weights: dict, config: dict, output_path: str):
    """Write Mimi weights to GGUF format."""

    writer = gguf.GGUFWriter(output_path, "mimi")

    # Write architecture metadata
    writer.add_name("mimi")

    # Standard llama.cpp hyperparameters
    transformer = config["transformer"]
    writer.add_context_length(transformer["max_seq_len"])
    writer.add_embedding_length(transformer["d_model"])
    writer.add_block_count(transformer["num_layers"])
    writer.add_head_count(transformer["num_heads"])
    writer.add_head_count_kv(transformer["num_heads"])  # No GQA
    writer.add_feed_forward_length(transformer["dim_feedforward"])
    writer.add_layer_norm_eps(1e-5)  # Default LayerNorm eps

    # SEANet config
    seanet = config["seanet"]
    writer.add_uint32("mimi.seanet.dimension", seanet["dimension"])
    writer.add_uint32("mimi.seanet.channels", seanet["channels"])
    writer.add_uint32("mimi.seanet.nfilters", seanet["nfilters"])
    writer.add_uint32("mimi.seanet.nresidual_layers", seanet["nresidual_layers"])
    writer.add_uint32("mimi.seanet.compress", seanet["compress"])
    writer.add_uint32("mimi.seanet.ksize", seanet["ksize"])
    writer.add_uint32("mimi.seanet.residual_ksize", seanet["residual_ksize"])
    writer.add_uint32("mimi.seanet.last_ksize", seanet["last_ksize"])
    writer.add_uint32("mimi.seanet.dilation_base", seanet["dilation_base"])
    writer.add_bool("mimi.seanet.causal", seanet["causal"])
    writer.add_bool("mimi.seanet.true_skip", seanet["true_skip"])

    # Store ratios as array
    ratios = seanet["ratios"]
    writer.add_array("mimi.seanet.ratios", ratios)

    # Transformer config
    transformer = config["transformer"]
    writer.add_uint32("mimi.transformer.d_model", transformer["d_model"])
    writer.add_uint32("mimi.transformer.num_heads", transformer["num_heads"])
    writer.add_uint32("mimi.transformer.num_layers", transformer["num_layers"])
    writer.add_uint32("mimi.transformer.dim_feedforward", transformer["dim_feedforward"])
    writer.add_uint32("mimi.transformer.context", transformer["context"])
    writer.add_uint32("mimi.transformer.max_period", transformer["max_period"])
    writer.add_uint32("mimi.transformer.max_seq_len", transformer["max_seq_len"])
    writer.add_float32("mimi.transformer.layer_scale", transformer["layer_scale"])
    writer.add_bool("mimi.transformer.causal", transformer["causal"])
    writer.add_bool("mimi.transformer.norm_first", transformer["norm_first"])

    # Quantizer config
    writer.add_uint32("mimi.quantizer.nq", config["quantizer_nq"])
    writer.add_uint32("mimi.quantizer.bins", config["quantizer_bins"])
    writer.add_uint32("mimi.quantizer.dim", config["quantizer_dim"])

    # Audio config
    writer.add_uint32("mimi.sample_rate", config["sample_rate"])
    writer.add_float32("mimi.frame_rate", config["frame_rate"])
    writer.add_uint32("mimi.channels", config["channels"])

    # Add minimal tokenizer (Mimi uses codebook indices, not text tokens)
    # Create a simple vocabulary with codebook tokens
    vocab_size = config["quantizer_bins"]  # 2048
    tokens = [f"<code_{i}>" for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size  # NORMAL

    writer.add_tokenizer_model("llama")  # Use llama tokenizer (SentencePiece compatible)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(0)

    # Write tensors
    for name, tensor in weights.items():
        # Ensure tensor is contiguous
        tensor = np.ascontiguousarray(tensor)

        # Determine data type - convert all to f32 for compatibility
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        data_type = gguf.GGMLQuantizationType.F32

        # Add .weight or .bias suffix as expected by llama.cpp
        # Most tensors are weights, biases have _b or .bias in name
        if not name.endswith(".weight") and not name.endswith(".bias"):
            if ".bias" in name or "_b." in name or name.endswith("_b"):
                name = name + ".bias"
            else:
                name = name + ".weight"

        writer.add_tensor(name, tensor, raw_dtype=data_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Wrote {len(weights)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Mimi codec to GGUF")
    parser.add_argument("--model-path", type=str, help="HuggingFace model path (e.g., kyutai/moshiko-pytorch-bf16)")
    parser.add_argument("--weights-file", type=str, help="Direct path to weights file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--num-codebooks", type=int, default=32, help="Number of codebooks (default: 32)")
    parser.add_argument("--decoder-only", action="store_true", help="Only include decoder weights (for TTS)")
    parser.add_argument("--encoder-only", action="store_true", help="Only include encoder weights (for STT)")
    parser.add_argument("--full", action="store_true", default=True, help="Include both encoder and decoder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.model_path and not args.weights_file:
        parser.error("Must specify either --model-path or --weights-file")

    # Load weights
    print("Loading weights...")
    weights = load_model_weights(args.model_path, args.weights_file)
    print(f"Loaded {len(weights)} tensors")

    if args.verbose:
        print("\nOriginal tensor names:")
        for name in sorted(weights.keys())[:20]:
            print(f"  {name}: {weights[name].shape}")
        if len(weights) > 20:
            print(f"  ... and {len(weights) - 20} more")

    # Check if conversion needed (PyTorch names have different patterns)
    needs_conversion = any(
        ".model." in k or
        k.startswith("encoder.0.") or
        k.startswith("decoder.0.") or
        ".in_proj_weight" in k
        for k in weights.keys()
    )

    if needs_conversion:
        print("Converting PyTorch names to MLX format...")
        weights = convert_pytorch_to_mlx_names(weights)

    if args.verbose:
        print("\nAfter MLX conversion:")
        for name in sorted(weights.keys())[:20]:
            print(f"  {name}: {weights[name].shape}")

    # Determine which parts to include
    include_encoder = not args.decoder_only
    include_decoder = not args.encoder_only

    if args.encoder_only:
        print("Filtering to encoder-only weights...")
        weights = filter_encoder_weights(weights)
        print(f"Kept {len(weights)} encoder tensors")
    elif args.decoder_only:
        print("Filtering to decoder-only weights...")
        weights = filter_decoder_weights(weights)
        print(f"Kept {len(weights)} decoder tensors")
    else:
        print("Including full model (encoder + decoder)...")

    # Convert to GGUF names
    print("Converting to GGUF names...")
    weights = convert_to_gguf_names(weights, include_encoder=include_encoder)

    if args.verbose:
        print("\nFinal tensor names:")
        for name in sorted(weights.keys()):
            print(f"  {name}: {weights[name].shape}")

    # Set config
    config = DEFAULT_CONFIG.copy()
    config["quantizer_nq"] = args.num_codebooks

    # Write GGUF
    print(f"Writing GGUF to {args.output}...")
    write_mimi_gguf(weights, config, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
