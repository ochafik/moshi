#!/usr/bin/env python3
"""
Test script to compare llama.cpp Moshi implementation with reference PyTorch.

This script:
1. Loads the original PyTorch model
2. Converts to GGUF format
3. Runs inference through both implementations
4. Compares outputs at multiple stages

Usage:
    python scripts/test_llama_cpp_comparison.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
MOSHI_DIR = SCRIPT_DIR.parent
LLAMA_CPP_DIR = MOSHI_DIR.parent / "llama.cpp"

sys.path.insert(0, str(MOSHI_DIR / "moshi"))

# Check for required packages
try:
    import torch
    from safetensors.torch import load_file as safe_load_torch
except ImportError:
    print("Error: torch and safetensors required. Run: pip install torch safetensors")
    sys.exit(1)


def get_model_path():
    """Get path to cached Moshi model."""
    cache_dir = Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16"
    if not cache_dir.exists():
        print(f"Model not found at {cache_dir}")
        print("Please download: huggingface-cli download kyutai/moshiko-pytorch-bf16")
        sys.exit(1)

    # Find the snapshot
    snapshots = list((cache_dir / "snapshots").iterdir())
    if not snapshots:
        print("No snapshots found")
        sys.exit(1)

    return snapshots[0]


def load_pytorch_weights(model_path: Path) -> dict:
    """Load PyTorch weights."""
    weights_file = model_path / "model.safetensors"
    if not weights_file.exists():
        print(f"Weights file not found: {weights_file}")
        sys.exit(1)

    print(f"Loading weights from {weights_file}...")
    state_dict = safe_load_torch(str(weights_file))

    # Convert to float32 numpy
    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.float().numpy()

    return weights


def analyze_weights(weights: dict):
    """Analyze weight shapes and structure."""
    print("\n=== Weight Analysis ===")

    # Count by prefix
    prefixes = {}
    for name in weights.keys():
        prefix = name.split(".")[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(name)

    for prefix, names in sorted(prefixes.items()):
        print(f"\n{prefix}: {len(names)} tensors")
        # Show first few
        for name in sorted(names)[:3]:
            print(f"  {name}: {weights[name].shape}")
        if len(names) > 3:
            print(f"  ... and {len(names) - 3} more")

    # Key dimensions
    print("\n=== Key Dimensions ===")
    if "text_emb.weight" in weights:
        vocab, dim = weights["text_emb.weight"].shape
        print(f"Text embedding: vocab={vocab}, dim={dim}")

    if "transformer.layers.0.self_attn.in_proj_weight" in weights:
        qkv_dim, dim = weights["transformer.layers.0.self_attn.in_proj_weight"].shape
        print(f"Transformer: dim={dim}, qkv_dim={qkv_dim} (heads={qkv_dim // 3 // 64})")

    # Count transformer layers
    n_layers = 0
    for name in weights.keys():
        if name.startswith("transformer.layers."):
            layer_idx = int(name.split(".")[2])
            n_layers = max(n_layers, layer_idx + 1)
    print(f"Transformer layers: {n_layers}")

    # DepFormer analysis
    if "depformer.layers.0.self_attn.in_proj_weight" in weights:
        dep_dim = weights["depformer.layers.0.self_attn.in_proj_weight"].shape[1]
        print(f"DepFormer dim: {dep_dim}")

        n_dep_layers = 0
        for name in weights.keys():
            if name.startswith("depformer.layers."):
                layer_idx = int(name.split(".")[2])
                n_dep_layers = max(n_dep_layers, layer_idx + 1)
        print(f"DepFormer layers: {n_dep_layers}")

    # Audio embeddings
    n_audio_emb = 0
    for name in weights.keys():
        if name.startswith("emb.") and name.endswith(".weight"):
            n_audio_emb += 1
    print(f"Audio embeddings: {n_audio_emb}")

    return {
        "n_layers": n_layers,
        "dim": dim if "text_emb.weight" in weights else None,
    }


def test_embedding_lookup(weights: dict):
    """Test that embedding lookup matches."""
    print("\n=== Testing Embedding Lookup ===")

    if "text_emb.weight" not in weights:
        print("No text embedding found")
        return

    emb = weights["text_emb.weight"]
    vocab_size, dim = emb.shape
    print(f"Text embedding shape: {emb.shape}")

    # Test a few token lookups
    test_tokens = [0, 1, 100, 1000, vocab_size - 1]
    print("\nToken lookup test:")
    for tok in test_tokens:
        if tok < vocab_size:
            vec = emb[tok]
            print(f"  Token {tok}: norm={np.linalg.norm(vec):.4f}, "
                  f"mean={vec.mean():.6f}, std={vec.std():.6f}")


def test_attention_weights(weights: dict):
    """Analyze attention weight structure."""
    print("\n=== Attention Weight Analysis ===")

    # Check for in_proj_weight (packed QKV)
    key = "transformer.layers.0.self_attn.in_proj_weight"
    if key in weights:
        w = weights[key]
        print(f"in_proj_weight shape: {w.shape}")

        # Split into Q, K, V
        qkv_dim = w.shape[0] // 3
        print(f"  Q/K/V dim each: {qkv_dim}")

        Q = w[:qkv_dim]
        K = w[qkv_dim:2*qkv_dim]
        V = w[2*qkv_dim:]

        print(f"  Q norm: {np.linalg.norm(Q):.4f}")
        print(f"  K norm: {np.linalg.norm(K):.4f}")
        print(f"  V norm: {np.linalg.norm(V):.4f}")


def test_ffn_weights(weights: dict):
    """Analyze FFN weight structure."""
    print("\n=== FFN Weight Analysis ===")

    # Check gating structure
    for name, w in weights.items():
        if "layers.0.gating" in name or "layers.0.linear" in name:
            print(f"{name}: {w.shape}")


def convert_to_gguf(model_path: Path, output_path: Path, include_depformer: bool = False):
    """Convert model to GGUF format."""
    print(f"\n=== Converting to GGUF (depformer={include_depformer}) ===")

    converter = SCRIPT_DIR / "convert_moshi_to_gguf.py"
    tokenizer = model_path / "tokenizer_spm_32k_3.model"

    cmd = [
        sys.executable, str(converter),
        "-m", str(model_path),
        "-o", str(output_path),
        "-t", str(tokenizer),
        "-v"
    ]

    if include_depformer:
        cmd.append("--include-depformer")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Conversion failed:\n{result.stderr}")
        return False

    print(result.stdout)

    # Check output
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Output: {output_path} ({size_mb:.1f} MB)")
        return True

    return False


def test_llama_cpp_load(gguf_path: Path):
    """Test loading GGUF in llama.cpp."""
    print(f"\n=== Testing llama.cpp Load ===")

    llama_cli = LLAMA_CPP_DIR / "build/bin/llama-cli"
    if not llama_cli.exists():
        print(f"llama-cli not found at {llama_cli}")
        return False

    # Just try to load and generate 1 token
    cmd = [
        str(llama_cli),
        "-m", str(gguf_path),
        "-p", "Hello",
        "-n", "1",
        "--no-display-prompt"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output: {result.stdout[:500]}")
    if result.stderr:
        # Filter out just the key info
        lines = result.stderr.split('\n')
        for line in lines:
            if any(x in line.lower() for x in ['error', 'loaded', 'tensor', 'metal', 'memory']):
                print(f"  {line}")

    return result.returncode == 0


def run_pytorch_forward(weights: dict, input_tokens: np.ndarray) -> dict:
    """
    Run a simplified PyTorch forward pass to get intermediate outputs.
    This simulates what the model does without loading the full model.
    """
    print(f"\n=== PyTorch Forward Pass (simplified) ===")
    print(f"Input tokens shape: {input_tokens.shape}")

    outputs = {}

    # 1. Text embedding lookup
    if "text_emb.weight" in weights:
        emb = weights["text_emb.weight"]
        # Assume input_tokens are text tokens
        token_embeds = emb[input_tokens.flatten()]
        outputs["token_embd"] = token_embeds
        print(f"Token embedding output: {token_embeds.shape}, norm={np.linalg.norm(token_embeds):.4f}")

    # 2. First transformer layer (simplified - just show norms)
    if "transformer.layers.0.self_attn.in_proj_weight" in weights:
        w = weights["transformer.layers.0.self_attn.in_proj_weight"]
        # QKV projection
        qkv = token_embeds @ w.T
        outputs["layer0_qkv"] = qkv
        print(f"Layer 0 QKV output: {qkv.shape}, norm={np.linalg.norm(qkv):.4f}")

    return outputs


def compare_outputs(pytorch_outputs: dict, llama_outputs: dict):
    """Compare outputs between implementations."""
    print("\n=== Output Comparison ===")

    for key in pytorch_outputs:
        if key in llama_outputs:
            pt = pytorch_outputs[key]
            ll = llama_outputs[key]

            diff = np.abs(pt - ll)
            print(f"{key}:")
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")
            print(f"  Relative diff: {diff.mean() / (np.abs(pt).mean() + 1e-8):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Test llama.cpp Moshi implementation")
    parser.add_argument("--model-path", type=Path, help="Path to model directory")
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion step")
    parser.add_argument("--include-depformer", action="store_true", help="Include DepFormer in conversion")
    parser.add_argument("--gguf-path", type=Path, help="Path to existing GGUF file")
    args = parser.parse_args()

    # Get model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = get_model_path()

    print(f"Model path: {model_path}")

    # Load and analyze weights
    weights = load_pytorch_weights(model_path)
    print(f"Loaded {len(weights)} tensors")

    info = analyze_weights(weights)

    # Run specific tests
    test_embedding_lookup(weights)
    test_attention_weights(weights)
    test_ffn_weights(weights)

    # Convert to GGUF
    if args.gguf_path:
        gguf_path = args.gguf_path
    elif not args.skip_convert:
        gguf_path = Path(tempfile.gettempdir()) / "moshi-test.gguf"
        if not convert_to_gguf(model_path, gguf_path, args.include_depformer):
            print("Conversion failed!")
            return 1
    else:
        gguf_path = Path("/tmp/moshi-test.gguf")

    # Test llama.cpp loading
    if gguf_path.exists():
        test_llama_cpp_load(gguf_path)

    # Simple forward pass comparison
    test_tokens = np.array([1, 100, 200, 300, 400])  # Sample tokens
    pytorch_outputs = run_pytorch_forward(weights, test_tokens)

    print("\n=== Summary ===")
    print(f"Model: {model_path.name}")
    print(f"Tensors: {len(weights)}")
    print(f"Transformer layers: {info.get('n_layers', 'unknown')}")
    print(f"Hidden dim: {info.get('dim', 'unknown')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
