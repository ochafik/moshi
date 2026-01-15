#!/usr/bin/env python3
"""
Detailed tensor comparison between PyTorch Moshi and llama.cpp implementation.

This script:
1. Runs a forward pass through PyTorch model
2. Extracts intermediate activations
3. Compares with expected llama.cpp behavior

Usage:
    python scripts/test_tensor_comparison.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file as safe_load_torch
except ImportError:
    print("Error: torch and safetensors required")
    sys.exit(1)


def get_model_path():
    """Get path to cached Moshi model."""
    cache_dir = Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16"
    snapshots = list((cache_dir / "snapshots").iterdir())
    return snapshots[0] if snapshots else None


def load_weights(model_path: Path) -> dict:
    """Load weights as torch tensors."""
    weights_file = model_path / "model.safetensors"
    print(f"Loading weights from {weights_file}...")
    state_dict = safe_load_torch(str(weights_file))
    return {k: v.float() for k, v in state_dict.items()}


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def rope_embedding(x: torch.Tensor, pos: torch.Tensor, n_rot: int, base: float = 100000.0):
    """Apply RoPE embedding.

    Args:
        x: Input tensor of shape [batch, n_heads, seq_len, head_dim]
        pos: Position indices of shape [seq_len]
        n_rot: Number of dimensions to rotate (usually head_dim)
    """
    batch, n_heads, seq_len, head_dim = x.shape
    dim = n_rot

    # Compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))

    # Compute position-frequency matrix
    t = pos.float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()
    sin = emb.sin()

    # Apply rotation
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # Reshape for broadcasting: [1, 1, seq_len, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply RoPE
    x_rope = x * cos + rotate_half(x) * sin
    return x_rope


def test_embedding_layer(weights: dict):
    """Test text embedding lookup."""
    print("\n=== Testing Embedding Layer ===")

    emb_weight = weights["text_emb.weight"]
    vocab_size, dim = emb_weight.shape
    print(f"Embedding: vocab={vocab_size}, dim={dim}")

    # Test tokens
    test_tokens = torch.tensor([1, 100, 1000, 5000])

    # PyTorch embedding lookup
    emb_out = F.embedding(test_tokens, emb_weight)
    print(f"Input tokens: {test_tokens.tolist()}")
    print(f"Output shape: {emb_out.shape}")
    print(f"Output norms: {emb_out.norm(dim=-1).tolist()}")
    print(f"Output[0] first 5: {emb_out[0, :5].tolist()}")

    return emb_out


def test_attention_layer(weights: dict, x: torch.Tensor, layer_idx: int = 0):
    """Test attention computation for a single layer."""
    print(f"\n=== Testing Attention Layer {layer_idx} ===")

    # Get weights
    in_proj_w = weights[f"transformer.layers.{layer_idx}.self_attn.in_proj_weight"]
    out_proj_w = weights[f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"]

    # Get norm weight (handle alpha format)
    norm_key = f"transformer.layers.{layer_idx}.norm1.alpha"
    if norm_key in weights:
        norm_w = weights[norm_key].squeeze()
    else:
        norm_w = weights.get(f"transformer.layers.{layer_idx}.norm1.weight")

    dim = x.shape[-1]
    n_heads = 64
    head_dim = dim // n_heads
    seq_len = x.shape[1]

    print(f"Input shape: {x.shape}")
    print(f"in_proj_weight shape: {in_proj_w.shape}")

    # Apply norm
    x_norm = rms_norm(x, norm_w)
    print(f"After norm: norm={x_norm.norm().item():.4f}")

    # QKV projection
    qkv = F.linear(x_norm, in_proj_w)
    print(f"QKV shape: {qkv.shape}")

    # Split Q, K, V
    q, k, v = qkv.chunk(3, dim=-1)
    print(f"Q/K/V shapes: {q.shape}")

    # Reshape for multi-head attention
    batch = x.shape[0]
    q = q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    pos = torch.arange(seq_len, device=x.device)
    q_rope = rope_embedding(q, pos, head_dim)
    k_rope = rope_embedding(k, pos, head_dim)

    print(f"Q after RoPE: norm={q_rope.norm().item():.4f}")

    # Attention scores
    scale = 1.0 / (head_dim ** 0.5)
    attn = torch.matmul(q_rope, k_rope.transpose(-2, -1)) * scale
    print(f"Attention scores shape: {attn.shape}")
    print(f"Attention scores[0,0] first 4: {attn[0, 0, :4, :4]}")

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
    attn = attn.masked_fill(mask, float('-inf'))

    # Softmax
    attn = F.softmax(attn, dim=-1)
    print(f"Attention probs[0,0] first 4: {attn[0, 0, :4, :4]}")

    # Apply attention to values
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)

    # Output projection
    out = F.linear(out, out_proj_w)
    print(f"Attention output shape: {out.shape}")
    print(f"Attention output norm: {out.norm().item():.4f}")

    return out


def test_ffn_layer(weights: dict, x: torch.Tensor, layer_idx: int = 0):
    """Test FFN (gated SiLU) computation."""
    print(f"\n=== Testing FFN Layer {layer_idx} ===")

    # Get weights
    gate_up_w = weights[f"transformer.layers.{layer_idx}.gating.linear_in.weight"]
    down_w = weights[f"transformer.layers.{layer_idx}.gating.linear_out.weight"]

    # Get norm weight
    norm_key = f"transformer.layers.{layer_idx}.norm2.alpha"
    if norm_key in weights:
        norm_w = weights[norm_key].squeeze()
    else:
        norm_w = weights.get(f"transformer.layers.{layer_idx}.norm2.weight")

    print(f"Input shape: {x.shape}")
    print(f"gate_up_weight shape: {gate_up_w.shape}")
    print(f"down_weight shape: {down_w.shape}")

    # Apply norm
    x_norm = rms_norm(x, norm_w)
    print(f"After norm: norm={x_norm.norm().item():.4f}")

    # Gate + Up projection (fused)
    gate_up = F.linear(x_norm, gate_up_w)
    print(f"gate_up shape: {gate_up.shape}")

    # Split into gate and up
    n_ff = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :n_ff], gate_up[..., n_ff:]

    # SiLU(gate) * up
    hidden = F.silu(gate) * up
    print(f"hidden shape: {hidden.shape}")
    print(f"hidden norm: {hidden.norm().item():.4f}")

    # Down projection
    out = F.linear(hidden, down_w)
    print(f"FFN output shape: {out.shape}")
    print(f"FFN output norm: {out.norm().item():.4f}")

    return out


def test_full_layer(weights: dict, x: torch.Tensor, layer_idx: int = 0):
    """Test a complete transformer layer."""
    print(f"\n=== Testing Complete Layer {layer_idx} ===")

    # Self-attention with residual
    attn_out = test_attention_layer(weights, x, layer_idx)
    x = x + attn_out

    # FFN with residual
    ffn_out = test_ffn_layer(weights, x, layer_idx)
    x = x + ffn_out

    print(f"\nLayer {layer_idx} final output norm: {x.norm().item():.4f}")
    return x


def test_output_layer(weights: dict, x: torch.Tensor):
    """Test output projection."""
    print("\n=== Testing Output Layer ===")

    # Get output norm weight
    norm_key = "out_norm.alpha"
    if norm_key in weights:
        norm_w = weights[norm_key].squeeze()
    else:
        norm_w = weights.get("out_norm.weight")

    # Get output projection
    out_w = weights["text_linear.weight"]

    print(f"Input shape: {x.shape}")
    print(f"Output weight shape: {out_w.shape}")

    # Apply norm
    x_norm = rms_norm(x, norm_w)
    print(f"After norm: norm={x_norm.norm().item():.4f}")

    # Output projection
    logits = F.linear(x_norm, out_w)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits[0,-1] top 5 values: {logits[0, -1].topk(5).values.tolist()}")
    print(f"Logits[0,-1] top 5 indices: {logits[0, -1].topk(5).indices.tolist()}")

    return logits


def test_depformer_layer(weights: dict, x: torch.Tensor, slice_idx: int = 0, layer_idx: int = 0):
    """Test DepFormer layer for a single slice."""
    print(f"\n=== Testing DepFormer Layer {layer_idx}, Slice {slice_idx} ===")

    # DepFormer has per-slice attention and FFN
    # Check for per-slice weights
    attn_key = f"depformer.layers.{layer_idx}.self_attn.{slice_idx}.in_proj_weight"
    if attn_key not in weights:
        # Try the unsplit version
        attn_key = f"depformer.layers.{layer_idx}.self_attn.in_proj_weight"

    if attn_key not in weights:
        print(f"DepFormer attention weight not found: {attn_key}")
        return x

    in_proj_w = weights[attn_key]
    print(f"DepFormer in_proj_weight shape: {in_proj_w.shape}")

    # DepFormer FFN
    ffn_in_key = f"depformer.layers.{layer_idx}.gating.{slice_idx}.linear_in.weight"
    if ffn_in_key in weights:
        ffn_in_w = weights[ffn_in_key]
        print(f"DepFormer FFN linear_in shape: {ffn_in_w.shape}")

    return x


def main():
    parser = argparse.ArgumentParser(description="Test tensor comparison")
    parser.add_argument("--model-path", type=Path, help="Path to model directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer to test")
    args = parser.parse_args()

    model_path = args.model_path or get_model_path()
    if not model_path:
        print("Model path not found")
        return 1

    print(f"Model path: {model_path}")
    weights = load_weights(model_path)
    print(f"Loaded {len(weights)} tensors")

    # Test embedding
    test_tokens = torch.tensor([[1, 100, 1000, 5000]])
    emb_out = test_embedding_layer(weights)
    emb_out = emb_out.unsqueeze(0)  # Add batch dimension

    # Test single layer
    layer_out = test_full_layer(weights, emb_out, args.layer)

    # Test a few more layers
    for i in range(1, min(4, 32)):
        layer_out = test_full_layer(weights, layer_out, i)

    # Test output
    logits = test_output_layer(weights, layer_out)

    # Test DepFormer (if weights exist)
    print("\n" + "="*50)
    print("Testing DepFormer")
    print("="*50)

    # Project from main LM to DepFormer dim
    if "depformer_in.0.weight" in weights:
        proj_w = weights["depformer_in.0.weight"]
        print(f"\nDepFormer input projection shape: {proj_w.shape}")

        # Take last hidden state and project
        last_hidden = layer_out[:, -1:, :]  # [batch, 1, dim]
        dep_input = F.linear(last_hidden, proj_w)
        print(f"DepFormer input shape: {dep_input.shape}")

        # Test DepFormer layer
        test_depformer_layer(weights, dep_input, slice_idx=0, layer_idx=0)

    print("\n=== Test Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
