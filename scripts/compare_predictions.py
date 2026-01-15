#!/usr/bin/env python3
"""
Compare top token predictions between PyTorch Moshi and llama.cpp.

This script:
1. Tokenizes a prompt
2. Runs PyTorch forward pass to get logits
3. Compares top-k predictions between implementations
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.torch import load_file as safe_load_torch
    import sentencepiece as spm
except ImportError:
    print("Error: torch, safetensors, and sentencepiece required")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
LLAMA_CPP_DIR = SCRIPT_DIR.parent.parent / "llama.cpp"


def get_model_path():
    """Get path to cached Moshi model."""
    cache_dir = Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16"
    snapshots = list((cache_dir / "snapshots").iterdir())
    return snapshots[0] if snapshots else None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def rope_embedding(x: torch.Tensor, pos: torch.Tensor, n_rot: int, base: float = 100000.0):
    """Apply RoPE embedding."""
    batch, n_heads, seq_len, head_dim = x.shape
    dim = n_rot

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    t = pos.float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()
    sin = emb.sin()

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + rotate_half(x) * sin


def pytorch_forward(weights: dict, token_ids: torch.Tensor, n_layers: int = 32):
    """Run simplified PyTorch forward pass."""
    # Text embedding
    emb_weight = weights["text_emb.weight"]
    x = torch.nn.functional.embedding(token_ids, emb_weight)
    x = x.unsqueeze(0)  # Add batch dim

    n_heads = 64
    dim = x.shape[-1]
    head_dim = dim // n_heads
    seq_len = x.shape[1]

    # Run through transformer layers
    for il in range(n_layers):
        # Self-attention
        norm_key = f"transformer.layers.{il}.norm1.alpha"
        if norm_key in weights:
            norm_w = weights[norm_key].squeeze()
        else:
            norm_w = weights.get(f"transformer.layers.{il}.norm1.weight")

        in_proj_w = weights[f"transformer.layers.{il}.self_attn.in_proj_weight"]
        out_proj_w = weights[f"transformer.layers.{il}.self_attn.out_proj.weight"]

        x_norm = rms_norm(x, norm_w)
        qkv = torch.nn.functional.linear(x_norm, in_proj_w)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
        v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)

        pos = torch.arange(seq_len, device=x.device)
        q = rope_embedding(q, pos, head_dim)
        k = rope_embedding(k, pos, head_dim)

        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(1, seq_len, dim)
        out = torch.nn.functional.linear(out, out_proj_w)

        x = x + out

        # FFN
        norm2_key = f"transformer.layers.{il}.norm2.alpha"
        if norm2_key in weights:
            norm2_w = weights[norm2_key].squeeze()
        else:
            norm2_w = weights.get(f"transformer.layers.{il}.norm2.weight")

        gate_up_w = weights[f"transformer.layers.{il}.gating.linear_in.weight"]
        down_w = weights[f"transformer.layers.{il}.gating.linear_out.weight"]

        x_norm = rms_norm(x, norm2_w)
        gate_up = torch.nn.functional.linear(x_norm, gate_up_w)

        n_ff = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :n_ff], gate_up[..., n_ff:]
        hidden = torch.nn.functional.silu(gate) * up
        ffn_out = torch.nn.functional.linear(hidden, down_w)

        x = x + ffn_out

    # Output norm and projection
    out_norm_key = "out_norm.alpha"
    if out_norm_key in weights:
        out_norm_w = weights[out_norm_key].squeeze()
    else:
        out_norm_w = weights.get("out_norm.weight")

    x = rms_norm(x, out_norm_w)
    logits = torch.nn.functional.linear(x, weights["text_linear.weight"])

    return logits


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and llama.cpp predictions")
    parser.add_argument("--model-path", type=Path, help="Path to model directory")
    parser.add_argument("--gguf-path", type=Path, default=Path("/tmp/moshi-7b-full.gguf"))
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--top-k", type=int, default=10, help="Compare top-k predictions")
    args = parser.parse_args()

    model_path = args.model_path or get_model_path()
    if not model_path:
        print("Model path not found")
        return 1

    print(f"Model path: {model_path}")

    # Load tokenizer
    tokenizer_path = Path("/tmp/tokenizer_spm_32k_3.model")
    if not tokenizer_path.exists():
        print(f"Tokenizer not found: {tokenizer_path}")
        return 1

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))

    # Tokenize prompt
    token_ids = sp.EncodeAsIds(args.prompt)
    print(f"Prompt: '{args.prompt}'")
    print(f"Token IDs: {token_ids}")

    # Load PyTorch weights
    print("\nLoading PyTorch weights...")
    weights_file = model_path / "model.safetensors"
    weights = safe_load_torch(str(weights_file))
    weights = {k: v.float() for k, v in weights.items()}

    # Run PyTorch forward pass
    print("Running PyTorch forward pass...")
    token_tensor = torch.tensor(token_ids)
    with torch.no_grad():
        logits = pytorch_forward(weights, token_tensor)

    # Get top-k predictions for last token
    last_logits = logits[0, -1]
    top_values, top_indices = torch.topk(last_logits, args.top_k)

    print(f"\nPyTorch top-{args.top_k} predictions for next token:")
    for i, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
        token = sp.IdToPiece(idx)
        print(f"  {i+1}. [{idx}] '{token}' (logit: {val:.4f})")

    # Run llama.cpp and get its predictions
    print(f"\nRunning llama.cpp inference...")
    llama_cli = LLAMA_CPP_DIR / "build/bin/llama-cli"
    if not llama_cli.exists():
        print(f"llama-cli not found: {llama_cli}")
        return 1

    # Generate a few tokens with llama.cpp
    cmd = [
        str(llama_cli),
        "-m", str(args.gguf_path),
        "-p", args.prompt,
        "-n", "5",
        "--no-display-prompt",
        "--temp", "0.0",  # Greedy sampling
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"llama.cpp failed: {result.stderr}")
        return 1

    # Parse output
    output = result.stdout.strip()
    print(f"\nllama.cpp output: '{output}'")

    # Tokenize llama.cpp output to get first predicted token
    if output:
        first_token = output.split()[0] if output.split() else output
        llama_first_id = sp.EncodeAsIds(first_token)
        print(f"llama.cpp first token ID(s): {llama_first_id}")

        # Check if llama.cpp prediction matches PyTorch top predictions
        pytorch_top_ids = top_indices.tolist()
        if llama_first_id and llama_first_id[0] in pytorch_top_ids:
            rank = pytorch_top_ids.index(llama_first_id[0]) + 1
            print(f"\n✓ llama.cpp prediction matches PyTorch top-{rank}")
        else:
            print(f"\n✗ llama.cpp prediction not in PyTorch top-{args.top_k}")
            print(f"  llama.cpp predicted: {llama_first_id}")
            print(f"  PyTorch top-{args.top_k}: {pytorch_top_ids}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
