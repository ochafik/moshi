#!/usr/bin/env python3
"""
Debug layer-by-layer comparison to find where PyTorch and llama.cpp diverge.

This script runs a forward pass step by step and prints intermediate values
that can be compared with llama.cpp's debug output.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
from moshi.models import loaders


def rms_norm(x, weight, eps=1e-8):
    """RMS normalization matching llama.cpp."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def main():
    print("Loading Moshi...")
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_path, device='cpu', dtype=torch.float32)
    moshi.eval()

    # Single token: 11725 ("Hello")
    token_id = 11725
    print(f"\nProcessing token {token_id}...")

    with torch.no_grad():
        # Text embedding
        x = moshi.text_emb.weight[token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, 4096]
        print(f"\n=== EMBEDDING ===")
        print(f"Shape: {x.shape}")
        print(f"Norm: {x.norm().item():.6f}")
        print(f"First 8: {x[0, 0, :8].tolist()}")
        print(f"Last 8: {x[0, 0, -8:].tolist()}")

        # Process first layer only
        layer = moshi.transformer.layers[0]

        print(f"\n=== LAYER 0 ===")

        # Attention norm
        norm1_weight = layer.norm1.alpha.squeeze()
        h = rms_norm(x, norm1_weight, eps=1e-8)
        print(f"\nAfter attn_norm:")
        print(f"  Norm: {h.norm().item():.6f}")
        print(f"  First 8: {h[0, 0, :8].tolist()}")

        # QKV projection
        in_proj = layer.self_attn.in_projs[0]  # First (and only) chunk
        qkv = torch.nn.functional.linear(h, in_proj.weight)
        dim = 4096
        q, k, v = qkv[..., :dim], qkv[..., dim:2*dim], qkv[..., 2*dim:]

        print(f"\nAfter QKV projection:")
        print(f"  Q norm: {q.norm().item():.6f}, first 8: {q[0, 0, :8].tolist()}")
        print(f"  K norm: {k.norm().item():.6f}, first 8: {k[0, 0, :8].tolist()}")
        print(f"  V norm: {v.norm().item():.6f}, first 8: {v[0, 0, :8].tolist()}")

        # Reshape for attention
        n_heads = 64
        head_dim = 64
        q = q.view(1, 1, n_heads, head_dim).transpose(1, 2)  # [1, n_heads, 1, head_dim]
        k = k.view(1, 1, n_heads, head_dim).transpose(1, 2)
        v = v.view(1, 1, n_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        def rope_embedding(x, pos, n_rot, base=100000.0):
            batch, n_heads, seq_len, head_dim = x.shape
            inv_freq = 1.0 / (base ** (torch.arange(0, n_rot, 2, dtype=torch.float32) / n_rot))
            t = pos.float()
            freqs = torch.einsum('i,j->ij', t, inv_freq)  # [seq_len, n_rot/2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, n_rot]

            cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, n_rot]
            sin = emb.sin().unsqueeze(0).unsqueeze(0)

            def rotate_half(x):
                x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)

            return x * cos + rotate_half(x) * sin

        pos = torch.tensor([0])  # Position 0
        q_rope = rope_embedding(q, pos, head_dim)
        k_rope = rope_embedding(k, pos, head_dim)

        print(f"\nAfter RoPE:")
        print(f"  Q_rope norm: {q_rope.norm().item():.6f}")
        print(f"  Q_rope[0,0,0,:8]: {q_rope[0, 0, 0, :8].tolist()}")
        print(f"  K_rope[0,0,0,:8]: {k_rope[0, 0, 0, :8].tolist()}")

        # Attention score (single position = just Q.K^T / sqrt(d))
        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)  # For single position, softmax(x) = 1

        print(f"\nAttention scores:")
        print(f"  Shape: {attn.shape}")
        print(f"  Attn[0,:,0,0] (should all be 1.0 for single pos): {attn[0, :4, 0, 0].tolist()}")

        # Attention output
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, dim)

        # Output projection
        out_proj = layer.self_attn.out_projs[0]
        attn_out = torch.nn.functional.linear(attn_out, out_proj.weight)

        print(f"\nAttention output:")
        print(f"  Norm: {attn_out.norm().item():.6f}")
        print(f"  First 8: {attn_out[0, 0, :8].tolist()}")

        # Residual
        x = x + attn_out

        print(f"\nAfter attention residual:")
        print(f"  Norm: {x.norm().item():.6f}")
        print(f"  First 8: {x[0, 0, :8].tolist()}")

        # FFN norm
        norm2_weight = layer.norm2.alpha.squeeze()
        h = rms_norm(x, norm2_weight, eps=1e-8)

        print(f"\nAfter FFN norm:")
        print(f"  Norm: {h.norm().item():.6f}")
        print(f"  First 8: {h[0, 0, :8].tolist()}")

        # Gated FFN
        gating = layer.gating
        gate_up = torch.nn.functional.linear(h, gating.linear_in.weight)
        n_ff = gate_up.shape[-1] // 2
        gate = gate_up[..., :n_ff]
        up = gate_up[..., n_ff:]

        print(f"\nFFN gate (before SiLU):")
        print(f"  Shape: {gate.shape}")
        print(f"  First 8: {gate[0, 0, :8].tolist()}")
        print(f"FFN up:")
        print(f"  First 8: {up[0, 0, :8].tolist()}")

        # SiLU activation
        gate_activated = torch.nn.functional.silu(gate)
        hidden = gate_activated * up

        print(f"\nAfter SiLU * up:")
        print(f"  Norm: {hidden.norm().item():.6f}")
        print(f"  First 8: {hidden[0, 0, :8].tolist()}")

        # Down projection
        ffn_out = torch.nn.functional.linear(hidden, gating.linear_out.weight)

        print(f"\nFFN output:")
        print(f"  Norm: {ffn_out.norm().item():.6f}")
        print(f"  First 8: {ffn_out[0, 0, :8].tolist()}")

        # Final residual
        x = x + ffn_out

        print(f"\n=== AFTER LAYER 0 ===")
        print(f"Norm: {x.norm().item():.6f}")
        print(f"First 8: {x[0, 0, :8].tolist()}")
        print(f"Last 8: {x[0, 0, -8:].tolist()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
