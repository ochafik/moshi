#!/usr/bin/env python3
"""Debug transformer layer-by-layer to find mismatch with C++."""

import torch
import json
import struct
import numpy as np
import sys
sys.path.insert(0, "moshi")

from moshi.models import loaders
from huggingface_hub import hf_hub_download

def main():
    # Load Mimi model
    print("Loading Mimi model...")
    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device="cpu")
    mimi.set_num_codebooks(8)
    mimi.eval()

    # Load test tokens
    with open("/tmp/speech_tokens.json") as f:
        data = json.load(f)
    codes = torch.tensor([data["audio_tokens"]], dtype=torch.long).transpose(1, 2)
    print(f"Codes shape: {codes.shape}")  # [1, 8, T]

    # Get transformer input (after quantizer decode + upsample)
    with torch.no_grad():
        z = mimi.quantizer.decode(codes)
        print(f"After quantizer: {z.shape}")  # [1, 256, T]

        z_up = mimi.upsample(z)
        print(f"After upsample: {z_up.shape}")  # [1, 512, 2T]

        # Get transformer
        transformer = mimi.decoder_transformer

        # Input to transformer - need to transpose for transformer
        # Transformer expects [B, T, C] but conv gives [B, C, T]
        x = z_up.transpose(1, 2)  # [B, T, C] = [1, 2T, 512]
        print(f"Transformer input shape: {x.shape}")

        # Save transformer input
        x_np = x[0].numpy()
        print(f"Input @ t=0: {x_np[0, :8]}")
        print(f"Input range: [{x_np.min():.4f}, {x_np.max():.4f}]")

        with open("/tmp/transformer_input.bin", "wb") as f:
            f.write(struct.pack("ii", x_np.shape[0], x_np.shape[1]))
            f.write(x_np.astype(np.float32).tobytes())
        print(f"Saved transformer input to /tmp/transformer_input.bin")

        # Now trace through transformer manually
        # Based on moshi/modules/transformer.py StreamingTransformerLayer
        # ProjectedTransformer wraps StreamingTransformer
        inner_transformer = transformer.transformer

        # First apply input projection if present
        if transformer.input_proj is not None:
            x = transformer.input_proj(x)
            print(f"After input_proj shape: {x.shape}")
            print(f"After input_proj @ t=0: {x[0, 0, :8].numpy()}")

        layer = inner_transformer.layers[0]

        # === Layer 0 ===
        print("\n=== Layer 0 ===")
        x_orig = x.clone()

        # norm1 (pre-attention norm)
        norm1_out = layer.norm1(x)
        print(f"After norm1 @ t=0: {norm1_out[0, 0, :8].numpy()}")
        print(f"norm1 range: [{norm1_out.min():.4f}, {norm1_out.max():.4f}]")

        # Save norm1 output
        with open("/tmp/layer0_norm1.bin", "wb") as f:
            data = norm1_out[0].numpy()
            f.write(struct.pack("ii", data.shape[0], data.shape[1]))
            f.write(data.astype(np.float32).tobytes())

        # self_attn (includes QKV projection, RoPE, attention, output projection)
        attn_out = layer.self_attn(norm1_out, norm1_out, norm1_out)
        print(f"After self_attn @ t=0: {attn_out[0, 0, :8].numpy()}")
        print(f"attn_out range: [{attn_out.min():.4f}, {attn_out.max():.4f}]")

        # Save attn output
        with open("/tmp/layer0_attn.bin", "wb") as f:
            data = attn_out[0].numpy()
            f.write(struct.pack("ii", data.shape[0], data.shape[1]))
            f.write(data.astype(np.float32).tobytes())

        # layer_scale_1
        if hasattr(layer, 'layer_scale_1') and layer.layer_scale_1 is not None:
            scaled_attn = layer.layer_scale_1(attn_out)
            print(f"After layer_scale_1 @ t=0: {scaled_attn[0, 0, :8].numpy()}")
        else:
            scaled_attn = attn_out
            print("No layer_scale_1")

        # residual
        after_res1 = x_orig + scaled_attn
        print(f"After residual1 @ t=0: {after_res1[0, 0, :8].numpy()}")

        # norm2
        norm2_out = layer.norm2(after_res1)
        print(f"After norm2 @ t=0: {norm2_out[0, 0, :8].numpy()}")

        # FFN
        ffn_out = layer.linear2(layer.activation(layer.linear1(norm2_out)))
        print(f"After FFN @ t=0: {ffn_out[0, 0, :8].numpy()}")

        # layer_scale_2
        if hasattr(layer, 'layer_scale_2') and layer.layer_scale_2 is not None:
            scaled_ffn = layer.layer_scale_2(ffn_out)
            print(f"After layer_scale_2 @ t=0: {scaled_ffn[0, 0, :8].numpy()}")
        else:
            scaled_ffn = ffn_out

        # residual
        layer0_out = after_res1 + scaled_ffn
        print(f"Layer 0 output @ t=0: {layer0_out[0, 0, :8].numpy()}")

        # Save layer 0 output
        with open("/tmp/layer0_output.bin", "wb") as f:
            data = layer0_out[0].numpy()
            f.write(struct.pack("ii", data.shape[0], data.shape[1]))
            f.write(data.astype(np.float32).tobytes())

        # Full transformer output
        transformer_out = transformer(z_up)
        if isinstance(transformer_out, (list, tuple)):
            transformer_out = transformer_out[0]
        print(f"\nFull transformer output shape: {transformer_out.shape}")

        # Back to [B, C, T] for SEANet
        out_np = transformer_out[0].numpy()  # [C, T]
        print(f"Transformer output @ t=0: {out_np[:8, 0]}")
        print(f"Output range: [{out_np.min():.4f}, {out_np.max():.4f}]")

        with open("/tmp/transformer_output.bin", "wb") as f:
            f.write(struct.pack("ii", out_np.shape[0], out_np.shape[1]))
            f.write(out_np.astype(np.float32).tobytes())
        print(f"\nSaved transformer output to /tmp/transformer_output.bin")

        # Also dump some attention internals
        print("\n=== Attention Internals ===")
        self_attn = layer.self_attn

        # Get the in_proj weights
        in_proj = self_attn.in_projs[0]  # First/only in_proj
        print(f"in_proj weight shape: {in_proj.weight.shape}")  # [1536, 512] = [3*512, 512]

        # Compute QKV
        qkv = torch.nn.functional.linear(norm1_out, in_proj.weight, in_proj.bias)
        print(f"QKV shape: {qkv.shape}")  # [1, T, 1536]

        # Split into Q, K, V and reshape for heads
        n_heads = 8
        head_dim = 64
        q, k, v = qkv.chunk(3, dim=-1)  # Each [1, T, 512]
        print(f"Q shape: {q.shape}")

        # Reshape for multi-head: [B, T, H, D] -> [B, H, T, D]
        q = q.view(1, -1, n_heads, head_dim).transpose(1, 2)
        k = k.view(1, -1, n_heads, head_dim).transpose(1, 2)
        print(f"Q after reshape: {q.shape}")  # [1, 8, T, 64]

        print(f"Q @ t=0, head=0: {q[0, 0, 0, :8].numpy()}")
        print(f"K @ t=0, head=0: {k[0, 0, 0, :8].numpy()}")

if __name__ == "__main__":
    main()
