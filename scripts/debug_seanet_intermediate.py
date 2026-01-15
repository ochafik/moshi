#!/usr/bin/env python3
"""
Debug SEANet intermediate values by hooking into the decoder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Debug SEANet intermediates")
    parser.add_argument("input", type=Path, help="JSON file with audio tokens")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Load audio tokens
    print(f"Loading audio tokens from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    audio_tokens = data["audio_tokens"]

    # Load Mimi
    print(f"\nLoading Mimi decoder on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders

    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.set_num_codebooks(8)
    mimi.eval()

    # Convert tokens to tensor [B, K, T]
    codes = torch.tensor([audio_tokens], dtype=torch.long, device=args.device)
    codes = codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]
    print(f"Audio codes shape: {codes.shape}")

    # Get the decoder
    decoder = mimi.decoder

    # Print decoder structure
    print("\nDecoder model structure:")
    for i, layer in enumerate(decoder.model):
        print(f"  [{i}] {layer.__class__.__name__}")

    # Manually run through the decoder with debug output
    print("\n=== Decoding with intermediate outputs ===")

    with torch.no_grad():
        # First go through quantizer decode
        from moshi.modules.transformer import StreamingTransformer

        # Quantizer decode
        z = mimi.quantizer.decode(codes)  # [B, D, T]
        print(f"\nQuantizer output: shape={z.shape}")
        print(f"  range: [{z.min().item():.4f}, {z.max().item():.4f}]")
        print(f"  first 4 values: {z[0, :4, 0].tolist()}")

        # Upsample 2x
        upsample = mimi.upsample
        z_up = upsample(z)
        print(f"\n2x upsample output: shape={z_up.shape}")
        print(f"  range: [{z_up.min().item():.4f}, {z_up.max().item():.4f}]")
        print(f"  first 4 values: {z_up[0, :4, 0].tolist()}")

        # Decoder transformer
        dec_transformer = mimi.decoder_transformer
        # Note: Mimi's decoder_transformer handles the permute internally
        x_out = dec_transformer(z_up)
        # Handle list output (some models return list of outputs)
        if isinstance(x_out, (list, tuple)):
            x = x_out[0]
            print(f"  Transformer returned {len(x_out)} outputs")
        else:
            x = x_out
        print(f"\nTransformer output (SEANet input): shape={x.shape}")
        print(f"  range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        # x is [B, D, T] format
        print(f"  first 4 channels @ t=0: {x[0, :4, 0].tolist()}")

        # Now run through SEANet decoder layer by layer
        print("\n=== SEANet Decoder ===")
        cur = x

        for i, layer in enumerate(decoder.model):
            layer_type = layer.__class__.__name__
            cur = layer(cur)

            if hasattr(cur, 'shape'):
                minv, maxv = cur.min().item(), cur.max().item()
                # First 4 values at t=0, channel 0
                if cur.dim() == 3:
                    first_vals = cur[0, :min(4, cur.shape[1]), 0].tolist()
                else:
                    first_vals = cur.flatten()[:4].tolist()
                print(f"[{i}] {layer_type}: shape={tuple(cur.shape)}, range=[{minv:.6f}, {maxv:.6f}]")
                print(f"     first 4 ch@t=0: {[f'{v:.6f}' for v in first_vals]}")
            else:
                print(f"[{i}] {layer_type}: output type={type(cur)}")

        audio = cur
        print(f"\nFinal audio: shape={audio.shape}")
        print(f"  range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
        print(f"  first 10 samples: {audio[0, 0, :10].tolist()}")


if __name__ == "__main__":
    main()
