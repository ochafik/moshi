#!/usr/bin/env python3
"""
Debug Mimi decoder by exporting intermediate values at each stage.
Used to compare with C++ implementation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Debug Mimi decoder intermediate values")
    parser.add_argument("input", type=Path, help="JSON file with audio tokens")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Load audio tokens
    print(f"Loading audio tokens from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    audio_tokens = data["audio_tokens"]
    print(f"Loaded {len(audio_tokens)} frames")

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

    print(f"\nMimi frame_rate: {mimi.frame_rate}")
    print(f"Mimi encoder_frame_rate: {mimi.encoder_frame_rate}")
    print(f"Mimi sample_rate: {mimi.sample_rate}")

    # Step 1: Quantizer decode (decode_latent)
    print("\n=== Step 1: Quantizer decode (decode_latent) ===")
    with torch.no_grad():
        emb = mimi.decode_latent(codes)  # [B, D, T]
        print(f"Quantizer output shape: {emb.shape}")
        print(f"Quantizer output range: [{emb.min().item():.6f}, {emb.max().item():.6f}]")
        print(f"First 8 values at t=0: {emb[0, :8, 0].tolist()}")

    # Step 2: Upsample to encoder frame rate
    print("\n=== Step 2: _to_encoder_framerate (upsample) ===")
    with torch.no_grad():
        # This is what _to_encoder_framerate does
        frame_rate = mimi.encoder_frame_rate  # 25.0
        new_frame_rate = mimi.frame_rate  # 12.5
        if frame_rate != new_frame_rate:
            target_length = int(emb.shape[-1] * new_frame_rate / frame_rate)
            emb_up = F.interpolate(emb, size=target_length, mode="linear")
            print(f"Upsampled from {emb.shape[-1]} to {emb_up.shape[-1]} frames")
            print(f"Upsample output shape: {emb_up.shape}")
            print(f"Upsample output range: [{emb_up.min().item():.6f}, {emb_up.max().item():.6f}]")
            print(f"First 8 values at t=0: {emb_up[0, :8, 0].tolist()}")
            emb = emb_up
        else:
            print("No upsampling needed (frame rates match)")

    # Step 3: Decoder transformer
    print("\n=== Step 3: Decoder transformer ===")
    with torch.no_grad():
        if mimi.decoder_transformer is not None:
            print(f"decoder_transformer: {type(mimi.decoder_transformer)}")
            (emb_tr,) = mimi.decoder_transformer(emb)
            print(f"Transformer output shape: {emb_tr.shape}")
            print(f"Transformer output range: [{emb_tr.min().item():.6f}, {emb_tr.max().item():.6f}]")
            print(f"First 8 values at t=0: {emb_tr[0, :8, 0].tolist()}")
            emb = emb_tr
        else:
            print("No decoder_transformer")

    # Step 4: SEANet decoder - trace through layers
    print("\n=== Step 4: SEANet decoder ===")
    decoder = mimi.decoder
    with torch.no_grad():
        if hasattr(decoder, 'model'):
            model_seq = decoder.model
            x = emb
            for i, layer in enumerate(model_seq):
                layer_name = type(layer).__name__

                if i == 0:  # Init conv
                    print(f"\nLayer {i}: {layer_name} (init_conv)")
                    print(f"  Input shape: {x.shape}")
                    print(f"  Input range: [{x.min().item():.6f}, {x.max().item():.6f}]")
                    print(f"  Input first 8 values at t=0: {x[0, :8, 0].tolist()}")
                    print(f"  Input first 8 values at t=1: {x[0, :8, 1].tolist()}")

                x_out = layer(x)

                if i == 0:  # After init conv
                    print(f"  Output shape: {x_out.shape}")
                    print(f"  Output range: [{x_out.min().item():.6f}, {x_out.max().item():.6f}]")
                    print(f"  Output first 8 values at t=0: {x_out[0, :8, 0].tolist()}")

                if i == 1:  # After first ELU
                    print(f"\nLayer {i}: {layer_name} (ELU after init_conv)")
                    print(f"  Output shape: {x_out.shape}")
                    print(f"  Output range: [{x_out.min().item():.6f}, {x_out.max().item():.6f}]")
                    print(f"  Output first 8 values at t=0: {x_out[0, :8, 0].tolist()}")

                if i == 2:  # First ConvTranspose
                    print(f"\nLayer {i}: {layer_name} (upsample[0])")
                    print(f"  Output shape: {x_out.shape}")
                    print(f"  Output range: [{x_out.min().item():.6f}, {x_out.max().item():.6f}]")
                    print(f"  Output first 8 values at t=0: {x_out[0, :8, 0].tolist()}")

                x = x_out

            print(f"\n=== Final output ===")
            print(f"Shape: {x.shape}")
            print(f"Range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        else:
            out = decoder(emb)
            print(f"SEANet output shape: {out.shape}")
            print(f"SEANet output range: [{out.min().item():.6f}, {out.max().item():.6f}]")


if __name__ == "__main__":
    main()
