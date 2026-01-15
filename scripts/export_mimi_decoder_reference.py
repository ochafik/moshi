#!/usr/bin/env python3
"""
Export Mimi decoder reference data for C++ validation.

This script runs the Python Mimi decoder and exports intermediate tensors
at each stage, allowing comparison with C++ implementation.

Usage:
    python scripts/export_mimi_decoder_reference.py audio_tokens.json -o /tmp/mimi_reference

Outputs:
    /tmp/mimi_reference/
    ├── config.json                    # Mimi configuration
    ├── quantizer_output.safetensors   # After RVQ dequantization
    ├── transformer_layer_0.safetensors # After each transformer layer
    ├── ...
    ├── seanet_stage_0.safetensors     # After each SEANet upsample stage
    ├── ...
    └── audio.wav                      # Final audio output
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
    """Save audio as WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Export Mimi decoder reference data")
    parser.add_argument("input", type=Path, help="JSON file with audio tokens")
    parser.add_argument("-o", "--output", type=Path, default=Path("/tmp/mimi_reference"),
                        help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load audio tokens
    print(f"Loading audio tokens from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    audio_tokens = data["audio_tokens"]
    n_codebooks = data.get("n_codebooks", len(audio_tokens[0]) if audio_tokens else 8)
    print(f"Loaded {len(audio_tokens)} frames with {n_codebooks} codebooks")

    # Load Mimi
    print(f"Loading Mimi decoder on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders

    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.set_num_codebooks(n_codebooks)
    mimi.eval()

    # Export config
    config = {
        "sample_rate": mimi.sample_rate,
        "frame_rate": mimi.frame_rate,
        "n_codebooks": n_codebooks,
        "encoder_dim": mimi.encoder.dimension,
        "decoder_dim": mimi.decoder.dimension,
        "n_transformer_layers": len(mimi.decoder_transformer.layers) if hasattr(mimi, 'decoder_transformer') else 0,
        "quantizer_bins": mimi.quantizer.bins,
        "quantizer_dim": mimi.quantizer.dimension,
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config: {config}")

    # Convert tokens to tensor [B, K, T]
    codes = torch.tensor([audio_tokens], dtype=torch.long, device=args.device)
    codes = codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]
    codes = codes.clamp(0, 2047)
    print(f"Codes shape: {codes.shape}")

    # Step 1: Quantizer decode (embedding lookup)
    print("Step 1: Quantizer decode...")
    with torch.no_grad():
        # Get the quantized embedding
        # The quantizer's decode method returns the sum of all codebook embeddings
        quantized = mimi.quantizer.decode(codes)
        save_file({"quantized": quantized.float().cpu()},
                  str(args.output / "quantizer_output.safetensors"))
        print(f"  Quantizer output shape: {quantized.shape}")

    # Step 2: Decoder transformer
    if hasattr(mimi, 'decoder_transformer') and mimi.decoder_transformer is not None:
        print("Step 2: Decoder transformer...")
        transformer_inputs = []
        transformer_outputs = []

        # Hook to capture layer outputs
        def make_hook(layer_idx, storage):
            def hook(module, input, output):
                storage.append((layer_idx, output.detach().cpu().clone()))
            return hook

        handles = []
        for i, layer in enumerate(mimi.decoder_transformer.layers):
            h = layer.register_forward_hook(make_hook(i, transformer_outputs))
            handles.append(h)

        with torch.no_grad():
            # Run through decoder transformer
            # The transformer expects [B, T, C] format
            x = quantized.transpose(1, 2)  # [B, C, T] -> [B, T, C]
            transformer_out = mimi.decoder_transformer(x)
            transformer_out = transformer_out.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        # Remove hooks
        for h in handles:
            h.remove()

        # Save transformer layer outputs
        tensors = {}
        for layer_idx, output in transformer_outputs:
            tensors[f"layer_{layer_idx}"] = output.float()
        save_file(tensors, str(args.output / "transformer_outputs.safetensors"))
        print(f"  Saved {len(tensors)} transformer layer outputs")
    else:
        print("Step 2: No decoder transformer found, using quantized directly")
        transformer_out = quantized

    # Step 3: SEANet decoder
    print("Step 3: SEANet decoder...")
    seanet_outputs = []

    # Hook SEANet layers
    handles = []
    for i, layer in enumerate(mimi.decoder.model):
        def make_seanet_hook(idx):
            def hook(module, input, output):
                seanet_outputs.append((idx, output.detach().cpu().clone()))
            return hook
        h = layer.register_forward_hook(make_seanet_hook(i))
        handles.append(h)

    with torch.no_grad():
        audio = mimi.decoder(transformer_out)

    # Remove hooks
    for h in handles:
        h.remove()

    # Save SEANet outputs
    tensors = {}
    for idx, output in seanet_outputs:
        tensors[f"stage_{idx}"] = output.float()
    save_file(tensors, str(args.output / "seanet_outputs.safetensors"))
    print(f"  Saved {len(tensors)} SEANet stage outputs")
    print(f"  Final audio shape: {audio.shape}")

    # Save final audio
    audio_np = audio.squeeze().cpu().numpy()
    save_wav(str(args.output / "audio.wav"), audio_np, mimi.sample_rate)
    print(f"Saved audio: {len(audio_np)} samples, {len(audio_np)/mimi.sample_rate:.2f}s")

    # Also save raw audio tensor
    save_file({"audio": audio.float().cpu()}, str(args.output / "audio.safetensors"))

    print(f"\nReference data saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
