#!/usr/bin/env python3
"""
Decode audio tokens from llama.cpp Moshi TTS using Mimi.

This script takes the JSON output from llama.cpp's tts tool and decodes
the audio tokens to a WAV file using the Mimi neural codec.

Usage:
    python scripts/decode_mimi.py /tmp/moshi_audio_tokens.json -o /tmp/output.wav
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
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
    parser = argparse.ArgumentParser(description="Decode Mimi audio tokens")
    parser.add_argument("input", type=Path, help="JSON file with audio tokens")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output WAV file (default: input.wav)")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Load audio tokens
    print(f"Loading audio tokens from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    audio_tokens = data["audio_tokens"]
    sample_rate = data.get("sample_rate", 24000)
    frame_rate = data.get("frame_rate", 12.5)

    print(f"Loaded {len(audio_tokens)} frames ({len(audio_tokens)/frame_rate:.2f}s)")
    print(f"First frame tokens: {audio_tokens[0]}")

    # Load Mimi
    print(f"\nLoading Mimi decoder on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders

    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.set_num_codebooks(8)
    mimi.eval()
    print(f"Mimi loaded: sample_rate={mimi.sample_rate}, frame_rate={mimi.frame_rate}")

    # Convert tokens to tensor [B, K, T]
    codes = torch.tensor([audio_tokens], dtype=torch.long, device=args.device)
    codes = codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]
    print(f"Audio codes shape: {codes.shape}")

    # Clamp to valid range
    valid_mask = (codes >= 0) & (codes < 2048)
    invalid_count = (~valid_mask).sum().item()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} invalid tokens, clamping to valid range")
        codes = codes.clamp(0, 2047)

    # Decode
    print("Decoding audio...")
    with torch.no_grad():
        audio = mimi.decode(codes)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # Save
    audio_np = audio.squeeze().cpu().numpy()
    output_path = args.output or args.input.with_suffix('.wav')
    save_wav(str(output_path), audio_np, mimi.sample_rate)

    duration = len(audio_np) / mimi.sample_rate
    print(f"\nSaved audio to {output_path}")
    print(f"Duration: {duration:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
