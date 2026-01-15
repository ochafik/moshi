#!/usr/bin/env python3
"""
Test Mimi decoder with actual moshi library.

This script:
1. Loads the Mimi codec model
2. Generates sample audio codes
3. Decodes them to audio
4. Saves the output

Usage:
    PYTHONPATH=/path/to/moshi/moshi python test_mimi_decoder.py --output test.wav
"""

import argparse
import sys
from pathlib import Path

# Add moshi to path
sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))

import numpy as np
import torch


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
    """Save audio as WAV file."""
    import struct

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with open(path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVE')

        # fmt chunk
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # chunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', 1))   # num channels
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))  # byte rate
        f.write(struct.pack('<H', 2))   # block align
        f.write(struct.pack('<H', 16))  # bits per sample

        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Test Mimi decoder")
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/mimi_test.wav"),
                        help="Output WAV file path")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of codebooks")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length (frames)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    print("Loading Mimi model...")
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    # Download Mimi weights from HuggingFace
    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    print(f"Downloaded Mimi weights: {mimi_path}")

    # Load Mimi decoder
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.eval()

    print(f"Mimi loaded successfully")
    print(f"  Sample rate: {mimi.sample_rate}")
    print(f"  Frame rate: {mimi.frame_rate}")
    print(f"  Num codebooks: {mimi.num_codebooks}")

    # Generate sample codes
    print(f"\nGenerating {args.num_codebooks} x {args.seq_len} sample codes...")
    torch.manual_seed(42)

    # Use valid codebook indices (0-2047)
    codes = torch.randint(0, 2048, (1, args.num_codebooks, args.seq_len), device=args.device)
    print(f"Codes shape: {codes.shape}")
    print(f"Sample codes (first codebook, first 10): {codes[0, 0, :10].tolist()}")

    # Decode to audio
    print("\nDecoding codes to audio...")
    with torch.no_grad():
        audio = mimi.decode(codes)

    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # Convert to numpy and save
    audio_np = audio.squeeze().cpu().numpy()
    save_wav(str(args.output), audio_np, mimi.sample_rate)

    duration = len(audio_np) / mimi.sample_rate
    print(f"\nSaved audio to {args.output}")
    print(f"Duration: {duration:.2f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
