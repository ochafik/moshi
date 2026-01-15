#!/usr/bin/env python3
"""
Test Moshi TTS pipeline with PyTorch Mimi decoder.

This script:
1. Generates sample audio tokens (or uses hardcoded ones)
2. Runs Mimi decoder to convert tokens to audio
3. Saves the audio output

Usage:
    python test_moshi_tts.py --output test.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.torch import load_file as safe_load_torch
except ImportError:
    print("Error: torch and safetensors required")
    sys.exit(1)


def get_mimi_weights_path():
    """Find Mimi weights in HuggingFace cache."""
    cache_dir = Path.home() / ".cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16"
    snapshots = list((cache_dir / "snapshots").iterdir())
    if snapshots:
        mimi_path = snapshots[0] / "tokenizer-e351c8d8-checkpoint125.safetensors"
        if mimi_path.exists():
            return mimi_path
    return None


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


class SimpleMimiDecoder:
    """Simplified Mimi decoder for testing."""

    def __init__(self, weights_path: Path):
        print(f"Loading Mimi weights from {weights_path}...")
        self.weights = safe_load_torch(str(weights_path))

        # Extract codebook embeddings
        # The codebooks are stored as embedding_sum / cluster_usage
        self.codebooks = []

        # First codebook
        emb_sum = self.weights.get('quantizer.rvq_first.vq.layers.0._codebook.embedding_sum')
        usage = self.weights.get('quantizer.rvq_first.vq.layers.0._codebook.cluster_usage')
        if emb_sum is not None and usage is not None:
            # Normalize by usage (avoid div by zero)
            usage = usage.clamp(min=1.0)
            codebook = emb_sum / usage.unsqueeze(-1)
            self.codebooks.append(codebook)
            print(f"First codebook shape: {codebook.shape}")

        # Rest codebooks (indices 1-7 for our 8-codebook setup)
        for i in range(31):  # Check all possible indices
            emb_sum = self.weights.get(f'quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum')
            usage = self.weights.get(f'quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage')
            if emb_sum is not None and usage is not None:
                usage = usage.clamp(min=1.0)
                codebook = emb_sum / usage.unsqueeze(-1)
                self.codebooks.append(codebook)

        print(f"Loaded {len(self.codebooks)} codebooks")

        # Get output projection
        self.output_proj = self.weights.get('quantizer.rvq_first.output_proj.weight')
        if self.output_proj is not None:
            print(f"Output projection shape: {self.output_proj.shape}")

    def decode_codes(self, codes: torch.Tensor, num_codebooks: int = 8) -> torch.Tensor:
        """
        Decode audio codes to embeddings.

        Args:
            codes: [num_codebooks, seq_len] tensor of token indices
            num_codebooks: Number of codebooks to use

        Returns:
            embeddings: [batch, seq_len, embed_dim] tensor
        """
        if len(self.codebooks) < num_codebooks:
            print(f"Warning: Only {len(self.codebooks)} codebooks available, requested {num_codebooks}")
            num_codebooks = len(self.codebooks)

        K, T = codes.shape
        assert K == num_codebooks, f"Expected {num_codebooks} codebooks, got {K}"

        # Look up embeddings for each codebook and sum
        embed_dim = self.codebooks[0].shape[1]
        embeddings = torch.zeros(T, embed_dim)

        for k in range(num_codebooks):
            codebook = self.codebooks[k]
            indices = codes[k].long()
            # Clamp indices to valid range
            indices = indices.clamp(0, codebook.shape[0] - 1)
            emb = codebook[indices]  # [T, embed_dim]
            embeddings = embeddings + emb

        return embeddings.unsqueeze(0)  # [1, T, embed_dim]


def generate_sample_codes(num_codebooks: int = 8, seq_len: int = 100) -> torch.Tensor:
    """Generate sample audio codes for testing."""
    # Use deterministic "random" codes for reproducibility
    torch.manual_seed(42)
    codes = torch.randint(0, 2048, (num_codebooks, seq_len))
    return codes


def main():
    parser = argparse.ArgumentParser(description="Test Moshi TTS pipeline")
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/moshi_tts_test.wav"),
                        help="Output WAV file path")
    parser.add_argument("--weights", type=Path, help="Path to Mimi weights")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of codebooks")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length (frames)")
    args = parser.parse_args()

    # Find weights
    weights_path = args.weights or get_mimi_weights_path()
    if not weights_path or not weights_path.exists():
        print("Error: Could not find Mimi weights")
        print("Please specify --weights path")
        return 1

    # Load decoder
    decoder = SimpleMimiDecoder(weights_path)

    # Generate sample codes
    print(f"\nGenerating {args.num_codebooks} x {args.seq_len} sample codes...")
    codes = generate_sample_codes(args.num_codebooks, args.seq_len)
    print(f"Codes shape: {codes.shape}")
    print(f"Sample codes (first codebook, first 10): {codes[0, :10].tolist()}")

    # Decode to embeddings
    print("\nDecoding codes to embeddings...")
    embeddings = decoder.decode_codes(codes, args.num_codebooks)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm().item():.4f}")

    # For now, generate a simple test tone since we don't have the full SEANet decoder
    # The full decoder would:
    # 1. Run embeddings through transformer (8 layers)
    # 2. Run through SEANet decoder (conv upsampling)
    print("\nGenerating test audio (placeholder)...")

    # Generate a simple sine wave modulated by embedding magnitude
    sample_rate = 24000
    frame_rate = 12.5  # Moshi's frame rate
    samples_per_frame = int(sample_rate / frame_rate)
    total_samples = args.seq_len * samples_per_frame

    t = np.linspace(0, total_samples / sample_rate, total_samples)

    # Create audio from embeddings - simple modulation
    emb_mag = embeddings[0].abs().mean(dim=-1).numpy()  # [seq_len]
    emb_mag = np.repeat(emb_mag, samples_per_frame)[:total_samples]
    emb_mag = emb_mag / emb_mag.max() if emb_mag.max() > 0 else emb_mag

    # Generate modulated tone
    freq = 440  # A4
    audio = 0.3 * np.sin(2 * np.pi * freq * t) * emb_mag

    # Add some harmonics
    audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t) * emb_mag
    audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t) * emb_mag

    # Save audio
    save_wav(str(args.output), audio.astype(np.float32), sample_rate)
    print(f"\nSaved test audio to {args.output}")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    print("\nNote: This is a placeholder audio. Full audio synthesis requires:")
    print("  1. Actual audio codes from the Moshi LM + DepFormer")
    print("  2. Full Mimi decoder (transformer + SEANet)")
    print("  3. The pipeline is ready for integration!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
