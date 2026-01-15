#!/usr/bin/env python3
"""
DSM Text-to-Speech CLI tool (Delayed Streams Modeling TTS model).

This uses the dedicated 1.6B TTS model from Kyutai, NOT the 7B dialogue model.

Usage:
    python dsm_tts.py "Hello, how are you?" -o output.wav
    python dsm_tts.py "Hello, how are you?" --voice vctk/p225_023.wav
    python dsm_tts.py "Hello" --device cuda
"""

import argparse
import struct
import numpy as np
import torch
import sys

torch._dynamo.config.suppress_errors = True


def synthesize(text: str, voice: str = "vctk/p225_023.wav",
               cfg_coef: float = 2.0, device: str = "cpu",
               n_q: int = 16) -> tuple:
    """Synthesize speech from text using DSM TTS model."""
    from moshi.models.tts import get_default_tts_model

    tts = get_default_tts_model(n_q=n_q, device=device)
    audio = tts.simple_generate(text, voice, cfg_coef=cfg_coef, show_progress=False)[0]

    return audio.cpu().numpy(), tts.mimi.sample_rate


def save_wav(audio: np.ndarray, sample_rate: int, output_path: str):
    """Save audio as WAV file."""
    audio_np = np.clip(audio, -1, 1)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    with open(output_path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="DSM Text-to-Speech (1.6B model)")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="/tmp/tts_output.wav",
                        help="Output WAV file")
    parser.add_argument("--voice", default="vctk/p225_023.wav",
                        help="Voice from kyutai/tts-voices repo")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="CFG coefficient (default: 2.0)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-q", type=int, default=16,
                        help="Number of codebooks (8-32, higher=better quality)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"Text: {args.text}", file=sys.stderr)
        print(f"Voice: {args.voice}", file=sys.stderr)
        print(f"Device: {args.device}", file=sys.stderr)
        print("Generating...", file=sys.stderr)

    audio, sample_rate = synthesize(
        args.text, args.voice, args.cfg, args.device, args.n_q
    )

    save_wav(audio, sample_rate, args.output)

    if args.verbose:
        duration = len(audio) / sample_rate
        print(f"Saved to: {args.output} ({duration:.2f}s)", file=sys.stderr)
    else:
        print(args.output)


if __name__ == "__main__":
    main()
