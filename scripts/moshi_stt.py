#!/usr/bin/env python3
"""
Moshi Speech-to-Text CLI tool.

Usage:
    python moshi_stt.py audio.wav
    python moshi_stt.py audio.wav -o transcript.txt
    python moshi_stt.py audio.wav --device cuda
"""

import argparse
import torch
import sys

torch._dynamo.config.suppress_errors = True


def transcribe(audio_path: str, device: str = "cpu") -> str:
    """Transcribe audio file to text using Moshi STT."""
    from moshi.models import loaders, LMGen

    # Load models
    stt_repo = "kyutai/stt-1b-en_fr"
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(stt_repo)

    mimi = checkpoint_info.get_mimi(device=device)
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.float32)
    tokenizer = checkpoint_info.get_text_tokenizer()

    # Load audio
    import sphn
    wav, sr = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    wav = torch.from_numpy(wav).to(device=device)

    # Ensure [B, C, T] shape
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    # Pad audio per STT config
    stt_config = checkpoint_info.stt_config
    pad_left = int(stt_config.get('audio_silence_prefix_seconds', 0.0) * 24000)
    pad_right = int((stt_config.get('audio_delay_seconds', 0.0) + 1.0) * 24000)
    wav = torch.nn.functional.pad(wav, (pad_left, pad_right), mode='constant')

    # Run streaming inference
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen_config = checkpoint_info.lm_gen_config
    lm_gen = LMGen(lm, **lm_gen_config)

    text_tokens = []

    with lm_gen.streaming(1), mimi.streaming(1):
        chunks = [chunk for chunk in wav.split(frame_size, dim=2) if chunk.shape[-1] == frame_size]

        for chunk in chunks:
            codes = mimi.encode(chunk)
            tokens = lm_gen.step(codes)

            if tokens is not None:
                text_token = tokens[0, 0].item()
                text_tokens.append(text_token)

    # Decode text
    valid_tokens = [t for t in text_tokens if t >= 4 and t < tokenizer.GetPieceSize()]
    text = tokenizer.Decode(valid_tokens)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Moshi Speech-to-Text")
    parser.add_argument("audio", help="Path to audio file (WAV, 24kHz preferred)")
    parser.add_argument("-o", "--output", help="Output file for transcript")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"Loading audio: {args.audio}", file=sys.stderr)
        print(f"Device: {args.device}", file=sys.stderr)

    text = transcribe(args.audio, args.device)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        if args.verbose:
            print(f"Saved to: {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
