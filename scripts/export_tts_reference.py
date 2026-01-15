#!/usr/bin/env python3
"""
Export TTS reference data for C++ validation.

This script runs the Python TTS pipeline and exports intermediate tensors
at each stage, allowing comparison with C++ implementation.

Usage:
    python scripts/export_tts_reference.py "Hello world" -o /tmp/tts_reference

Outputs:
    /tmp/tts_reference/
    ├── config.json           # Model configuration
    ├── text_tokens.json      # Input text tokens
    ├── audio_tokens.json     # Generated audio tokens (for Mimi decoder test)
    ├── transformer_layer_0.safetensors  # Hidden states after each layer
    ├── ...
    ├── depformer_output.safetensors     # DepFormer outputs
    └── audio.wav             # Final audio output
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


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
    parser = argparse.ArgumentParser(description="Export TTS reference data")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", type=Path, default=Path("/tmp/tts_reference"),
                        help="Output directory")
    parser.add_argument("--voice", default="vctk/p225_023.wav",
                        help="Voice from kyutai/tts-voices repo")
    parser.add_argument("--n-q", type=int, default=8,
                        help="Number of codebooks (8-32)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum generation steps")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading TTS model on {args.device}...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))

    from moshi.models.tts import get_default_tts_model, TTSModel

    tts = get_default_tts_model(n_q=args.n_q, device=args.device)
    tts.max_gen_length = args.max_steps

    # Export model config
    config = {
        "dim": tts.lm.dim,
        "n_layers": len(list(tts.lm.transformer.layers)),
        "n_heads": tts.lm.transformer.layers[0].self_attn.num_heads,
        "n_q": tts.lm.n_q,
        "dep_q": tts.lm.dep_q,
        "card": tts.lm.card,
        "text_card": tts.lm.text_card,
        "delays": tts.lm.delays,
        "depformer_dim": tts.lm.depformer.d_model if tts.lm.depformer else 0,
        "depformer_layers": len(list(tts.lm.depformer.layers)) if tts.lm.depformer else 0,
        "sample_rate": tts.mimi.sample_rate,
        "frame_rate": tts.mimi.frame_rate,
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config}")

    # Tokenize text
    text_tokens = tts.tokenizer.encode(args.text)
    with open(args.output / "text_tokens.json", "w") as f:
        json.dump({
            "text": args.text,
            "tokens": text_tokens,
            "pieces": [tts.tokenizer.id_to_piece(t) for t in text_tokens]
        }, f, indent=2)
    print(f"Text tokens: {text_tokens}")

    # Prepare entries and attributes
    entries = tts.prepare_script([args.text], padding_between=1)
    voice_path = tts.get_voice_path(args.voice)
    attrs = tts.make_condition_attributes([voice_path], cfg_coef=2.0)

    # Storage for intermediate outputs
    collected_frames = []
    transformer_outputs = []
    depformer_outputs = []

    # Hook to capture transformer outputs
    original_forward = tts.lm.transformer.forward
    def hooked_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        transformer_outputs.append(result.detach().cpu().clone())
        return result
    tts.lm.transformer.forward = hooked_forward

    # Hook to capture depformer outputs
    if tts.lm.depformer:
        original_dep_forward = tts.lm.depformer.forward
        def hooked_dep_forward(*args, **kwargs):
            result = original_dep_forward(*args, **kwargs)
            depformer_outputs.append(result.detach().cpu().clone())
            return result
        tts.lm.depformer.forward = hooked_dep_forward

    print(f"Running TTS generation (max {args.max_steps} steps)...")

    # Generate with frame callback
    def on_frame(frame):
        collected_frames.append(frame.detach().cpu().clone())

    tts_result = tts.generate([entries], [attrs], on_frame=on_frame)

    print(f"Generated {len(collected_frames)} frames")

    # Extract audio tokens from frames [B, 1+Q, 1]
    audio_tokens = []
    for frame in collected_frames:
        if (frame != -1).all():  # Skip padding frames
            tokens = frame[0, 1:, 0].tolist()  # Skip text token (index 0)
            audio_tokens.append([int(t) for t in tokens])

    # Save audio tokens
    with open(args.output / "audio_tokens.json", "w") as f:
        json.dump({
            "n_frames": len(audio_tokens),
            "n_codebooks": args.n_q,
            "audio_tokens": audio_tokens,
            "sample_rate": tts.mimi.sample_rate,
            "frame_rate": tts.mimi.frame_rate,
        }, f, indent=2)
    print(f"Saved {len(audio_tokens)} audio token frames")

    # Save transformer outputs (first few steps)
    if transformer_outputs:
        tensors = {}
        for i, out in enumerate(transformer_outputs[:10]):
            tensors[f"step_{i}"] = out.float()
        save_file(tensors, str(args.output / "transformer_outputs.safetensors"))
        print(f"Saved {len(tensors)} transformer output snapshots")

    # Save depformer outputs
    if depformer_outputs:
        tensors = {}
        for i, out in enumerate(depformer_outputs[:10]):
            tensors[f"step_{i}"] = out.float()
        save_file(tensors, str(args.output / "depformer_outputs.safetensors"))
        print(f"Saved {len(tensors)} depformer output snapshots")

    # Decode audio with Mimi
    if audio_tokens:
        print("Decoding audio with Mimi...")
        codes = torch.tensor([audio_tokens], dtype=torch.long, device=args.device)
        codes = codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]
        codes = codes.clamp(0, 2047)

        with torch.no_grad():
            audio = tts.mimi.decode(codes)

        audio_np = audio.squeeze().cpu().numpy()
        save_wav(str(args.output / "audio.wav"), audio_np, tts.mimi.sample_rate)
        print(f"Saved audio: {len(audio_np)} samples, {len(audio_np)/tts.mimi.sample_rate:.2f}s")

    # Save summary
    summary = {
        "text": args.text,
        "voice": args.voice,
        "n_q": args.n_q,
        "n_frames": len(audio_tokens),
        "end_step": tts_result.end_steps[0],
        "transcript": tts_result.all_transcripts[0],
    }
    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReference data saved to {args.output}")
    print(f"Files: config.json, text_tokens.json, audio_tokens.json, audio.wav")

    return 0


if __name__ == "__main__":
    sys.exit(main())
