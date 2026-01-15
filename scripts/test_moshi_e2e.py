#!/usr/bin/env python3
"""
End-to-end Moshi TTS test with PyTorch.

This script:
1. Loads Moshi LM and Mimi decoder
2. Runs text through the main transformer
3. Runs DepFormer to generate audio tokens
4. Decodes audio tokens with Mimi
5. Saves output audio and intermediate tensors for comparison

Usage:
    python test_moshi_e2e.py --prompt "Hello world" --output /tmp/moshi_e2e.wav
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
    parser = argparse.ArgumentParser(description="Test Moshi end-to-end TTS")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Text prompt for TTS")
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/moshi_e2e.wav"),
                        help="Output WAV file path")
    parser.add_argument("--num-tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--save-tensors", type=Path,
                        help="Save intermediate tensors to this directory")
    args = parser.parse_args()

    print(f"Loading Moshi models on {args.device}...")

    # Import moshi modules
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders, LMGen

    # Download and load Mimi
    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.set_num_codebooks(8)
    print(f"Mimi loaded: sample_rate={mimi.sample_rate}, frame_rate={mimi.frame_rate}")

    # Download and load Moshi LM
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_path, device=args.device, dtype=torch.float32)
    print(f"Moshi LM loaded: {moshi}")

    # Load tokenizer
    tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, "tokenizer_spm_32k_3.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={sp.GetPieceSize()}")

    # Tokenize prompt
    token_ids = sp.EncodeAsIds(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {[sp.IdToPiece(t) for t in token_ids]}")

    # Create LMGen for generation
    lm_gen = LMGen(moshi, temp=0.0, temp_text=0.0, top_k=1, top_k_text=1)

    # Generate tokens
    print(f"\nGenerating {args.num_tokens} tokens...")

    generated_text_tokens = []
    generated_audio_tokens = []

    # For TTS, we feed silent audio (all zeros/special token) and let the model generate
    # based on the text context we provide through the cache
    batch_size = 1
    n_codebooks = 8

    # Initialize with prompt tokens
    # Moshi expects input as [B, 1+8, T] where:
    # - [:, 0, :] = text tokens
    # - [:, 1:9, :] = audio tokens (8 codebooks)

    with torch.no_grad(), lm_gen.streaming(batch_size), mimi.streaming(batch_size):
        # First, process the prompt tokens
        prompt_len = len(token_ids)

        # Create input tensor: text tokens + silent audio
        silence_token = 0  # Use 0 or a designated silence token

        for i, text_tok in enumerate(token_ids):
            # Input: previous text token + silent audio
            input_tokens = torch.zeros(batch_size, 1 + n_codebooks, 1, dtype=torch.long, device=args.device)
            input_tokens[:, 0, 0] = text_tok
            input_tokens[:, 1:, 0] = silence_token

            # Step through the model (builds context)
            out = lm_gen.step(input_tokens)
            if out is not None:
                # out shape: [B, 1+8, 1]
                text_out = out[:, 0, 0].item()
                audio_out = out[:, 1:, 0].squeeze(0).tolist()
                print(f"  Prompt step {i+1}/{prompt_len}: text={text_out}, audio={audio_out[:2]}...")

        # Now generate new tokens
        print(f"\nGenerating continuation...")
        for i in range(args.num_tokens):
            # Use the last generated tokens as input (or initial for first step)
            if generated_text_tokens:
                input_tokens = torch.zeros(batch_size, 1 + n_codebooks, 1, dtype=torch.long, device=args.device)
                input_tokens[:, 0, 0] = generated_text_tokens[-1]
                if generated_audio_tokens:
                    for cb in range(n_codebooks):
                        input_tokens[:, 1 + cb, 0] = generated_audio_tokens[-1][cb]
            else:
                # First generation step - use last prompt token
                input_tokens = torch.zeros(batch_size, 1 + n_codebooks, 1, dtype=torch.long, device=args.device)
                input_tokens[:, 0, 0] = token_ids[-1] if token_ids else 0
                input_tokens[:, 1:, 0] = silence_token

            out = lm_gen.step(input_tokens)
            if out is not None:
                text_out = out[:, 0, 0].item()
                audio_out = out[:, 1:, 0].squeeze(0).tolist()

                generated_text_tokens.append(text_out)
                generated_audio_tokens.append(audio_out)

                # Decode text token
                text_piece = sp.IdToPiece(text_out) if text_out < sp.GetPieceSize() else f"[{text_out}]"
                print(f"  Gen {i+1}/{args.num_tokens}: text={text_out} '{text_piece}', audio[0:2]={audio_out[:2]}")

    # Decode generated audio
    if generated_audio_tokens:
        print(f"\nDecoding {len(generated_audio_tokens)} audio frames...")

        # Convert to tensor: [B, K, T]
        audio_codes = torch.tensor([generated_audio_tokens], dtype=torch.long, device=args.device)
        audio_codes = audio_codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]

        print(f"Audio codes shape: {audio_codes.shape}")

        # Decode with Mimi
        with torch.no_grad():
            audio = mimi.decode(audio_codes)

        print(f"Audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

        # Save audio
        audio_np = audio.squeeze().cpu().numpy()
        save_wav(str(args.output), audio_np, mimi.sample_rate)

        duration = len(audio_np) / mimi.sample_rate
        print(f"\nSaved audio to {args.output}")
        print(f"Duration: {duration:.2f} seconds")
    else:
        print("\nNo audio tokens generated!")

    # Print generated text
    if generated_text_tokens:
        generated_text = sp.DecodeIds([t for t in generated_text_tokens if t < sp.GetPieceSize()])
        print(f"\nGenerated text: '{generated_text}'")

    # Save intermediate tensors for comparison
    if args.save_tensors:
        args.save_tensors.mkdir(parents=True, exist_ok=True)
        with open(args.save_tensors / "generation_log.json", "w") as f:
            json.dump({
                "prompt": args.prompt,
                "prompt_tokens": token_ids,
                "generated_text_tokens": generated_text_tokens,
                "generated_audio_tokens": generated_audio_tokens,
            }, f, indent=2)
        print(f"\nSaved tensors to {args.save_tensors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
