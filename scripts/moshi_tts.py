#!/usr/bin/env python3
"""
Moshi Text-to-Speech Pipeline

This script demonstrates TTS using Moshi with text forcing:
1. Tokenize input text
2. Run Moshi LM in streaming mode with silent audio input
3. Force text tokens by overwriting the cache (teacher forcing)
4. Collect generated audio tokens from DepFormer
5. Decode audio tokens with Mimi

The Moshi model is designed for dialogue:
- Input: 8 user audio codebooks (microphone stream)
- Output: text + 8 audio codebooks (speaker stream)

For TTS, we:
- Provide silent audio as "microphone" input
- Override the model's text predictions with our desired text
- Collect the generated audio

Usage:
    python moshi_tts.py --text "Hello world" --output /tmp/tts_output.wav
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Moshi TTS")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/moshi_tts.wav"))
    parser.add_argument("--num-gen-tokens", type=int, default=100,
                        help="Number of tokens to generate after prompt")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading Moshi models on {args.device}...")

    # Import moshi
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders, LMGen

    # Load Mimi
    mimi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_path, device=args.device)
    mimi.set_num_codebooks(8)
    mimi.eval()
    print(f"Mimi loaded: sample_rate={mimi.sample_rate}, frame_rate={mimi.frame_rate}")

    # Load Moshi LM
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    dtype = torch.bfloat16 if args.device != 'cpu' else torch.float32
    moshi_lm = loaders.get_moshi_lm(moshi_path, device=args.device, dtype=dtype)
    moshi_lm.eval()
    print(f"Moshi LM loaded")
    print(f"  n_q (input audio codebooks): {moshi_lm.n_q}")
    print(f"  dep_q (output audio codebooks): {moshi_lm.dep_q}")
    print(f"  num_codebooks (total): {moshi_lm.num_codebooks}")

    # Load tokenizer
    tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, "tokenizer_spm_32k_3.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={sp.GetPieceSize()}")

    # Tokenize text
    text_tokens = sp.EncodeAsIds(args.text)
    print(f"\nText: '{args.text}'")
    print(f"Tokens: {text_tokens}")
    print(f"  -> {[sp.IdToPiece(t) for t in text_tokens]}")

    # Create LMGen for generation
    lm_gen = LMGen(moshi_lm, temp=args.temp, temp_text=args.temp)

    # Calculate expected input size:
    # The model expects: num_codebooks - dep_q - 1 = 17 - 8 - 1 = 8 user audio tokens
    # These represent the "user's microphone" audio stream (codebooks 8-15)
    n_user_audio = moshi_lm.num_codebooks - moshi_lm.dep_q - 1  # 8
    batch_size = 1

    # Use the initial token for audio (2048 is the special initial token)
    audio_initial_token = moshi_lm.initial_token_id  # 2048
    text_initial_token = moshi_lm.text_initial_token_id  # 32000
    text_padding_token = moshi_lm.text_padding_token_id  # 3

    print(f"\nModel configuration:")
    print(f"  User audio input channels: {n_user_audio}")
    print(f"  Audio initial token: {audio_initial_token}")
    print(f"  Text initial token: {text_initial_token}")
    print(f"  Text padding token: {text_padding_token}")

    generated_audio_tokens = []
    generated_text_tokens = []

    # Need to step through the delay period first
    max_delay = lm_gen.max_delay
    print(f"  Max delay: {max_delay} steps")

    print(f"\nGenerating speech with text forcing...")

    with torch.no_grad():
        # Use streaming mode
        with lm_gen.streaming(batch_size) as gen_state, mimi.streaming(batch_size):
            step_count = 0
            total_steps = len(text_tokens) + args.num_gen_tokens + max_delay

            # Get the streaming state for text forcing
            streaming_state = lm_gen._streaming_state

            # Combined loop for prompt + generation
            for i in range(total_steps):
                # Create input: [B, n_user_audio, 1] - just the user audio stream
                # This is what the model expects: silent audio from "microphone"
                input_tokens = torch.full(
                    (batch_size, n_user_audio, 1),
                    audio_initial_token,
                    dtype=torch.long,
                    device=args.device
                )

                # Before stepping, force our text token into the cache
                # The cache has shape [B, num_codebooks, max_delay+2]
                # Position 0 is text, positions 1-8 are output audio, positions 9-16 are user audio
                if streaming_state is not None and i > 0:
                    # Determine the text token to force
                    if i <= len(text_tokens):
                        # Force prompt token (offset by 1 since step 0 used initial)
                        forced_text = text_tokens[i - 1] if i - 1 < len(text_tokens) else text_padding_token
                    else:
                        # After prompt, use padding to let model generate freely
                        # Or we could use the last generated token
                        forced_text = text_padding_token

                    # Write to cache at position 0 (text)
                    CT = streaming_state.cache.shape[2]
                    pos = (streaming_state.offsets % CT)[:, None, None]
                    streaming_state.cache[:, :1, :].scatter_(-1, pos,
                        torch.full((batch_size, 1, 1), forced_text,
                                   dtype=torch.long, device=args.device))

                out = lm_gen.step(input_tokens)
                step_count += 1

                if out is not None:
                    # out shape: [B, 1+dep_q, 1] = [B, 9, 1] - text + 8 audio codebooks
                    text_out = out[0, 0, 0].item()
                    audio_out = out[0, 1:, 0].tolist()  # 8 audio tokens
                    generated_text_tokens.append(text_out)
                    generated_audio_tokens.append(audio_out)

                    text_piece = sp.IdToPiece(text_out) if 0 <= text_out < sp.GetPieceSize() else f"[{text_out}]"
                    if len(generated_audio_tokens) <= 5 or len(generated_audio_tokens) % 10 == 0:
                        print(f"  Step {step_count}: text='{text_piece}', audio[0]={audio_out[0]}")

            print(f"  Total steps: {step_count}, generated: {len(generated_audio_tokens)} frames")

    # Decode audio
    if generated_audio_tokens:
        print(f"\nDecoding {len(generated_audio_tokens)} audio frames...")

        # Convert to tensor: [B, K, T] where K=8 codebooks
        audio_codes = torch.tensor([generated_audio_tokens], dtype=torch.long, device=args.device)
        audio_codes = audio_codes.transpose(1, 2)  # [B, T, K] -> [B, K, T]
        print(f"Audio codes shape: {audio_codes.shape}")

        # Check for invalid tokens
        valid_mask = (audio_codes >= 0) & (audio_codes < 2048)
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid tokens, clamping to valid range")
            audio_codes = audio_codes.clamp(0, 2047)

        with torch.no_grad():
            audio = mimi.decode(audio_codes)

        print(f"Audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

        # Save
        audio_np = audio.squeeze().cpu().numpy()
        save_wav(str(args.output), audio_np, mimi.sample_rate)

        duration = len(audio_np) / mimi.sample_rate
        print(f"\nSaved audio to {args.output}")
        print(f"Duration: {duration:.2f} seconds")
    else:
        print("\nNo audio tokens generated!")

    # Print generated text
    valid_text_tokens = [t for t in generated_text_tokens if 0 <= t < sp.GetPieceSize()]
    if valid_text_tokens:
        generated_text = sp.DecodeIds(valid_text_tokens)
        print(f"\nGenerated text: '{generated_text}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
