#!/usr/bin/env python3
"""Test STT inference with the official Moshi STT model."""

import argparse
import torch
import numpy as np
from pathlib import Path

def test_stt(audio_path: str, device: str = "cpu"):
    """Run STT on an audio file and compare with expected text."""

    print("Loading models...")

    # Load Mimi for audio encoding
    from moshi.models import loaders

    # Load STT checkpoint info
    stt_repo = "kyutai/stt-1b-en_fr"
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(stt_repo)

    # Get Mimi codec
    mimi = checkpoint_info.get_mimi(device=device)
    print(f"Mimi loaded: {mimi.frame_rate} fps, {mimi.sample_rate} Hz")

    # Get STT LM model
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.float32)
    print(f"STT LM loaded: dim={lm.dim}, layers={len(lm.transformer.layers)}, n_q={lm.n_q}, dep_q={lm.dep_q}")

    # Get text tokenizer
    tokenizer = checkpoint_info.get_text_tokenizer()
    print(f"Tokenizer loaded: {tokenizer.GetPieceSize()} tokens")

    # Load audio file
    import sphn
    print(f"\nLoading audio: {audio_path}")
    wav, sr = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    wav = torch.from_numpy(wav).to(device=device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # Add batch dim
    print(f"Audio shape: {wav.shape}, sample rate: {sr}")

    # Encode with Mimi
    print("\nEncoding audio with Mimi...")
    with torch.no_grad():
        audio_tokens = mimi.encode(wav)  # [B, n_q, T]
    print(f"Audio tokens shape: {audio_tokens.shape}")

    # Save audio tokens for debugging
    tokens_np = audio_tokens[0].cpu().numpy()
    print(f"Token ranges: min={tokens_np.min()}, max={tokens_np.max()}")

    # Run STT inference
    print("\nRunning STT inference...")
    B, K, T = audio_tokens.shape

    # Build input sequence
    # STT model: audio tokens + text padding token for each frame
    pad_token = lm.existing_text_padding_id  # Usually 3
    initial_token = lm.text_initial_token_id  # Start of sequence token

    print(f"Text padding token: {pad_token}")
    print(f"Text initial token: {initial_token}")

    # For STT, we feed audio tokens and predict text
    # Input: [text, audio_0, audio_1, ..., audio_{n_q-1}]
    # The model outputs text logits for each step

    all_text_tokens = []

    with torch.no_grad():
        # Prepare input sequence for training-style forward pass
        # codes shape: [B, K, T] where K = 1 (text) + n_q (audio)
        text_codes = torch.full((B, 1, T), pad_token, dtype=torch.long, device=device)
        codes = torch.cat([text_codes, audio_tokens], dim=1)  # [B, 1+n_q, T]
        print(f"Input codes shape: {codes.shape}")

        # Use the model's forward method
        output = lm.forward(codes)
        text_logits = output.text_logits  # [B, 1, T, text_card]
        print(f"Text logits shape: {text_logits.shape}")

        # Decode text tokens
        text_tokens = text_logits.argmax(dim=-1)  # [B, 1, T]
        text_tokens = text_tokens[0, 0].cpu().tolist()
        print(f"Raw text tokens: {text_tokens[:50]}...")

    # Decode with tokenizer
    # Filter out special tokens
    filtered_tokens = [t for t in text_tokens if t < tokenizer.GetPieceSize() and t not in [0, 1, 2, 3]]
    text = tokenizer.Decode(filtered_tokens)
    print(f"\n=== STT Result ===")
    print(f"Transcription: {text}")
    print(f"==================")

    return text


def generate_test_audio(output_path: str, text: str = "Hello, how are you today?"):
    """Generate a test audio file using the TTS model."""
    print(f"Generating test audio for: '{text}'")

    from moshi.models.tts import get_default_tts_model

    tts = get_default_tts_model(n_q=16, device='cpu')
    audio = tts.simple_generate(text, 'vctk/p225_023.wav', cfg_coef=2.0, show_progress=False)[0]

    # Save as WAV
    import struct
    with open(output_path, 'wb') as f:
        audio_np = np.clip(audio.cpu().numpy(), -1, 1)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        f.write(b'WAVEfmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, 24000, 48000, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio_int16) * 2))
        f.write(audio_int16.tobytes())

    print(f"Saved test audio to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test STT inference")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--generate", action="store_true", help="Generate test audio first")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--text", default="Hello, how are you today?", help="Text for test audio generation")
    args = parser.parse_args()

    audio_path = args.audio

    if args.generate or audio_path is None:
        audio_path = "/tmp/stt_test_audio.wav"
        generate_test_audio(audio_path, args.text)

    transcription = test_stt(audio_path, args.device)

    print(f"\nExpected: {args.text}")
    print(f"Got: {transcription}")


if __name__ == "__main__":
    main()
