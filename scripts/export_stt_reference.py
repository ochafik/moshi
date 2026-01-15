#!/usr/bin/env python3
"""Export STT reference data for C++ verification."""

import torch
import numpy as np
import json
torch._dynamo.config.suppress_errors = True

from moshi.models import loaders, LMGen

def export_reference(audio_path: str, output_dir: str = "/tmp/stt_reference"):
    """Export reference tensors for C++ verification."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    stt_repo = "kyutai/stt-1b-en_fr"
    print(f"Loading STT model from {stt_repo}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(stt_repo)

    mimi = checkpoint_info.get_mimi(device='cpu')
    lm = checkpoint_info.get_moshi(device='cpu', dtype=torch.float32)
    tokenizer = checkpoint_info.get_text_tokenizer()

    print(f"Model config: dim={lm.dim}, layers={len(lm.transformer.layers)}, n_q={lm.n_q}")

    # Load and process audio
    import sphn
    wav, sr = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    wav = torch.from_numpy(wav).to(device='cpu')

    # Ensure [B, C, T] shape
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    # Pad audio
    stt_config = checkpoint_info.stt_config
    pad_left = int(stt_config.get('audio_silence_prefix_seconds', 0.0) * 24000)
    pad_right = int((stt_config.get('audio_delay_seconds', 0.0) + 1.0) * 24000)
    wav = torch.nn.functional.pad(wav, (pad_left, pad_right), mode='constant')

    # Encode full audio first to get reference tokens
    print("Encoding audio with Mimi...")
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    # Encode in one shot for reference
    full_codes = mimi.encode(wav)
    print(f"Full audio tokens shape: {full_codes.shape}")

    # Save audio tokens as JSON
    audio_tokens = full_codes[0].cpu().numpy().tolist()
    with open(f"{output_dir}/audio_tokens.json", "w") as f:
        json.dump({"audio_tokens": audio_tokens, "n_q": lm.n_q, "n_frames": len(audio_tokens[0])}, f)
    print(f"Saved audio tokens to {output_dir}/audio_tokens.json")

    # Run streaming inference
    lm_gen_config = checkpoint_info.lm_gen_config
    lm_gen = LMGen(lm, **lm_gen_config)

    text_tokens = []
    all_logits = []

    with lm_gen.streaming(1), mimi.streaming(1):
        chunks = [chunk for chunk in wav.split(frame_size, dim=2) if chunk.shape[-1] == frame_size]
        print(f"Processing {len(chunks)} frames...")

        for i, chunk in enumerate(chunks):
            codes = mimi.encode(chunk)
            tokens = lm_gen.step(codes)

            if tokens is not None:
                text_token = tokens[0, 0].item()
                text_tokens.append(text_token)

    # Decode text
    valid_tokens = [t for t in text_tokens if t >= 4 and t < tokenizer.GetPieceSize()]
    text = tokenizer.Decode(valid_tokens)

    print(f"\n=== STT Result ===")
    print(f"Transcription: {text}")
    print(f"==================")

    # Save reference data
    reference = {
        "audio_path": audio_path,
        "text_tokens": text_tokens,
        "transcription": text,
        "model_config": {
            "dim": lm.dim,
            "n_layers": len(lm.transformer.layers),
            "n_heads": lm.transformer.layers[0].self_attn.num_heads,
            "n_q": lm.n_q,
            "text_card": lm.text_card,
            "card": lm.card,
            "frame_rate": mimi.frame_rate,
            "sample_rate": mimi.sample_rate,
        }
    }

    with open(f"{output_dir}/reference.json", "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Saved reference to {output_dir}/reference.json")

    # Export embedding weights for verification
    print("Exporting key tensors...")

    # Text embedding
    text_emb = lm.text_emb.weight.data.cpu().numpy()
    np.save(f"{output_dir}/text_emb.npy", text_emb)
    print(f"  text_emb: {text_emb.shape}")

    # Audio embeddings (first few)
    for i in range(min(4, lm.n_q)):
        emb = lm.emb[i].weight.data.cpu().numpy()
        np.save(f"{output_dir}/emb_{i}.npy", emb)
        print(f"  emb_{i}: {emb.shape}")

    # Text linear output
    text_linear_w = lm.text_linear.weight.data.cpu().numpy()
    text_linear_b = lm.text_linear.bias.data.cpu().numpy() if lm.text_linear.bias is not None else None
    np.save(f"{output_dir}/text_linear_w.npy", text_linear_w)
    if text_linear_b is not None:
        np.save(f"{output_dir}/text_linear_b.npy", text_linear_b)
    print(f"  text_linear: {text_linear_w.shape}")

    print(f"\nReference data saved to {output_dir}/")
    return text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="/tmp/stt_long_test.wav")
    parser.add_argument("--output", default="/tmp/stt_reference")
    args = parser.parse_args()

    export_reference(args.audio, args.output)
