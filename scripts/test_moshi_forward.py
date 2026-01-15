#!/usr/bin/env python3
"""
Test Moshi forward pass to get intermediate tensors for comparison.

This script runs a single forward pass through the Moshi LM and extracts
intermediate values for comparison with llama.cpp.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Test Moshi forward pass")
    parser.add_argument("--prompt", type=str, default="Hello",
                        help="Text prompt")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/moshi_comparison"),
                        help="Directory to save comparison data")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Moshi on {args.device}...")

    # Import moshi modules
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from moshi.models import loaders

    # Load Moshi LM (without streaming)
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_path, device=args.device, dtype=torch.float32)
    moshi.eval()

    # Load tokenizer
    tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, "tokenizer_spm_32k_3.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    # Tokenize prompt
    token_ids = sp.EncodeAsIds(args.prompt)
    print(f"Prompt: '{args.prompt}'")
    print(f"Token IDs: {token_ids}")

    # Get model info
    print(f"\nModel structure:")
    print(f"  Text vocab: {moshi.text_emb.weight.shape}")
    print(f"  Main transformer: {len(moshi.transformer.layers)} layers")
    print(f"  DepFormer: {len(moshi.depformer.layers)} layers")
    print(f"  DepFormer slices: {len(moshi.depformer_in)}")

    # Create input tensor
    batch_size = 1
    seq_len = len(token_ids)

    # Text tokens
    text_tokens = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    print(f"\nInput text tokens shape: {text_tokens.shape}")

    # Get text embeddings
    with torch.no_grad():
        text_emb = moshi.text_emb(text_tokens)
        print(f"Text embeddings shape: {text_emb.shape}")
        print(f"Text embeddings norm: {text_emb.norm().item():.4f}")

        # For comparison, save first few values
        text_emb_sample = text_emb[0, 0, :16].tolist()
        print(f"Text emb[0, 0, :16]: {[f'{x:.4f}' for x in text_emb_sample]}")

        # Run through transformer (non-streaming)
        # We need to call the forward method properly
        # First, let's understand the input format

        # Moshi expects: [B, K, T] where K = 1 + 16 (text + audio codebooks)
        # But for text-only, we can use the text path

        # Actually let's look at how to call forward properly
        print("\nRunning forward pass...")

        # Create full input (text + empty audio)
        n_audio_codebooks = 16  # Moshi uses 16 input codebooks
        full_input = torch.zeros(batch_size, 1 + n_audio_codebooks, seq_len, dtype=torch.long, device=args.device)
        full_input[:, 0, :] = text_tokens  # Text tokens in first position
        # Audio tokens stay at 0 (will be treated as padding/silence)

        print(f"Full input shape: {full_input.shape}")

        # Forward pass through the model
        # This returns (text_logits, audio_logits_per_codebook)
        try:
            output = moshi(full_input)

            # LMOutput has: logits, mask, text_logits, text_mask
            print(f"Output type: {type(output)}")
            if hasattr(output, 'text_logits'):
                text_logits = output.text_logits
                print(f"Text logits shape: {text_logits.shape}")  # [B, 1, T, text_card]

                # Get top predictions for last position
                last_logits = text_logits[0, 0, -1]  # [text_card]
                top_values, top_indices = torch.topk(last_logits, 10)
                print(f"\nTop 10 next token predictions:")
                for i, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
                    token = sp.IdToPiece(idx) if idx < sp.GetPieceSize() else f"[{idx}]"
                    print(f"  {i+1}. [{idx}] '{token}' (logit: {val:.4f})")

                # Audio logits
                if hasattr(output, 'logits') and output.logits is not None:
                    audio_logits = output.logits
                    print(f"Audio logits shape: {audio_logits.shape}")  # [B, K, T, card]

                # Save comparison data
                comparison_data = {
                    "prompt": args.prompt,
                    "token_ids": token_ids,
                    "text_emb_sample": text_emb_sample,
                    "text_logits_last_sample": last_logits[:16].tolist(),
                    "top_predictions": [{"token_id": int(idx), "logit": float(val)}
                                       for val, idx in zip(top_values.tolist(), top_indices.tolist())],
                }
                with open(args.save_dir / "pytorch_forward.json", "w") as f:
                    json.dump(comparison_data, f, indent=2)
                print(f"\nSaved comparison data to {args.save_dir / 'pytorch_forward.json'}")
            else:
                print(f"Output has no text_logits: {dir(output)}")

        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

            # Try alternative: manually run embedding + transformer
            print("\nTrying manual forward pass...")

            # Get combined embeddings (text + audio)
            x = text_emb  # Start with text embeddings

            # Add audio embeddings (silence = 0 tokens)
            for i, emb in enumerate(moshi.emb):
                audio_codes = full_input[:, 1 + i, :] if 1 + i < full_input.shape[1] else None
                if audio_codes is not None:
                    x = x + emb(audio_codes)

            print(f"Combined embeddings shape: {x.shape}")
            print(f"Combined embeddings norm: {x.norm().item():.4f}")

            # Run through transformer
            # Get positions
            positions = torch.arange(seq_len, device=args.device).unsqueeze(0)

            # Run transformer layers
            for il, layer in enumerate(moshi.transformer.layers):
                # Norm
                h = layer.norm1(x)

                # Self-attention
                attn_out = layer.self_attn(h, positions=positions)
                x = x + attn_out

                # FFN
                h = layer.norm2(x)
                ffn_out = layer.gating(h)
                x = x + ffn_out

                if il < 2:
                    print(f"Layer {il} output norm: {x.norm().item():.4f}")

            # Output norm
            x = moshi.out_norm(x)
            print(f"Output norm result: {x.norm().item():.4f}")

            # Text logits
            text_logits = moshi.text_linear(x)
            print(f"Text logits shape: {text_logits.shape}")

            # Get top predictions
            last_logits = text_logits[0, -1]
            top_values, top_indices = torch.topk(last_logits, 10)
            print(f"\nTop 10 next token predictions:")
            for i, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
                token = sp.IdToPiece(idx) if idx < sp.GetPieceSize() else f"[{idx}]"
                print(f"  {i+1}. [{idx}] '{token}' (logit: {val:.4f})")

            # Save comparison data
            comparison_data = {
                "prompt": args.prompt,
                "token_ids": token_ids,
                "text_emb_sample": text_emb_sample,
                "combined_emb_sample": (text_emb[0, 0, :16]).tolist(),
                "transformer_out_sample": x[0, -1, :16].tolist(),
                "text_logits_last_sample": last_logits[:16].tolist(),
                "top_predictions": [{"token_id": int(idx), "logit": float(val)}
                                   for val, idx in zip(top_values.tolist(), top_indices.tolist())],
            }
            with open(args.save_dir / "pytorch_forward.json", "w") as f:
                json.dump(comparison_data, f, indent=2)
            print(f"\nSaved comparison data to {args.save_dir / 'pytorch_forward.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
