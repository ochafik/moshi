#!/usr/bin/env python3
"""
Test Moshi forward pass with TEXT ONLY (no audio embeddings).
This should match llama.cpp output since llama.cpp doesn't add audio embeddings.
"""

import sys
from pathlib import Path
import json
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
from moshi.models import loaders


def main():
    print("Loading Moshi LM...")
    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_path, device='cpu', dtype=torch.float32)
    moshi.eval()

    # Load tokenizer
    tokenizer_path = hf_hub_download(loaders.DEFAULT_REPO, "tokenizer_spm_32k_3.model")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    # Tokenize
    prompt = "Hello"
    token_ids = sp.EncodeAsIds(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {token_ids}")

    # Get text embedding ONLY (no audio)
    with torch.no_grad():
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        x = moshi.text_emb(token_tensor)  # [1, seq_len, 4096]

        print(f"\nText embedding only:")
        print(f"  Shape: {x.shape}")
        print(f"  Norm: {x.norm().item():.4f}")

        # Run through transformer directly (handles RoPE internally)
        print(f"\nRunning through {len(moshi.transformer.layers)} transformer layers...")

        x = moshi.transformer(x)
        print(f"  After transformer: output norm = {x.norm().item():.4f}")

        # Output norm
        x = moshi.out_norm(x)
        print(f"\nAfter output norm: {x.norm().item():.4f}")

        # Text logits
        logits = moshi.text_linear(x)
        print(f"Logits shape: {logits.shape}")

        # Get top predictions
        last_logits = logits[0, -1]
        top_values, top_indices = torch.topk(last_logits, 10)

        print(f"\nTop 10 predictions (TEXT ONLY - should match llama.cpp):")
        for i, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
            token = sp.IdToPiece(idx) if idx < sp.GetPieceSize() else f"[{idx}]"
            print(f"  {i+1}. [{idx}] '{token}' (logit: {val:.4f})")

        # Save for comparison
        comparison_data = {
            "prompt": prompt,
            "token_ids": token_ids,
            "text_only": True,
            "top_predictions": [{"token_id": int(idx), "logit": float(val), "token": sp.IdToPiece(idx) if idx < sp.GetPieceSize() else f"[{idx}]"}
                               for val, idx in zip(top_values.tolist(), top_indices.tolist())],
        }
        with open("/tmp/moshi_comparison/pytorch_text_only.json", "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nSaved to /tmp/moshi_comparison/pytorch_text_only.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
