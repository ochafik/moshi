#!/usr/bin/env python3
"""
Compare embedding values between PyTorch and GGUF.
"""

import sys
from pathlib import Path
import numpy as np
import torch

try:
    import gguf
except ImportError:
    print("Error: gguf package required. Install with: pip install gguf")
    sys.exit(1)

def main():
    # Load PyTorch weights
    print("Loading PyTorch embeddings...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    moshi_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_path, device='cpu', dtype=torch.float32)

    # Get text embedding for token 11725 ("Hello")
    token_id = 11725
    pt_text_emb = moshi.text_emb.weight[token_id].detach().numpy()
    print(f"PyTorch text_emb[{token_id}] norm: {np.linalg.norm(pt_text_emb):.6f}")
    print(f"PyTorch text_emb[{token_id}][0:8]: {pt_text_emb[:8]}")

    # Load GGUF and get text embedding
    print("\nLoading GGUF embeddings...")
    gguf_path = "/tmp/moshi-7b-full.gguf"

    reader = gguf.GGUFReader(gguf_path)

    # Find token_embd tensor
    emb_tensor = None
    for tensor in reader.tensors:
        if tensor.name == "token_embd.weight":
            emb_tensor = tensor
            break

    if emb_tensor is None:
        print("Error: token_embd.weight not found in GGUF")
        return 1

    print(f"GGUF token_embd shape: {emb_tensor.shape}")
    print(f"GGUF token_embd dtype: {emb_tensor.tensor_type}")

    # Read the embedding data
    # GGUF stores embeddings as [emb_dim, vocab_size] in row-major order
    emb_dim, vocab_size = emb_tensor.shape

    # Get raw data
    emb_data = emb_tensor.data

    # Determine dtype and read data
    # tensor_type: 0=F32, 1=F16, 30=BF16
    tensor_type = emb_tensor.tensor_type
    print(f"Tensor type value: {tensor_type}")

    if hasattr(tensor_type, 'value'):
        tensor_type = tensor_type.value

    if tensor_type == 0:  # F32
        float32_data = np.frombuffer(emb_data.tobytes(), dtype=np.float32)
        float32_data = float32_data.reshape(emb_dim, vocab_size)
        gguf_emb = float32_data[:, token_id]  # Get column for token
    elif tensor_type == 1:  # F16
        float16_data = np.frombuffer(emb_data.tobytes(), dtype=np.float16)
        float16_data = float16_data.reshape(emb_dim, vocab_size)
        gguf_emb = float16_data[:, token_id].astype(np.float32)
    elif tensor_type == 30:  # BF16
        # Read as uint16 and convert
        raw_data = np.frombuffer(emb_data.tobytes(), dtype=np.uint16)
        raw_data = raw_data.reshape(emb_dim, vocab_size)
        # BF16 to float32 for specific column
        bf16_col = raw_data[:, token_id]
        float32_bits = bf16_col.astype(np.uint32) << 16
        gguf_emb = float32_bits.view(np.float32)
    else:
        print(f"Unknown tensor type: {tensor_type}")
        return 1

    print(f"GGUF text_emb[{token_id}] norm: {np.linalg.norm(gguf_emb):.6f}")
    print(f"GGUF text_emb[{token_id}][0:8]: {gguf_emb[:8]}")

    # Compare
    diff = np.abs(pt_text_emb - gguf_emb)
    print(f"\nComparison:")
    print(f"Max absolute difference: {diff.max():.8f}")
    print(f"Mean absolute difference: {diff.mean():.8f}")
    print(f"Cosine similarity: {np.dot(pt_text_emb, gguf_emb) / (np.linalg.norm(pt_text_emb) * np.linalg.norm(gguf_emb)):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
