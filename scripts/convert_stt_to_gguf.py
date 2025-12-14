#!/usr/bin/env python3
"""Convert Kyutai STT model (1B) to GGUF format."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as st
from huggingface_hub import hf_hub_download

# Add llama.cpp gguf-py to path
LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", "/Users/ochafik/github/llama.cpp")
sys.path.insert(0, os.path.join(LLAMA_CPP_PATH, "gguf-py"))

import gguf

def download_model(repo_id: str = "kyutai/stt-1b-en_fr"):
    """Download model files from HuggingFace."""
    print(f"Downloading from {repo_id}...")

    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_path = hf_hub_download(repo_id, "model.safetensors")

    # Find tokenizer file
    tokenizer_name = config.get("tokenizer_name", "tokenizer_en_fr_audio_8000.model")
    tokenizer_path = hf_hub_download(repo_id, tokenizer_name)

    return config, model_path, tokenizer_path

def convert_to_gguf(config: dict, model_path: str, tokenizer_path: str, output_path: str, ftype: str = "f16"):
    """Convert STT model to GGUF format."""
    print(f"Loading model from {model_path}...")
    tensors = st.load_file(model_path)
    print(f"Loaded {len(tensors)} tensors")

    # Determine data type
    if ftype == "f32":
        gguf_type = gguf.GGMLQuantizationType.F32
        torch_type = torch.float32
    else:
        gguf_type = gguf.GGMLQuantizationType.F16
        torch_type = torch.float16

    # Create GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "moshi-stt")

    # Add metadata
    gguf_writer.add_name("moshi-stt")

    # Model parameters
    gguf_writer.add_uint32("moshi-stt.embedding_length", config["dim"])
    gguf_writer.add_uint32("moshi-stt.block_count", config["num_layers"])
    gguf_writer.add_uint32("moshi-stt.attention.head_count", config["num_heads"])
    gguf_writer.add_uint32("moshi-stt.context_length", config["context"])
    gguf_writer.add_uint32("moshi-stt.vocab_size_text", config["text_card"])
    gguf_writer.add_uint32("moshi-stt.vocab_size_audio", config["card"])
    gguf_writer.add_uint32("moshi-stt.audio_codebooks", config["n_q"])
    gguf_writer.add_float32("moshi-stt.rope.freq_base", float(config["max_period"]))

    # FFN dimension (hidden_scale * dim)
    ffn_dim = int(config["dim"] * config.get("hidden_scale", 4.125))
    gguf_writer.add_uint32("moshi-stt.feed_forward_length", ffn_dim)

    # Delays array
    delays = config.get("delays", [0] * 33)
    gguf_writer.add_array("moshi-stt.delays", delays)

    # STT-specific config
    stt_config = config.get("stt_config", {})
    gguf_writer.add_float32("moshi-stt.audio_delay_seconds", stt_config.get("audio_delay_seconds", 0.5))
    gguf_writer.add_float32("moshi-stt.audio_silence_prefix_seconds", stt_config.get("audio_silence_prefix_seconds", 0.0))

    # Add tokenizer info
    gguf_writer.add_string("tokenizer.ggml.model", "llama")

    # Load sentencepiece tokenizer
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)

        tokens = []
        scores = []
        token_types = []

        for i in range(sp.GetPieceSize()):
            tokens.append(sp.IdToPiece(i).encode("utf-8"))
            scores.append(sp.GetScore(i))
            if sp.IsUnknown(i):
                token_types.append(2)  # unknown
            elif sp.IsControl(i):
                token_types.append(3)  # control
            elif sp.IsByte(i):
                token_types.append(6)  # byte
            else:
                token_types.append(1)  # normal

        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        gguf_writer.add_token_types(token_types)
        gguf_writer.add_bos_token_id(sp.bos_id() if sp.bos_id() >= 0 else 1)
        gguf_writer.add_eos_token_id(sp.eos_id() if sp.eos_id() >= 0 else 2)
        gguf_writer.add_unk_token_id(sp.unk_id() if sp.unk_id() >= 0 else 0)

        print(f"Added {len(tokens)} tokens from tokenizer")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")

    # Add tensors
    print("Converting tensors...")
    tensor_count = 0

    for name, tensor in tensors.items():
        # Convert to target dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32).to(torch_type)
        elif tensor.dtype != torch_type:
            tensor = tensor.to(torch_type)

        # Get numpy array
        data = tensor.numpy()

        # Map tensor name to GGUF name
        gguf_name = name

        # Add tensor
        gguf_writer.add_tensor(gguf_name, data)
        tensor_count += 1

        if tensor_count % 50 == 0:
            print(f"  Converted {tensor_count} tensors...")

    print(f"Total tensors: {tensor_count}")

    # Write file
    print(f"Writing GGUF to {output_path}...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Convert Kyutai STT model to GGUF")
    parser.add_argument("--repo", default="kyutai/stt-1b-en_fr", help="HuggingFace repo ID")
    parser.add_argument("--output", "-o", default="/tmp/moshi-stt.gguf", help="Output GGUF path")
    parser.add_argument("--ftype", default="f16", choices=["f16", "f32"], help="Output format")
    args = parser.parse_args()

    config, model_path, tokenizer_path = download_model(args.repo)
    convert_to_gguf(config, model_path, tokenizer_path, args.output, args.ftype)

if __name__ == "__main__":
    main()
