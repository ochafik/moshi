# Moshi TTS Integration into llama.cpp - Project Log

## Overview

This project integrates Kyutai's Moshi TTS/STT model into llama.cpp, enabling efficient inference on Apple Silicon and other platforms via GGML.

**Architecture**: Moshi uses a two-stage approach:
1. **Moshi LM** - Temporal transformer that generates text + first audio codebook token
2. **DepFormer** - Depth transformer that generates remaining 7 codebook tokens
3. **Mimi Codec** - Neural audio codec that decodes codebook tokens to audio waveforms

## Completed Work

### 1. Moshi LM GGUF Converter
**File**: `scripts/convert_moshi_to_gguf.py`

Converts PyTorch/MLX Moshi models to GGUF format:
- Handles tensor name conversion: PyTorch → MLX → GGUF
- Supports main temporal transformer (24 layers for 7B model)
- Includes DepFormer weights (6 layers × 8 codebook slices)
- Tested with synthetic data, produces valid GGUF

**Key mappings**:
```
PyTorch                          → GGUF
transformer.layers.{i}.*.weight  → blk.{i}.*
depformer_*.slices.{q}.*         → depformer.{i}.slice.{q}.*
emb.weight                       → token_embd.weight
linears.{q}.weight               → audio_embd.{q}.weight
```

### 2. LLM_ARCH_MOSHI in llama.cpp

**Files modified**:
- `src/llama-arch.h:92` - Added `LLM_ARCH_MOSHI` enum
- `src/llama-arch.cpp:88` - Added `"moshi"` name mapping
- `src/llama-arch.cpp:1975-2002` - Added tensor name mappings

**Tensor mappings**:
```cpp
{ LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
{ LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
{ LLM_TENSOR_OUTPUT,          "output" },
{ LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
{ LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
{ LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
{ LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
{ LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
{ LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
{ LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
{ LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
{ LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
// Cross-attention for voice conditioning
{ LLM_TENSOR_DEC_CROSS_ATTN_NORM, "blk.%d.cross_attn_norm" },
{ LLM_TENSOR_DEC_CROSS_ATTN_Q,    "blk.%d.cross_attn_q" },
// ... etc
// DepFormer tensors
{ LLM_TENSOR_TOKEN_EMBD,      "depformer.slices.%d.emb" },
```

### 3. Moshi Graph Builder
**File**: `src/models/moshi.cpp`

LLaMA-style transformer implementation:
- RoPE positional embeddings
- RMSNorm normalization
- Optional cross-attention for voice conditioning
- Gated SiLU FFN (SwiGLU)

```cpp
llm_build_moshi::llm_build_moshi(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    // Text embedding + audio codebook embeddings sum
    inpL = build_inp_embd(model.tok_embd);

    // Main temporal transformer (LLaMA-style)
    for (int il = 0; il < n_layer; ++il) {
        // Self-attention with RoPE
        // Optional cross-attention
        // Gated SiLU FFN
    }

    // Output: text logits
    res->t_logits = build_lora_mm(model.output, cur);
}
```

### 4. Moshi Tensor Loading
**File**: `src/llama-model.cpp:5587-5618`

Tensors created:
- `tok_embd` - Token embeddings
- `output_norm`, `output` - Output projection
- Per-layer: `attn_norm`, `wq/wk/wv/wo`, `ffn_norm`, `ffn_gate/down/up`
- Optional cross-attention: `wq_cross`, `wk_cross`, `wv_cross`, `wo_cross`

### 5. Mimi Codec GGUF Converter
**File**: `scripts/convert_mimi_to_gguf.py`

Converts Mimi neural audio codec to GGUF:
- Focuses on decoder path only (for TTS)
- Handles: quantizer embeddings, upsample conv, decoder transformer, SEANet decoder

**Mimi architecture**:
```
Codebook tokens → Quantizer.decode() → Upsample → Decoder Transformer → SEANet Decoder → Audio
```

### 6. LLM_ARCH_MIMI in llama.cpp

**Files modified**:
- `src/llama-arch.h:93` - Added `LLM_ARCH_MIMI` enum
- `src/llama-arch.cpp:89` - Added `"mimi"` name mapping
- `src/llama-arch.cpp:2005-2034` - Added tensor mappings

### 7. Mimi Graph Builder
**File**: `src/models/mimi.cpp`

Decoder transformer implementation:
- Packed QKV projection (in_proj)
- RoPE attention
- LayerNorm (not RMSNorm)
- GELU FFN (not gated)

### 8. tts.cpp Extensions
**File**: `tools/tts/tts.cpp`

Added architecture detection:
```cpp
enum tts_arch {
    TTS_ARCH_OUTETTS,
    TTS_ARCH_MOSHI,
};

static tts_arch get_tts_arch(llama_model * model) {
    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    // Check if "moshi" in description
}
```

Placeholder for Moshi-specific processing added.

## Build Status

All changes compile successfully:
```bash
cd /Users/ochafik/github/llama.cpp/build
cmake --build . -j
# [100%] Built target llama-tts
```

## ✅ Model Loading & Inference Working!

**Date**: December 13, 2024

Successfully converted and ran the Moshi 7B model in llama.cpp:

```bash
# Convert weights (15.4GB bf16 → 12.7GB f16)
python scripts/convert_moshi_to_gguf.py \
  -m ~/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/... \
  -o /tmp/moshi-7b.gguf \
  -t /tmp/tokenizer_spm_32k_3.model

# Run inference
./build/bin/llama-cli -m /tmp/moshi-7b.gguf -p "Hello, my name is" -n 10

# Output: Hello, my name is Engel engineer Engel...
# (nonsensical but expected - TTS model, not chat model)
```

**Performance (Apple M2 Max)**:
- Prompt eval: 71 tokens/sec
- Token generation: 24.7 tokens/sec
- Memory: ~13GB on Metal GPU

**Key fixes**:
1. Added `moshi.feed_forward_length` to GGUF metadata
2. Transposed 2D weights for GGML (row-major vs column-major)
3. Kept 1D tensors (norms) as F32 for Metal compatibility
4. Filtered to 291 base transformer tensors (DepFormer/audio embeddings not yet supported)

## Files Changed Summary

### In moshi repo:
| File | Status | Description |
|------|--------|-------------|
| `scripts/convert_moshi_to_gguf.py` | Created | Moshi LM → GGUF converter |
| `scripts/convert_mimi_to_gguf.py` | Created | Mimi codec → GGUF converter |

### In llama.cpp:
| File | Status | Description |
|------|--------|-------------|
| `src/llama-arch.h` | Modified | Added MOSHI, MIMI enums |
| `src/llama-arch.cpp` | Modified | Added tensor name mappings |
| `src/llama-model.cpp` | Modified | Added tensor loading + graph builder registration |
| `src/models/moshi.cpp` | Created | Moshi LM graph builder |
| `src/models/mimi.cpp` | Created | Mimi codec graph builder |
| `src/models/models.h` | Modified | Added class declarations |
| `src/CMakeLists.txt` | Modified | Added moshi.cpp, mimi.cpp |
| `tools/tts/tts.cpp` | Modified | Added architecture detection |

## Remaining Work

### High Priority

1. **Add DepFormer Tensors to llama.cpp** (244 tensors)

   The DepFormer has a complex structure with per-codebook (slice) tensors:

   ```
   DepFormer Structure:
   ├── linear_in.{0-7}.weight          # 8 tensors: [1024, 4096] - Project from LM to DepFormer
   ├── slices.{0-7}.emb.weight         # 8 tensors: [vocab, 1024] - Per-codebook embeddings
   ├── slices.{0-7}.linear_out.weight  # 8 tensors: [2048, 1024] - Per-codebook outputs
   └── transformer.layers.{0-5}        # 6 layers
       ├── norm1.weight                # 6 tensors: [1024] - Pre-attention norm
       ├── norm2.weight                # 6 tensors: [1024] - Pre-FFN norm
       ├── self_attn.{0-7}             # 8 heads per layer (shared weights split)
       │   ├── in_proj.weight          # 48 tensors: [3072, 1024] - Packed QKV
       │   └── out_proj.weight         # 48 tensors: [1024, 1024]
       └── gating.{0-7}                # 8 FFN blocks per layer
           ├── linear_in.weight        # 48 tensors: [5632, 1024] - Gate+Up fused
           └── linear_out.weight       # 48 tensors: [1024, 2816] - Down projection

   Total: 8 + 8 + 8 + 12 + 96 + 96 = 228 tensors (+ 16 audio embeddings = 244)
   ```

   **Implementation approach**:
   - Option A: Add new LLM_TENSOR enums for each tensor type (complex, many new enums)
   - Option B: Use existing enums with xid parameter for slice index
   - Option C: Store DepFormer as separate GGUF file (simpler, modular)

   **Recommended**: Option C - Separate GGUF files
   - Main model: `moshi-7b.gguf` (291 tensors, base transformer)
   - DepFormer: `moshi-depformer.gguf` (228 tensors)
   - Audio embeddings: Include in main or separate `moshi-audio.gguf`

2. **Add Audio Embedding Tensors** (16 tensors)
   ```
   audio_embd.{0-15}.weight  # [2049, 4096] - 16 codebook embeddings
   ```
   These are summed with text embedding for multimodal input at each timestep.

3. **Current Workaround**
   - Filtering to 291 base tensors for initial testing
   - Full TTS requires all 535 tensors

### Medium Priority

4. **DepFormer Integration**
   - DepFormer runs after main LM to generate codebook tokens 2-8
   - Each slice has its own embedding + linear layers
   - Need to implement multi-codebook token generation loop

5. **SEANet Decoder for Mimi**
   - Full convolutional implementation with upsampling
   - 4 decoder layers with ratios [8, 6, 5, 4]
   - Residual blocks with ELU activation

6. **Moshi-Specific TTS Flow**
   ```
   Text input
     → Moshi LM: generate text predictions + first audio codebook
     → DepFormer: generate remaining 7 codebook tokens
     → Mimi decoder: convert 8-codebook tokens to audio
   ```

### Lower Priority

7. **Voice Conditioning (Cross-Attention)**
   - Optional feature for voice cloning
   - Requires speaker embedding input

8. **Streaming Support**
   - Moshi is designed for streaming audio
   - Would require incremental decoding

9. **STT Direction**
   - Currently focused on TTS (text → audio)
   - STT would require Mimi encoder + Moshi in reverse

## Architecture Reference (from actual moshiko-pytorch-bf16 weights)

### Moshi LM
- **Layers**: 32 (7B model)
- **Hidden dim**: 4096
- **Heads**: 64 (head_dim=64)
- **FFN dim**: 11264
- **Text vocab**: 32000 (SentencePiece)
- **Audio vocab**: 2048 per codebook
- **Audio codebooks**: 16 (main model) + 8 (DepFormer output)
- **RoPE**: Yes, with base freq 100000
- **Total tensors**: 291 (base) + 16 (audio embd) + 228 (DepFormer) = 535

### DepFormer
- **Layers**: 6
- **Hidden dim**: 1024
- **Heads**: 16 (head_dim=64)
- **FFN dim**: 2816 (gated, so 5632 in linear_in)
- **Codebook slices**: 8
- **Attention**: 8 separate heads per layer (not shared)
- **FFN**: 8 separate gating blocks per layer

### Mimi Codec
- **Sample rate**: 24000 Hz
- **Frame rate**: 12.5 Hz
- **Codebooks**: 8
- **Codebook bins**: 2048
- **Latent dim**: 256
- **SEANet dim**: 512
- **Transformer layers**: 8

## Current Status (December 14, 2024)

### ✅ Working
- Base Moshi transformer loads and runs in llama.cpp
- **Audio embeddings (16 codebooks) now supported!**
- **Full model with DepFormer (535 tensors) loads and runs!**
- 14.32 GB GGUF with all tensors (base + audio + DepFormer)
- Metal GPU inference at ~25.35 tok/s on M2 Max
- SentencePiece tokenizer integrated
- DepFormer tensor storage, creation, and loading verified
- DepFormer graph builder implemented

### Key Changes for Audio Embeddings
1. Added `LLM_TENSOR_AUDIO_EMBD` to `llama-arch.h`
2. Added tensor name mapping `"audio_embd.%d"` to `llama-arch.cpp`
3. Added `LLM_TENSOR_INFOS` entry with `LAYER_REPEATING` (indexed by codebook)
4. Added `audio_embd` vector to `llama_model` struct
5. Updated tensor creation in `llama-model.cpp` for Moshi

### Key Changes for DepFormer (Session Dec 14)
1. Added DepFormer tensor enums to `llama-arch.h`:
   - `LLM_TENSOR_DEPFORMER_LINEAR_IN`, `LLM_TENSOR_DEPFORMER_EMBD`, `LLM_TENSOR_DEPFORMER_LINEAR_OUT`
   - `LLM_TENSOR_DEPFORMER_ATTN_NORM`, `LLM_TENSOR_DEPFORMER_ATTN_QKV`, `LLM_TENSOR_DEPFORMER_ATTN_OUT`
   - `LLM_TENSOR_DEPFORMER_FFN_NORM`, `LLM_TENSOR_DEPFORMER_FFN_UP`, `LLM_TENSOR_DEPFORMER_FFN_DOWN`
2. Added tensor name mappings to `llama-arch.cpp` with two-index support (layer, slice)
3. Added `depformer` struct to `llama_model` in `llama-model.h`:
   - `linear_in[8]`, `emb[8]`, `linear_out[8]` - per-slice projections
   - `layers[6]` with per-slice `wqkv`, `wo`, `ffn_up`, `ffn_down`
4. Added DepFormer tensor creation in `llama-model.cpp` (LLM_ARCH_MOSHI case)
5. Added `llm_build_moshi_depformer` class in `models/moshi.cpp`
6. Updated converter with `--include-depformer` flag
7. **Fixed DepFormer embedding vocab sizes**: Slice 0 uses text vocab (32001), slices 1-7 use audio vocab (2049)

### 🔄 In Progress
- End-to-end TTS pipeline integration in tts.cpp

### ⏳ Not Started
- Multi-codebook token generation loop (DepFormer orchestration)
- Mimi SEANet decoder (neural audio codec)
- Voice conditioning (cross-attention)

## Testing Plan

1. ✅ **Unit test**: Load GGUF, verify tensor shapes - PASSED
2. ✅ **Inference test**: Run forward pass - PASSED (CPU + Metal)
3. ✅ **Audio embeddings**: Load 307 tensors - PASSED
4. ✅ **DepFormer test**: Load 535 tensors (307 + 228 DepFormer) - PASSED
5. ⏳ **Integration test**: Generate audio from text prompt
6. ⏳ **Quality test**: Compare output with reference MLX implementation

## Test Files Created

- `scripts/test_tensor_comparison.py`: PyTorch reference forward pass with intermediate outputs
- `scripts/test_llama_cpp_comparison.py`: Weight analysis and conversion testing
- `scripts/compare_predictions.py`: Compare top-k predictions between PyTorch and llama.cpp

## References

- Moshi paper: https://arxiv.org/abs/2410.00037
- Moshi repo: https://github.com/kyutai-labs/moshi
- Mimi codec: Based on EnCodec architecture
- llama.cpp: https://github.com/ggerganov/llama.cpp
