# Moshi TTS/STT llama.cpp Integration Status

## Available Models

| Model | Params | Purpose | HuggingFace |
|-------|--------|---------|-------------|
| TTS | 1.6B | Text-to-Speech | `kyutai/tts-1.6b-en_fr` |
| STT | 1B | Speech-to-Text (EN+FR) | `kyutai/stt-1b-en_fr` |
| STT | 2.6B | Speech-to-Text (EN) | `kyutai/stt-2.6b-en` |
| Moshi | 7B | Dialogue (S2S) | `kyutai/moshiko-pytorch-bf16` |

**Note**: The base Moshi 7B model is for dialogue, NOT standalone TTS/STT.

---

## GGUF Conversion (Complete)

```bash
# Convert TTS model (1.6B) - ~3.7GB output
python scripts/convert_tts_to_gguf.py -o /tmp/moshi-tts.gguf

# Convert STT model (1B) - ~2GB output
python scripts/convert_stt_to_gguf.py -o /tmp/moshi-stt.gguf
```

### TTS GGUF Structure (418 tensors)
- `text_emb.*` - Text embeddings (8000 vocab)
- `emb.{0-31}.*` - 32 audio codebook embeddings
- `transformer.layers.{0-15}.*` - 16 transformer layers with cross-attention
- `depformer.*` - 4 DepFormer layers x 11 slices
- `depformer_in.{0-10}.*` - DepFormer input projections
- `depformer_emb.{0-30}.*` - DepFormer embeddings (low-rank)
- `linears.{0-31}.*` - Output projections for 32 codebooks
- `condition_provider.*` - CFG, control, speaker conditioning

### STT GGUF Structure (131 tensors)
- `text_emb.*` - Text embedding
- `emb.{0-31}.*` - 32 audio codebook embeddings
- `transformer.layers.{0-15}.*` - 16 transformer layers (self-attn only)
- `out_norm.*` - Output normalization
- `text_linear.*` - Text output projection (8000 vocab)

---

## WORKING - Python CLI Tools

### TTS (Text-to-Speech)
```bash
# Using DSM TTS model (1.6B)
cd /path/to/moshi/moshi
python3 ../scripts/dsm_tts.py "Hello, how are you today?" -o /tmp/output.wav

# Options:
#   --voice vctk/p225_023.wav  # Voice from kyutai/tts-voices
#   --cfg 2.0                   # CFG coefficient
#   --n-q 16                    # Codebooks (8-32, higher=better)
#   --device cuda               # Use GPU
#   -v                          # Verbose
```

### STT (Speech-to-Text)
```bash
# Using STT model (1B)
python3 ../scripts/moshi_stt.py audio.wav

# Options:
#   -o transcript.txt  # Save to file
#   --device cuda      # Use GPU
#   -v                 # Verbose
```

### Example: Round-Trip Test
```bash
# Generate speech
python3 scripts/dsm_tts.py "Testing one two three" -o /tmp/test.wav -v

# Transcribe it back
python3 scripts/moshi_stt.py /tmp/test.wav -v
# Output: "one, two, three."
```

---

## Implemented Components

### 1. Python Mimi Decoder
- **File**: `scripts/decode_mimi.py`
- **Status**: Working
- Decodes audio tokens from JSON to WAV using PyTorch Mimi
- Produces intelligible speech from valid audio tokens

### 2. Native Mimi Decoder (Partial)
- **File**: `tools/mimi_decode.cpp`
- **Status**: Skeleton working, needs transformer
- **Current Features**:
  - GGUF model loading
  - Quantizer decode (embedding lookup with output projection)
  - SEANet upsampling convolutions (4 stages with ratios 8, 6, 5, 4)
  - WAV file output
- **Missing**:
  - Decoder transformer (8 layers) - currently skipped
  - The transformer is needed for good audio quality

### 3. GGUF Conversions
- **Moshi 7B**: `scripts/convert_moshi_to_gguf.py` - 535 tensors
- **Mimi decoder**: `scripts/convert_mimi_to_gguf.py` - 209 tensors

### 4. llama.cpp DepFormer (for dialogue model)
- **Note**: This generates tokens but base Moshi isn't designed for TTS
- DepFormer CPU implementation generates 8 audio codebooks per frame
- Outputs JSON with audio tokens

---

## Architecture Reference

### DSM TTS Model (1.6B) Structure
```
dim: 2048
num_layers: 16 (with cross-attention for voice conditioning)
n_q: 32 codebooks
dep_q: 32 (DepFormer outputs all codebooks)
text_card: 8000
depformer_dim: 1024
depformer_num_layers: 4
```

### Mimi Decoder Structure
```
Input: 8 codebook indices per frame
  |
  v
Quantizer: Sum projected embeddings
  - embedding = embedding_sum / cluster_usage
  - output_proj: [d_model, codebook_dim]
  |
  v
Transformer (8 layers, 512d)
  - LayerNorm + Self-Attention + FFN
  - Layer scaling
  |
  v
SEANet decoder
  - init_conv: 512 -> 1024, kernel=7
  - 4 upsampling blocks with ratios [8, 6, 5, 4] = 960x
  - Each block: ConvTranspose1D + ELU + Residual
  - final_conv: 64 -> 1, kernel=3
  |
  v
Output: 24kHz mono audio (1920 samples per frame @ 12.5 fps)
```

---

## Completed

- [x] GGUF conversion for TTS model (1.6B) - 418 tensors
- [x] GGUF conversion for STT model (1B) - 131 tensors
- [x] Python STT inference working (streaming mode with LMGen)
- [x] Python TTS inference working (DSM model with CFG)
- [x] CLI tools: `dsm_tts.py` and `moshi_stt.py`
- [x] Round-trip verification (TTS -> STT)
- [x] **Native C++ STT inference** - `tools/moshi_stt.cpp`

## Native C++ STT Tool

The native C++ STT implementation is complete and produces identical output to Python.

### Usage
```bash
# Build
mkdir -p build && cd build && cmake .. && make

# Run (requires audio tokens as JSON)
DYLD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./moshi_stt audio_tokens.json -m /tmp/moshi-stt-f32.gguf
```

### Features
- Streaming inference with text token feedback (matches Python LMGen)
- KV caching for efficient autoregressive generation
- 16 transformer layers with RoPE, RMSNorm, GQA, SiLU FFN
- Outputs text tokens (decode with SentencePiece tokenizer)

### Key Implementation Details
- Frame 0 uses initial tokens (text=8000, audio=2048)
- Frame N uses previous text token + current audio tokens
- FFN hidden dim is 5632 (from `linear_out.ne[0]`)
- Text vocab is 8001 (includes BOS token at index 8000)

## Next Steps

### For Native llama.cpp Implementation

1. **TTS Model (Complex - Needs DepFormer)**
   - Architecture: 16 transformer layers WITH cross-attention
   - DepFormer: 4 layers x 11 slices -> 32 codebooks
   - Needs voice embedding from Mimi encoder (cross-attention source)
   - CFG (Classifier-Free Guidance) distillation
   - Implementation:
     ```cpp
     // For each step:
     // 1. Embed text + previous audio tokens
     // 2. Run transformer with cross-attention to voice embedding
     // 3. Run DepFormer (4 layers, 11 slices)
     // 4. Sample from 32 codebook distributions
     ```

3. **Complete Native Mimi Decoder**
   - Add 8-layer transformer with attention (currently skipped)
   - The transformer refines quantizer embeddings before SEANet
   - Without it, audio quality is degraded

---

## Voices Available

From HuggingFace `kyutai/tts-voices`:
- `vctk/p225_023.wav` through `vctk/p251_023.wav`
- `expresso/*`
- `cml-tts/*`
- `voice-donations/*`
