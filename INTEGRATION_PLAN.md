# Moshi STT Integration Plan

## Current Implementation Status

### Working Components (Ready to Commit)

| Component | File | Description |
|-----------|------|-------------|
| Mimi Encoder | `src/mimi.cpp`, `include/mimi.h` | SEANet + Transformer + RVQ encoder |
| Moshi STT | `src/moshi_stt.cpp`, `include/moshi_stt.h` | Autoregressive transformer decoder |
| STT CLI | `tools/moshi_stt.cpp` | Command-line transcription tool |
| Mimi CLI | `tools/mimi_encode.cpp` | Audio-to-tokens encoder |
| HTTP Server | `tools/moshi_server.cpp` | `/v1/audio/transcriptions` endpoint |
| GGUF Converters | `scripts/convert_*.py` | Model format converters |

### Key Implementation Details

1. **Mimi Encoder** (Audio → Tokens)
   - Sample rate: 24kHz
   - Frame rate: 12.5 Hz (1 frame per 80ms)
   - Output: 32 codebooks × 2048 vocabulary
   - Architecture: SEANet conv encoder → Transformer → RVQ quantizer

2. **Moshi STT** (Tokens → Text)
   - **Autoregressive text prediction** - feeds predicted text back as input
   - 16 transformer layers, 2048 dim, 16 heads
   - Text vocabulary: 8000 (sentencepiece)
   - Uses RoPE positional encoding
   - Causal attention with KV cache

## Integration Path to llama.cpp

### Phase 1: Standalone Server (Now)

Use `moshi_server` as standalone STT service:

```bash
# Build
cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
make moshi_server

# Convert models
python scripts/convert_mimi_to_gguf.py --out mimi-encoder.gguf
python scripts/convert_moshi_to_gguf.py --out moshi-stt.gguf

# Run server
./moshi_server --mimi mimi-encoder.gguf --stt moshi-stt.gguf --port 8080

# Test
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@audio.wav
```

### Phase 2: Port to ggml Graphs (Medium Term)

Convert manual loops to ggml computation graphs for GPU acceleration:

**Current (CPU-only, manual loops):**
```cpp
void matvec(float * out, const float * W, const float * in, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) sum += W[o * in_dim + i] * in[i];
        out[o] = sum;
    }
}
```

**Target (ggml, GPU-accelerated):**
```cpp
struct ggml_tensor * out = ggml_mul_mat(ctx, W, in);
```

Key operations to convert:
- [ ] Matrix multiplication (`ggml_mul_mat`)
- [ ] RMS normalization (`ggml_rms_norm`)
- [ ] RoPE (`ggml_rope`)
- [ ] Softmax (`ggml_soft_max`)
- [ ] SiLU activation (`ggml_silu`)
- [ ] Causal attention (`ggml_flash_attn_ext`)
- [ ] Convolution (for Mimi SEANet)

### Phase 3: mtmd Integration (Longer Term)

Add to `llama.cpp/tools/mtmd/`:

```
mtmd/
├── clip.cpp           # Vision encoder (existing)
├── mtmd-audio.cpp     # Whisper preprocessing (existing)
├── mtmd-mimi.cpp      # Mimi audio encoder (NEW)
├── mtmd-moshi-stt.cpp # Moshi STT model (NEW)
└── mtmd.cpp           # Main multimodal interface
```

Integration points:
1. Add `mtmd_audio_encode()` for Mimi
2. Add `mtmd_audio_transcribe()` for STT
3. Modify `mtmd.h` to expose audio transcription API

### Phase 4: llama-server Endpoint (Final)

Add to `llama.cpp/tools/server/server.cpp`:

```cpp
// POST /v1/audio/transcriptions
svr->Post(params.api_prefix + "/v1/audio/transcriptions", handle_audio_transcriptions);
```

Requirements:
- Sentencepiece tokenizer integration (embed vocab in GGUF or use external)
- Audio format conversion (ffmpeg or miniaudio)
- Streaming support (optional)

## Architecture Comparison

### Current (Moshi Repo)
```
audio.wav → [Mimi Encoder] → tokens → [Moshi STT] → text_tokens → decode
                ↓                         ↓
           mimi.cpp              moshi_stt.cpp
```

### Target (llama.cpp)
```
audio.wav → [mtmd-mimi] → tokens → [llama_decode] → text_tokens → decode
                ↓                         ↓
         GGUF model              GGUF model (using llama.cpp inference)
```

## Model Format

### GGUF Tensors (STT)
- `text_emb.weight` [8001, 2048] - text embeddings (includes initial token)
- `emb.{0-31}.weight` [2049, 2048] - audio codebook embeddings
- `layers.{0-15}.norm1_alpha` [2048] - attention norm
- `layers.{0-15}.norm2_alpha` [2048] - FFN norm
- `layers.{0-15}.attn.in_proj` [6144, 2048] - Q/K/V projection
- `layers.{0-15}.attn.out_proj` [2048, 2048] - output projection
- `layers.{0-15}.ffn.linear_in` [16896, 2048] - FFN gate+up
- `layers.{0-15}.ffn.linear_out` [2048, 8448] - FFN down
- `out_norm_alpha` [2048] - final norm
- `text_linear` [8000, 2048] - output projection

### GGUF Tensors (Mimi Encoder)
- SEANet conv weights
- Transformer layers
- RVQ codebooks (embedding_sum, cluster_usage)

## Testing

```bash
# E2E test
say -o test.aiff "Hello world"
afconvert -f WAVE -d LEF32@24000 test.aiff test.wav
./mimi_encode --model mimi-encoder.gguf --audio test.wav --out tokens.json
./moshi_stt moshi-stt.gguf tokens.json
# Output: 5468 896 ... (text token IDs)
```

## Files to Commit

### Essential
- `src/moshi_stt.cpp` - STT implementation
- `src/mimi.cpp` - Mimi encoder
- `include/moshi_stt.h` - STT header
- `include/mimi.h` - Mimi header
- `tools/moshi_stt.cpp` - STT CLI
- `tools/mimi_encode.cpp` - Encoder CLI
- `tools/moshi_server.cpp` - HTTP server
- `CMakeLists.txt` - Build system
- `scripts/convert_mimi_to_gguf.py` - Model converter
- `scripts/convert_moshi_to_gguf.py` - Model converter
- `include/httplib.h` - HTTP library

### Optional
- `scripts/moshi_stt.py` - Python reference
- `tests/test_stt_e2e.py` - E2E tests
