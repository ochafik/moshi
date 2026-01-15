# Mimi Decoder C++ Implementation Progress

## Status: ✅ SEANet GGML WORKING (0.9999 correlation)

### Test Results (2024-12-14)

| Component | Correlation | MAE | Notes |
|-----------|-------------|-----|-------|
| SEANet GGML (with Python transformer input) | **0.9999** | 0.0019 | Excellent match! |
| Full pipeline (C++ transformer + SEANet) | ~0.5 | ~0.15 | Transformer output mismatch |

### Component Validation

| Component | Status | Notes |
|-----------|--------|-------|
| Quantizer decode (codebook lookup + sum) | ✅ MATCH | Pure C++ implementation |
| 2x upsample (ConvTranspose1d) | ✅ MATCH | Fixed end-trimming bug |
| **SEANet decoder (GGML)** | ✅ **0.9999** | Fully converted to GGML ops |
| Decoder transformer (8 layers) | ❌ MISMATCH | Output values differ from Python |

### SEANet GGML Implementation

The SEANet decoder is now fully implemented using GGML operations:
- `ggml_conv_1d` - for init_conv, final_conv, residual convolutions
- `ggml_conv_transpose_1d` - for all 4 upsample layers
- `ggml_elu` - ELU activation
- `ggml_add` - bias addition and residual connections
- `ggml_tanh` - final activation

**Key discoveries:**
1. `ggml_conv_1d` requires F16 kernel weights (uses im2col internally)
2. `ggml_conv_transpose_1d` requires manual output trimming via `ggml_view`
3. Causal padding must be applied manually with `ggml_pad_ext`

### Remaining Issue: Transformer Mismatch

Python transformer output @ t=0: [0.93, 1.72, -0.19, 0.74, ...]
C++ transformer output @ t=0:   [0.52, 0.28, -0.11, -0.61, ...]

These values are completely different despite using the same input. Need to investigate:
- Causal mask implementation
- RoPE (rotary position embeddings)
- Layer normalization

### Key Bug Fixes Applied

1. **ConvTranspose1d padding/trimming** (CRITICAL)
   - **Bug**: C++ used symmetric padding (remove from both ends)
   - **Fix**: Changed to end-trimming only (like PyTorch's `unpad1d(y, (0, K-S))`)

2. **GGUF Weight Layout** (for GGML compatibility)
   - **GGML expects**: `ne[] = [K, IC, OC]` for conv1d kernels
   - Keep PyTorch layout without transpose for GGML compatibility

3. **ggml_conv_1d requires 4D input**
   - Input tensor must be `[seq_len, in_ch, batch, 1]` (ne[3]=1)

4. **ggml_cont not needed after ggml_add**
   - Removed unnecessary ggml_cont which was causing zeros

### Files

- **C++ Implementation**: `/Users/ochafik/github/moshi/tools/mimi_decode.cpp`
- **GGUF Model**: `/tmp/mimi-decoder.gguf`
- **Python Reference**: `/Users/ochafik/github/moshi/scripts/decode_mimi.py`
- **Debug Script**: `/Users/ochafik/github/moshi/scripts/debug_seanet_intermediate.py`

### Command Usage

```bash
# Build
cd build && make mimi_decode

# Run with Python SEANet input (for isolated SEANet testing)
SEANET_INPUT_BIN=/tmp/seanet_input.bin ./mimi_decode /tmp/speech_tokens.json /tmp/output.wav

# Run full pipeline (uses C++ transformer)
./mimi_decode /tmp/speech_tokens.json /tmp/output.wav

# Use pure C++ SEANet (for comparison)
USE_CPP_SEANET=1 ./mimi_decode /tmp/speech_tokens.json /tmp/output.wav
```

### Environment Variables

- `SEANET_INPUT_BIN` - Path to binary file with SEANet input (for isolated testing)
- `USE_CPP_SEANET=1` - Use pure C++ SEANet instead of GGML
