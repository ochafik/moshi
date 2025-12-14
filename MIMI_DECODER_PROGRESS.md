# Mimi Decoder C++ Implementation Progress

## Status: ✅ WORKING (High Correlation)

### Test Results (2024-12-14)

| Test | Frames | Duration | Correlation | MAE | Notes |
|------|--------|----------|-------------|-----|-------|
| speech_tokens.json | 25 | 2.0s | 0.9986 | 0.0056 | Excellent match |
| long_tokens.json | 50 | 4.0s | 0.9696 | 0.0210 | Good match, some accumulated error |

### Component Validation

| Component | Status | Notes |
|-----------|--------|-------|
| Quantizer decode (codebook lookup + sum) | ✅ MATCH | Pure C++ implementation |
| 2x upsample (ConvTranspose1d) | ✅ MATCH | Fixed end-trimming bug |
| Decoder transformer (8 layers) | ✅ MATCH | Uses GGML backend |
| SEANet decoder (4 upsamples + residual blocks) | ✅ MATCH | Fixed ConvTranspose1d trimming |

### Key Bug Fixes Applied

1. **ConvTranspose1d padding/trimming** (CRITICAL)
   - **Bug**: C++ used symmetric padding (remove from both ends)
   - **Fix**: Changed to end-trimming only (like PyTorch's `unpad1d(y, (0, K-S))`)
   - **Impact**: Correlation jumped from -0.55 to 0.999+ without transformer

### Architecture: Hybrid C++/GGML

**Using GGML:**
- Decoder transformer (self-attention, feed-forward, layer norm)

**Pure C++ (TODO: convert to GGML):**
- `conv1d()` - standard 1D convolution
- `conv_transpose1d()` - transposed convolution for upsampling
- `apply_elu()` - ELU activation
- Quantizer decode (codebook lookup)
- SEANet decoder (all conv layers)

### GGML Conversion Backlog

Priority order for converting pure C++ to GGML:

1. **conv1d()** → `ggml_conv_1d`
   - Used in SEANet init_conv, final_conv, all residual blocks
   - Critical for consistency and potential Metal/CUDA acceleration

2. **conv_transpose1d()** → `ggml_conv_transpose_1d`
   - Used for all upsampling layers (4 total in SEANet)
   - Need to handle end-trimming correctly

3. **apply_elu()** → `ggml_elu`
   - Used throughout SEANet decoder
   - Simple conversion

4. **Quantizer decode** → `ggml_get_rows` + `ggml_add`
   - Codebook lookup and summation across 8 codebooks
   - Would benefit from batch operations

5. **Full SEANet graph**
   - Once individual ops are converted, build full computation graph
   - Enables better optimization and memory management

### Files

- **C++ Implementation**: `/Users/ochafik/github/moshi/tools/mimi_decode.cpp`
- **GGUF Model**: `/tmp/mimi-decoder.gguf` (converted from Moshi weights)
- **Python Reference**: `/Users/ochafik/github/moshi/scripts/decode_mimi.py`
- **Debug Script**: `/Users/ochafik/github/moshi/scripts/debug_mimi_intermediate.py`

### Command Usage

```bash
# Build
cd build && make mimi_decode

# Run C++ decoder
./mimi_decode /tmp/speech_tokens.json /tmp/output.wav

# Run Python reference
python scripts/decode_mimi.py /tmp/speech_tokens.json -o /tmp/reference.wav
```

### Environment Variables

- `SKIP_TRANSFORMER=1` - Bypass transformer (for debugging SEANet only)
