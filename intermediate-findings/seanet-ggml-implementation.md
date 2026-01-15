# SEANet GGML Implementation - Findings & Plan

**Date**: 2024-12-14
**Status**: SEANet GGML working (0.9999 correlation), Transformer needs debugging

## Executive Summary

The SEANet decoder has been successfully converted to GGML operations and achieves **0.9999 correlation** with Python reference when given correct input. The remaining issue is the GGML transformer output doesn't match Python.

## Architecture Overview

```
Mimi Decoder Pipeline:
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Quantizer       │ --> │ 2x Upsample  │ --> │ Transformer     │ --> │ SEANet          │
│ (C++ codebook)  │     │ (C++ ConvTr) │     │ (GGML - broken) │     │ (GGML - works!) │
└─────────────────┘     └──────────────┘     └─────────────────┘     └─────────────────┘
       ✅                      ✅                    ❌                      ✅
```

## Test Results

| Component | Status | Correlation | Notes |
|-----------|--------|-------------|-------|
| Quantizer decode | ✅ MATCH | N/A | Pure C++ codebook lookup |
| 2x Upsample | ✅ MATCH | N/A | ConvTranspose1d with end-trimming |
| **SEANet (GGML)** | ✅ **WORKS** | **0.9999** | When given Python transformer output |
| Transformer (GGML) | ❌ MISMATCH | ~0.5 | Output values completely different |

## SEANet GGML Implementation Details

### Operations Used

| Operation | GGML Function | Notes |
|-----------|---------------|-------|
| Conv1d | `ggml_conv_1d` | Requires F16 kernel (uses im2col internally) |
| ConvTranspose1d | `ggml_conv_transpose_1d` | Requires manual output trimming via `ggml_view` |
| ELU | `ggml_elu` | Standard ELU activation |
| Bias add | `ggml_add` | With proper 4D reshape for broadcasting |
| Tanh | `ggml_tanh` | Final activation |
| Causal padding | `ggml_pad_ext` | Manual left-padding for causal convolutions |

### Key Discoveries

1. **ggml_conv_1d requires F16 kernel weights**
   - The im2col implementation internally requires `GGML_TYPE_F16`
   - Solution: Use `ggml_cast(ctx, kernel_f32, GGML_TYPE_F16)` in graph

2. **ggml_conv_1d expects 4D input tensor**
   - Input must be `[seq_len, in_ch, batch, 1]` with `ne[3]=1`
   - Not 2D `[seq_len, in_ch]` as one might expect

3. **Causal padding must be manual**
   - GGML conv_1d doesn't have causal mode
   - Use `ggml_pad_ext` to add zeros on left before conv

4. **ConvTranspose1d output trimming**
   - PyTorch uses `unpad1d(y, (0, kernel-stride))` - trim from END only
   - Must use `ggml_view` to trim output after conv_transpose

5. **ggml_cont was causing zeros**
   - Unnecessary `ggml_cont` after `ggml_add` produced all zeros
   - Simply use the add result directly

### Weight Layout

GGML expects weights in column-major format:
- `ggml_conv_1d` kernel: `ne[] = [K, IC, OC]`
- PyTorch stores: `[OC, IC, K]` (row-major)
- When loaded into GGML, the bytes are reinterpreted as column-major

The GGUF converter should NOT transpose - just write PyTorch tensor bytes directly.

## Verification Commands

```bash
# 1. Export Python transformer output as binary
cd /Users/ochafik/github/moshi
python3 << 'EOF'
import torch, json, struct, numpy as np, sys
sys.path.insert(0, "moshi")
from moshi.models import loaders
from huggingface_hub import hf_hub_download

mimi = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), device="cpu")
mimi.set_num_codebooks(8)
mimi.eval()

with open("/tmp/speech_tokens.json") as f:
    codes = torch.tensor([json.load(f)["audio_tokens"]], dtype=torch.long).transpose(1, 2)

with torch.no_grad():
    z = mimi.quantizer.decode(codes)
    z_up = mimi.upsample(z)
    x = mimi.decoder_transformer(z_up)
    x = x[0] if isinstance(x, (list, tuple)) else x

x_np = x[0].numpy()
with open("/tmp/seanet_input.bin", "wb") as f:
    f.write(struct.pack("ii", *x_np.shape))
    f.write(x_np.astype(np.float32).tobytes())
print(f"Saved SEANet input: {x_np.shape}")
EOF

# 2. Run C++ decoder with Python input
cd build
SEANET_INPUT_BIN=/tmp/seanet_input.bin ./mimi_decode /tmp/speech_tokens.json /tmp/test.wav

# 3. Compare correlation
python3 << 'EOF'
import numpy as np, struct, wave

# Load Python reference
with open("/tmp/seanet_ref_audio.bin", "rb") as f:
    n = struct.unpack("i", f.read(4))[0]
    ref = np.frombuffer(f.read(n * 4), dtype=np.float32)

# Load C++ output
with wave.open("/tmp/speech_tokens_native.wav", "rb") as w:
    cpp = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0

min_len = min(len(ref), len(cpp))
corr = np.corrcoef(ref[:min_len], cpp[:min_len])[0, 1]
print(f"Correlation: {corr:.6f}")  # Should be ~0.9999
EOF
```

## Transformer Issue Analysis

### Symptoms

Python transformer output @ t=0:
```
[0.93, 1.72, -0.19, 0.74, -0.68, -1.39, 0.86, -0.04]
Range: [-4.87, 4.04]
```

C++ GGML transformer output @ t=0:
```
[0.52, 0.28, -0.11, -0.61, -0.57, -0.54, -0.54, -0.13]
Range: [-5.08, 8.27]
```

Values are completely different despite same input.

### Possible Causes

1. **Causal Mask Implementation**
   - Python uses `causal=True` in attention
   - GGML might need explicit mask tensor

2. **RoPE Frequency Calculation**
   - Mimi uses specific RoPE configuration
   - Frequency base or scaling might differ

3. **LayerNorm**
   - Epsilon value differences
   - Pre-norm vs post-norm ordering

4. **Attention Scaling**
   - Scale factor: `1/sqrt(head_dim)` vs `1/sqrt(d_model)`

5. **FFN Activation**
   - GELU vs SiLU vs ReLU differences

## Plan: Transformer Debugging

### Phase 1: Isolate the Issue (Layer-by-Layer)

1. Export Python intermediate values after each operation:
   - After LayerNorm1
   - After QKV projection
   - After RoPE
   - After attention scores
   - After attention output
   - After LayerNorm2
   - After FFN

2. Compare with C++ at each stage to find divergence point

### Phase 2: Fix Identified Issues

Based on divergence point:
- If LayerNorm: Check epsilon, weight application order
- If QKV: Check projection weight layout
- If RoPE: Check frequency calculation, position indexing
- If Attention: Check mask, scaling, softmax
- If FFN: Check activation function, weight layout

### Phase 3: Validate End-to-End

Once transformer matches:
1. Run full pipeline without `SEANET_INPUT_BIN`
2. Compare final audio with Python reference
3. Target: >0.99 correlation

## Files Reference

| File | Purpose |
|------|---------|
| `tools/mimi_decode.cpp` | Main C++ implementation |
| `scripts/debug_seanet_intermediate.py` | Python debug script for SEANet |
| `scripts/convert_mimi_to_gguf.py` | GGUF weight converter |
| `/tmp/mimi-decoder.gguf` | Converted model weights |
| `/tmp/seanet_input.bin` | Python transformer output (for testing) |
| `/tmp/seanet_ref_audio.bin` | Python SEANet output (reference) |

## Appendix: SEANet Architecture

```
SEANet Decoder:
├── init_conv: Conv1d(512, 1024, k=7, causal)
├── ELU
├── Block 0: ratio=8, 1024→512
│   ├── ConvTranspose1d(1024, 512, k=16, s=8)
│   ├── ResBlock: ELU→Conv(512,256,k=3)→ELU→Conv(256,512,k=1)
│   └── ELU
├── Block 1: ratio=6, 512→256
│   ├── ConvTranspose1d(512, 256, k=12, s=6)
│   ├── ResBlock: ELU→Conv(256,128,k=3)→ELU→Conv(128,256,k=1)
│   └── ELU
├── Block 2: ratio=5, 256→128
│   ├── ConvTranspose1d(256, 128, k=10, s=5)
│   ├── ResBlock: ELU→Conv(128,64,k=3)→ELU→Conv(64,128,k=1)
│   └── ELU
├── Block 3: ratio=4, 128→64
│   ├── ConvTranspose1d(128, 64, k=8, s=4)
│   ├── ResBlock: ELU→Conv(64,32,k=3)→ELU→Conv(32,64,k=1)
│   └── ELU
├── final_conv: Conv1d(64, 1, k=3, causal)
└── Tanh
```

Total upsample factor: 8 × 6 × 5 × 4 = 960
Input: [512, T] → Output: [1, T×960]
