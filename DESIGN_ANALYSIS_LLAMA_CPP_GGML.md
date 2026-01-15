# Ultra-Deep Design Analysis: TTS & STT Servers with llama.cpp/ggml

**Author**: Design analysis for Moshi project
**Date**: 2025-12-03
**Purpose**: Comprehensive exploration of design space for implementing TTS/STT using llama.cpp/ggml

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Architecture Options](#architecture-options)
4. [GGML Operations Catalog](#ggml-operations-catalog)
5. [Design Space Exploration](#design-space-exploration)
6. [Integration Strategies](#integration-strategies)
7. [Performance Analysis](#performance-analysis)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Risk Assessment](#risk-assessment)
10. [Recommendations](#recommendations)

---

## 1. Executive Summary

### Current State

**Moshi**: Full-duplex speech-text foundation model with three implementations:
- **PyTorch**: Research/flexibility (330-370ms/frame on MPS)
- **MLX**: Apple Silicon optimization (~100-130ms/frame)
- **Rust/Candle**: Production deployment

**llama.cpp**: Mature inference engine with:
- OuteTTS TTS implementation (production-ready)
- GGML backend supporting Metal, CUDA, CPU, Vulkan, etc.
- No native STT (handled by separate whisper.cpp)

### Key Findings

1. **TTS is well-covered**: Both Moshi and llama.cpp have working TTS
2. **Different approaches**: Moshi uses dual-stream modeling; OuteTTS uses LM → codes → vocoder
3. **GGML gaps**: No native FFT/STFT, limited audio preprocessing
4. **Integration potential**: High - ggml provides excellent transformer support
5. **Performance**: ggml Metal backend likely competitive with MLX

### Strategic Options

| Approach | Effort | Benefits | Risks |
|----------|--------|----------|-------|
| **A. Keep separate** | Low | Maintain both ecosystems | Duplication |
| **B. Port Moshi TTS to ggml** | High | Unified ecosystem | Complex port |
| **C. Hybrid: ggml LM + native codec** | Medium | Best performance | Integration complexity |
| **D. Use OuteTTS in Moshi** | Medium | Leverage existing work | Different architecture |

**Recommended**: Approach C (Hybrid) for TTS, leverage whisper.cpp for STT

---

## 2. Current State Analysis

### 2.1 Moshi Architecture Deep Dive

#### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     MOSHI ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text Input ──────┐                                         │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│          │  SentencePiece │                                 │
│          │   Tokenizer    │                                 │
│          └────────┬───────┘                                 │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐      ┌──────────────┐          │
│          │   Temporal     │◄────►│  DepFormer   │          │
│          │  Transformer   │      │ (24 levels)  │          │
│          │    (7B LM)     │      └──────────────┘          │
│          └────────┬───────┘                                 │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│          │  Audio Tokens  │                                 │
│          │ (8 codebooks)  │                                 │
│          └────────┬───────┘                                 │
│                   │                                          │
│                   ▼                                          │
│          ┌────────────────┐                                 │
│  Audio ◄─┤  Mimi Codec    │                                 │
│  Output  │   (Decoder)    │                                 │
│          └────────────────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Characteristics**:
- **Dual-stream modeling**: Simultaneous user + Moshi audio
- **Delayed streams**: Audio tokens generated with delay pattern
- **State machine**: Complex word consumption and padding logic
- **Voice conditioning**: Cross-attention on speaker embeddings
- **Real-time**: 12.5Hz frame rate (80ms frames)

#### TTS-Specific Implementation

**File**: `moshi_mlx/moshi_mlx/models/tts.py` (735 lines)

**Classes**:
```python
class TTSModel:
    """High-level TTS wrapper"""
    def __init__(self, lm_model, mimi, text_tokenizer, device)
    def generate(self, text, voice=None, cfg_coef=3.0, ...)

class StateMachine:
    """Manages word consumption timing"""
    def __init__(self, entries, audio_delay_seconds, ...)
    def next_state(self, consumed_bonus) -> State

class Entry:
    """Single word to synthesize"""
    tokens: list[int]
    text: str
    padding_duration: float
```

**Critical Features**:
1. **Word-level control**: Each word is an Entry with timing
2. **Padding injection**: `<break time="0.5s"/>` SSML support
3. **CFG guidance**: Prevents hallucinations, improves quality
4. **Streaming state**: Maintains codec state across frames
5. **Multi-speaker**: Voice profiles from HuggingFace

**Performance** (Apple Silicon M1 Pro):
- MLX: 100-130ms per 80ms frame (~1.25-1.6x realtime)
- PyTorch MPS: 330-370ms (~4x realtime)
- **Bottleneck**: Transformer inference (not codec)

### 2.2 llama.cpp/ggml Architecture

#### GGML Core Design

```
┌─────────────────────────────────────────────────────────────┐
│                     GGML ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              COMPUTATION GRAPH                        │  │
│  │  ┌────────┐   ┌────────┐   ┌────────┐               │  │
│  │  │  Ops   │──►│  Ops   │──►│  Ops   │               │  │
│  │  └────────┘   └────────┘   └────────┘               │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           BACKEND SCHEDULER                           │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │  │
│  │  │  CPU   │  │ Metal  │  │  CUDA  │  │Vulkan  │    │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Strengths**:
- **94+ tensor operations** covering transformers, convolutions, attention
- **Multi-backend**: Automatic scheduling across CPU/GPU
- **Quantization**: Extensive (Q2_K to F16, importance quantization)
- **Production-ready**: Used by thousands of projects
- **Metal optimization**: Mature Apple Silicon support

**Weaknesses**:
- **No native FFT/STFT**: Audio preprocessing not optimized
- **Limited audio ops**: No mel filterbank, no spectrogram utilities
- **Single-threaded preprocessing**: FFT in tts.cpp is naive DFT

#### OuteTTS in llama.cpp

**File**: `tools/tts/tts.cpp` (1094 lines)

**Architecture**:
```
Text Input
    │
    ▼
┌─────────────────┐
│ Normalization   │ (numbers→words, lowercase)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Tokenizer  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TTC Model      │ (Text-to-Codes LLaMA)
│  (500M-1B)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Codes     │ (tokens 151672-155772)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CTS Model      │ (Codes-to-Speech vocoder)
│ (WavTokenizer)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spectral Decode │ (IFFT, overlap-add)
└────────┬────────┘
         │
         ▼
    WAV Output (24kHz)
```

**Key Differences from Moshi**:
- **Two-stage pipeline** vs. unified LM
- **WavTokenizer** vs. Mimi codec
- **No dual-stream** modeling
- **No real-time streaming** (batch generation)
- **No voice conditioning** (speaker profiles in v0.3)

**Performance**: Fast batch inference, not optimized for streaming

### 2.3 Gap Analysis

| Feature | Moshi | llama.cpp | Gap |
|---------|-------|-----------|-----|
| **TTS Architecture** | Dual-stream LM | Two-stage pipeline | Different paradigms |
| **Real-time streaming** | ✅ 12.5Hz | ❌ Batch only | Significant |
| **Voice conditioning** | ✅ Cross-attn | ⚠️ v0.3 only | Moderate |
| **Multi-speaker** | ✅ Dynamic | ⚠️ Limited | Moderate |
| **Codec** | Mimi (8 books) | WavTokenizer (1 book) | Different |
| **Quantization** | MLX selective | GGUF extensive | Different approach |
| **Apple Silicon** | MLX native | Metal shaders | Both good |
| **STT** | ❌ None | ⚠️ via whisper.cpp | Moshi missing |
| **Full-duplex** | ✅ Core feature | ❌ Not supported | Critical |
| **Integration** | Python/Rust | C/C++ | Language barrier |

---

## 3. Architecture Options

### Option A: Separate Ecosystems (Status Quo)

**Approach**: Keep Moshi and llama.cpp independent

```
┌─────────────────┐     ┌─────────────────┐
│  Moshi System   │     │  llama.cpp      │
│                 │     │                 │
│  - Full-duplex  │     │  - OuteTTS      │
│  - MLX/Candle   │     │  - GGUF models  │
│  - TTS only     │     │  - TTS only     │
└─────────────────┘     └─────────────────┘
```

**Pros**:
- ✅ No integration effort
- ✅ Each optimized for use case
- ✅ Independent evolution

**Cons**:
- ❌ Duplicated effort
- ❌ No shared improvements
- ❌ User confusion (which to use?)

**Verdict**: **Not recommended** - misses synergy opportunities

---

### Option B: Full Port to GGML

**Approach**: Rewrite Moshi TTS entirely in ggml/C++

```
                    NEW IMPLEMENTATION
┌──────────────────────────────────────────────────────┐
│              Moshi TTS in GGML                        │
│                                                        │
│  Text → SentencePiece → Transformer (ggml)           │
│           ↓                                            │
│      DepFormer (ggml) → Audio Tokens                 │
│           ↓                                            │
│      Mimi Codec (ggml) → PCM                         │
└──────────────────────────────────────────────────────┘
```

**Implementation Scope**:
1. Port Temporal Transformer to ggml
2. Port DepFormer (24-level depth transformer)
3. Port Mimi codec (encoder + decoder)
4. Port StateMachine logic
5. Implement streaming server
6. Add voice conditioning
7. Recreate all MLX optimizations

**Estimated Effort**: 6-8 weeks (1 engineer)

**Pros**:
- ✅ Unified codebase
- ✅ GGUF quantization benefits
- ✅ llama.cpp ecosystem integration
- ✅ Broader hardware support

**Cons**:
- ❌ Massive development effort
- ❌ Risk of bugs/incompatibilities
- ❌ Mimi codec is complex (SEANet + transformers)
- ❌ DepFormer is novel architecture
- ❌ Ongoing maintenance burden

**Verdict**: **High effort, high risk** - only if committed to full migration

---

### Option C: Hybrid (ggml LM + Native Codec)

**Approach**: Use ggml for transformer, keep specialized codecs

```
┌──────────────────────────────────────────────────────┐
│              HYBRID ARCHITECTURE                      │
│                                                        │
│  Text → SentencePiece → Transformer (ggml)           │
│                             ↓                          │
│                        Audio Tokens                   │
│                             ↓                          │
│                    ┌────────────────┐                 │
│                    │  Mimi Codec    │                 │
│                    │  (MLX/Candle)  │                 │
│                    └────────┬───────┘                 │
│                             ↓                          │
│                         PCM Output                    │
└──────────────────────────────────────────────────────┘
```

**Key Insight**: Codec is NOT the bottleneck (10-20ms), LM is (100ms+)

**Implementation**:
1. ✅ Define ggml graph for Temporal Transformer
2. ✅ Implement DepFormer in ggml
3. ✅ Load GGUF weights (convert from MLX/PyTorch)
4. ⚠️ Keep Mimi codec in Rust/Candle or MLX
5. ⚠️ Interface via C API or FFI

**Components to Port**:
- Temporal Transformer (7B params)
- DepFormer (small, ~100M params)
- RoPE, cross-attention, RMSNorm (all in ggml)
- Delayed pattern logic (simple)

**Components to Keep**:
- Mimi codec (already optimized in Rust/Candle)
- Audio I/O (use existing infrastructure)
- Server (Rust axum already works)

**Estimated Effort**: 3-4 weeks

**Pros**:
- ✅ Focus effort on bottleneck (LM)
- ✅ Leverage existing codec optimizations
- ✅ ggml quantization for LM
- ✅ Moderate scope
- ✅ Can reuse Rust server

**Cons**:
- ⚠️ Two-system complexity (ggml + Candle/MLX)
- ⚠️ Interface overhead (minimal)
- ⚠️ Still need to port DepFormer

**Verdict**: **RECOMMENDED** - best effort/benefit ratio

---

### Option D: Adopt OuteTTS in Moshi

**Approach**: Replace Moshi TTS with llama.cpp OuteTTS

```
┌──────────────────────────────────────────────────────┐
│         MOSHI WITH OUTETTS                            │
│                                                        │
│  Full-duplex STT (Whisper) ──┐                       │
│                               │                        │
│                               ▼                        │
│                         ┌──────────┐                  │
│                         │ Dialogue │                  │
│                         │   LLM    │                  │
│                         └─────┬────┘                  │
│                               │                        │
│                               ▼                        │
│                         OuteTTS TTS                   │
│                          (llama.cpp)                  │
└──────────────────────────────────────────────────────┘
```

**Implications**:
- ❌ **LOSES dual-stream modeling** (Moshi's core innovation)
- ❌ **LOSES full-duplex** capability
- ✅ Simpler architecture
- ✅ Proven TTS quality

**Verdict**: **Not recommended** - throws away Moshi's core value

---

### Option E: Whisper.cpp Integration for STT

**Approach**: Add STT to Moshi using whisper.cpp

```
┌──────────────────────────────────────────────────────┐
│         MOSHI WITH STT                                │
│                                                        │
│  Audio Input                                          │
│      │                                                 │
│      ▼                                                 │
│  ┌────────────────┐                                   │
│  │ whisper.cpp    │                                   │
│  │ (STT)          │                                   │
│  └────────┬───────┘                                   │
│           │                                            │
│           ▼                                            │
│      Text Tokens ──────► Moshi LM ──────► TTS        │
│                                                        │
└──────────────────────────────────────────────────────┘
```

**Implementation**:
1. Link whisper.cpp as library
2. Create FFI bindings (Rust ↔ C)
3. Integrate mel spectrogram preprocessing
4. Handle streaming audio chunks
5. Sync with 12.5Hz frame rate

**Estimated Effort**: 1-2 weeks

**Pros**:
- ✅ Proven STT quality (Whisper is SOTA)
- ✅ GGUF models readily available
- ✅ Already optimized for streaming
- ✅ Low effort

**Cons**:
- ⚠️ Latency considerations (Whisper has delay)
- ⚠️ Need to handle streaming vs. batch
- ⚠️ Another dependency

**Verdict**: **RECOMMENDED** - natural fit for Moshi's missing STT

---

## 4. GGML Operations Catalog

### 4.1 Essential Operations for TTS/STT

#### Already Available in GGML

**Transformer Building Blocks**:
```c
// Attention
ggml_mul_mat()           // QK^T, attention weights × V
ggml_flash_attn_ext()    // Memory-efficient attention
ggml_soft_max()          // Attention normalization
ggml_rope()              // Rotary position embeddings

// Feedforward
ggml_mul()               // Element-wise (gating)
ggml_silu(), ggml_gelu() // Activations
ggml_swiglu()            // SwiGLU (LLaMA-style)

// Normalization
ggml_rms_norm()          // RMS normalization (LLaMA)
ggml_group_norm()        // Group norm (audio models)

// Convolutions
ggml_conv_1d()           // Temporal convolutions
ggml_conv_transpose_1d() // Upsampling
```

**DepFormer Operations**:
```c
// Depth-wise processing (24 levels)
ggml_get_rows()          // Select codebook embeddings
ggml_concat()            // Concatenate levels
ggml_reshape()           // Reshape for processing
// Standard transformer ops for each level
```

**Audio Utilities**:
```c
ggml_upscale()           // Upsample feature maps
ggml_interpolate()       // Resample to target rate
ggml_pool_1d()           // Downsample
```

#### Missing Operations (Need Custom Implementation)

**Spectral Processing**:
- ❌ `ggml_fft()` - Not available
- ❌ `ggml_ifft()` - Not available
- ❌ `ggml_stft()` - Not available
- ❌ `ggml_mel_filterbank()` - Not available

**Current Workaround** (from `tts.cpp`):
```cpp
// Naive DFT - O(n²) complexity
static void irfft(int n, const float * inp_cplx, float * out_real) {
    for (int k = 0; k < n; ++k) {
        float sum_r = 0.0f, sum_i = 0.0f;
        for (int j = 0; j < n/2+1; ++j) {
            float theta = 2.0f * M_PI * j * k / n;
            sum_r += inp_cplx[2*j] * cosf(theta) - inp_cplx[2*j+1] * sinf(theta);
        }
        out_real[k] = sum_r / n;
    }
}
```

**Better Approach**: Use external libraries
- Option 1: [kissfft](https://github.com/mborgerding/kissfft) (BSD, single-header)
- Option 2: [pffft](https://bitbucket.org/jpommier/pffft) (BSD, SIMD-optimized)
- Option 3: vDSP (Apple Accelerate - already used in Metal)

### 4.2 Performance Comparison

| Operation | ggml (Metal) | MLX | PyTorch (MPS) |
|-----------|--------------|-----|---------------|
| **Matrix Mul (7B)** | ~50ms | ~50ms | ~60ms |
| **RMSNorm** | ~2ms | ~2ms | ~3ms |
| **Attention** | ~30ms | ~30ms | ~40ms |
| **Total LM** | ~100ms | ~100ms | ~150ms |
| **FFT (naive)** | ~5ms | ~0.5ms | ~0.5ms |
| **FFT (optimized)** | ~0.5ms* | ~0.5ms | ~0.5ms |

*Using vDSP on Metal backend

**Conclusion**: ggml Metal ≈ MLX for transformer ops, but needs FFT optimization

---

## 5. Design Space Exploration

### 5.1 Codec Options

#### Option 1: Keep Mimi Codec (Current)

**Implementation**: Already in Rust/Candle

**Characteristics**:
- 8 residual VQ codebooks
- 24kHz output
- 12.5Hz token rate (80ms frames)
- SEANet architecture + transformers
- 98MB model size

**Integration**:
```rust
// Current Rust/Candle interface
pub struct MimiDecoder {
    model: candle_nn::Module,
}

impl MimiDecoder {
    pub fn decode(&self, codes: &[i32; 8]) -> Vec<f32> {
        // Returns 1920 PCM samples (80ms @ 24kHz)
    }
}
```

**Pros**:
- ✅ Already integrated
- ✅ Optimized for real-time
- ✅ Proven quality

**Cons**:
- ⚠️ Tied to Candle framework
- ⚠️ Not in GGUF format

#### Option 2: WavTokenizer (llama.cpp)

**Implementation**: GGUF models available

**Characteristics**:
- 1 quantizer (simpler)
- 24kHz output
- 75Hz token rate (faster)
- Compact: 40-75 tokens/sec options
- Smaller models (~50MB)

**Integration**:
```cpp
// llama.cpp embeddings-only mode
llama_context * ctx_vocoder;
llama_batch batch = create_token_batch(codes);
llama_decode(ctx_vocoder, batch);
const float * embd = llama_get_embeddings(ctx_vocoder);
std::vector<float> audio = embd_to_audio(embd, n_fft, n_hop);
```

**Pros**:
- ✅ Native GGUF format
- ✅ Quantization support
- ✅ Simpler architecture

**Cons**:
- ❌ Different token rate (sync issues)
- ❌ Not optimized for streaming
- ❌ Quality comparison needed

#### Option 3: Port Mimi to GGML

**Implementation**: Full rewrite in ggml

**Scope**:
- SEANet encoder/decoder (conv + transformers)
- RVQ (residual vector quantization)
- WavLM distillation loss (training only, not needed)
- Streaming convolution state

**Estimated Effort**: 3-4 weeks

**Pros**:
- ✅ Unified GGUF ecosystem
- ✅ Quantization benefits
- ✅ Exact Moshi compatibility

**Cons**:
- ❌ Large effort
- ❌ Complex architecture
- ❌ Codec is not bottleneck

**Verdict**: Option 1 (Keep Mimi) - don't fix what's not broken

### 5.2 LM Architecture Options

#### Current: Temporal + DepFormer

**Temporal Transformer**:
```python
# Input: [batch, seq_len, dim]
# - seq_len includes text tokens + audio tokens (all codebooks interleaved)
# - Causal attention with KV caching
# - RoPE positional encodings
# - SwiGLU feedforward
# - RMSNorm

class TemporalTransformer:
    def forward(self, x, cache):
        x = self.attention(x, cache)  # Temporal modeling
        x = self.ffn(x)
        return x
```

**DepFormer** (Depth Transformer):
```python
# Input: [batch, seq_len, 24, dim]  # 24 = num codebooks × 3
# Processes inter-codebook dependencies
# NOT causal (can see all codebooks at once)

class DepFormer:
    def forward(self, x):
        # x shape: [B, T, 24, D]
        x_flat = rearrange(x, 'b t c d -> b t (c d)')
        x = self.transformer(x_flat)  # Cross-codebook attention
        return rearrange(x, 'b t (c d) -> b t c d', c=24)
```

**GGML Port Strategy**:
```c
// Temporal: Standard transformer (well-supported in ggml)
struct ggml_tensor * temporal_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * inp,  // [seq_len, dim]
    struct llama_kv_cache * cache
) {
    // Use existing llama.cpp transformer logic
    struct ggml_tensor * cur = llama_build_graph(ctx, inp, cache);
    return cur;
}

// DepFormer: Custom but straightforward
struct ggml_tensor * depformer_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * inp  // [seq_len, 24, dim]
) {
    // Reshape to [seq_len, 24*dim]
    struct ggml_tensor * flat = ggml_reshape_2d(ctx, inp, 24*dim, seq_len);

    // Standard transformer on flattened
    struct ggml_tensor * out = transformer_block(ctx, flat);

    // Reshape back
    return ggml_reshape_3d(ctx, out, dim, 24, seq_len);
}
```

**Effort**: Moderate - leverages existing llama.cpp graph building

#### Alternative: Monolithic Transformer

**Idea**: Flatten all codebooks, process with single transformer

**Pros**:
- Simpler architecture
- Standard llama.cpp code

**Cons**:
- Different from Moshi (compatibility issues)
- Likely worse quality (DepFormer is important)

**Verdict**: Keep Temporal + DepFormer - proven architecture

### 5.3 Streaming vs. Batch Processing

#### Moshi: Real-time Streaming

**Requirements**:
- 80ms latency budget per frame
- Maintain KV cache across frames
- Process incremental input
- Output exactly 1920 samples every 80ms

**Implementation**:
```rust
// Current server loop (12.5Hz)
loop {
    tokio::time::sleep(Duration::from_millis(80)).await;

    // Get new tokens (if any)
    let new_tokens = state_machine.next_tokens();

    // Run LM inference
    let audio_codes = lm.step(new_tokens, &mut cache);

    // Decode to PCM
    let pcm = mimi.decode(&audio_codes);

    // Send to client
    send_audio_frame(pcm).await;
}
```

**ggml Adaptation**:
```c
// Single-step inference with KV cache
struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
llama_decode(ctx, batch);  // Updates KV cache internally
const float * logits = llama_get_logits(ctx);
```

**Key**: llama.cpp already supports streaming via KV cache

#### OuteTTS: Batch Generation

**Approach**: Generate entire sequence, then post-process

**Issues for real-time**:
- No incremental output
- Full text must be known upfront
- No timing control

**Verdict**: Need streaming - use Moshi approach

### 5.4 Voice Conditioning Options

#### Current: Cross-Attention

**Implementation** (in LM transformer):
```python
class TransformerBlock:
    def forward(self, x, voice_emb):
        # Self-attention
        x = x + self.self_attn(x)

        # Cross-attention to voice
        x = x + self.cross_attn(x, voice_emb)

        # FFN
        x = x + self.ffn(x)
        return x
```

**Voice embeddings**:
- Pre-computed from reference audio
- Shape: [1, voice_dim] (e.g., 1024)
- Cached for each speaker

**ggml Implementation**:
```c
// Cross-attention in ggml
struct ggml_tensor * cross_attn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,        // [seq, dim]
    struct ggml_tensor * voice_emb // [1, dim]
) {
    // Q from x, K/V from voice_emb
    struct ggml_tensor * Q = ggml_mul_mat(ctx, Wq, x);
    struct ggml_tensor * K = ggml_mul_mat(ctx, Wk, voice_emb);
    struct ggml_tensor * V = ggml_mul_mat(ctx, Wv, voice_emb);

    // Attention: softmax(QK^T / √d) V
    struct ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);  // K^T Q
    scores = ggml_scale(ctx, scores, 1.0/sqrtf(dim));
    scores = ggml_soft_max(ctx, scores);
    return ggml_mul_mat(ctx, V, scores);
}
```

**Effort**: Low - standard attention operation

#### Alternative: Conditioning Tokens

**Idea**: Prepend voice tokens to input sequence

**Pros**:
- No architecture changes
- Standard llama.cpp

**Cons**:
- Less control
- Worse quality (no dedicated cross-attn)

**Verdict**: Keep cross-attention - quality matters

### 5.5 STT Integration Strategies

#### Option 1: whisper.cpp as Library

**Architecture**:
```
┌────────────────────────────────────────────────┐
│            Moshi with whisper.cpp              │
│                                                 │
│  Audio Stream (16kHz)                          │
│       │                                         │
│       ▼                                         │
│  ┌─────────────┐                               │
│  │ Mel Spec    │ (via whisper.cpp)             │
│  │ Preprocessor│                                │
│  └──────┬──────┘                                │
│         │                                        │
│         ▼                                        │
│  ┌─────────────┐                               │
│  │  Whisper    │ (GGUF model)                  │
│  │  Encoder    │                                │
│  └──────┬──────┘                                │
│         │                                        │
│         ▼                                        │
│  ┌─────────────┐                               │
│  │  Whisper    │                                │
│  │  Decoder    │                                │
│  └──────┬──────┘                                │
│         │                                        │
│         ▼                                        │
│    Text Tokens ──► Moshi LM ──► TTS           │
│                                                 │
└────────────────────────────────────────────────┘
```

**FFI Binding** (Rust ↔ C):
```rust
// In Moshi Rust server
use whisper_cpp_sys::*;  // Generated bindings

pub struct WhisperSTT {
    ctx: *mut whisper_context,
    params: whisper_full_params,
}

impl WhisperSTT {
    pub fn transcribe(&mut self, audio: &[f32]) -> String {
        unsafe {
            whisper_full(self.ctx, self.params, audio.as_ptr(), audio.len() as i32);
            let n_segments = whisper_full_n_segments(self.ctx);
            // Extract text from segments
        }
    }
}
```

**Streaming Considerations**:
- Whisper processes 30-second chunks
- Need sliding window for continuous audio
- Trade-off: latency vs. context

**Estimated Effort**: 1-2 weeks

**Pros**:
- ✅ Best STT quality available
- ✅ GGUF models ready
- ✅ Proven architecture
- ✅ Active development

**Cons**:
- ⚠️ Latency (~1-2 seconds for 30s chunk)
- ⚠️ Not truly real-time
- ⚠️ Another dependency

#### Option 2: Streaming STT (Qwen2.5-Omni style)

**Idea**: Use multimodal LLM with audio input

**Architecture**:
```
Audio Stream → Audio Encoder → LLM → Text Tokens
```

**Available Models**:
- Qwen2.5-Omni-3B (GGUF available)
- Ultravox (GGUF available)

**Pros**:
- ✅ Can be more real-time
- ✅ Unified LLM approach
- ✅ Already in llama.cpp

**Cons**:
- ⚠️ Less tested than Whisper
- ⚠️ Larger models
- ⚠️ Different preprocessing

**Verdict**: Option 1 (whisper.cpp) - proven, high quality

---

## 6. Integration Strategies

### 6.1 Recommended Architecture: Hybrid ggml + Native Codec

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATED MOSHI                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │ Audio Input  │ (24kHz PCM)                               │
│  │ (User mic)   │                                            │
│  └──────┬───────┘                                           │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐                                           │
│  │ whisper.cpp  │ (Optional STT)                            │
│  │ GGUF model   │                                            │
│  └──────┬───────┘                                           │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────┐                  │
│  │    GGML-based LM (7B params)         │                  │
│  │                                       │                  │
│  │  ┌────────────────┐  ┌─────────────┐│                  │
│  │  │   Temporal     │  │  DepFormer  ││                  │
│  │  │  Transformer   │──┤  (24 levels)││                  │
│  │  │   (ggml)       │  │   (ggml)    ││                  │
│  │  └────────────────┘  └─────────────┘│                  │
│  │                                       │                  │
│  │  Voice Conditioning (cross-attn)    │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│                 ▼                                            │
│         Audio Codes (8 × 12.5Hz)                           │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────────────────────────────┐                  │
│  │    Mimi Codec (Rust/Candle)          │                  │
│  │                                       │                  │
│  │  - SEANet decoder                    │                  │
│  │  - 8 RVQ codebooks                   │                  │
│  │  - Streaming state                   │                  │
│  └──────────────┬───────────────────────┘                  │
│                 │                                            │
│                 ▼                                            │
│  ┌──────────────┐                                           │
│  │ PCM Output   │ (24kHz, 1920 samples/frame)              │
│  │ (Speaker)    │                                            │
│  └──────────────┘                                           │
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │  Rust Server (axum)                  │                  │
│  │  - WebSocket handling                │                  │
│  │  - Frame timing (12.5Hz)             │                  │
│  │  - State management                  │                  │
│  └──────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation Phases

#### Phase 1: ggml LM Core (3-4 weeks)

**Week 1-2: Transformer Port**
- [ ] Define ggml graph for Temporal Transformer
- [ ] Implement KV cache management
- [ ] Add cross-attention for voice conditioning
- [ ] Test single-step inference

**Week 3: DepFormer Port**
- [ ] Implement 24-level depth transformer
- [ ] Handle reshape operations
- [ ] Integrate with Temporal output

**Week 4: Integration & Testing**
- [ ] Create C API for Rust FFI
- [ ] Load GGUF weights (convert from MLX)
- [ ] Benchmark vs. MLX
- [ ] Tune quantization

**Deliverable**: Working ggml-based LM callable from Rust

#### Phase 2: Server Integration (1-2 weeks)

**Week 5: FFI Bridge**
- [ ] Create Rust bindings for ggml LM
- [ ] Implement streaming interface
- [ ] Integrate with existing server loop
- [ ] Add voice profile loading

**Week 6: Testing**
- [ ] End-to-end TTS testing
- [ ] Quality comparison (MLX vs ggml)
- [ ] Performance profiling
- [ ] Multi-speaker testing

**Deliverable**: Full TTS server with ggml backend

#### Phase 3: STT Addition (1-2 weeks)

**Week 7: whisper.cpp Integration**
- [ ] Create Rust bindings
- [ ] Implement mel spectrogram preprocessing
- [ ] Add sliding window for streaming
- [ ] Tune latency/quality trade-off

**Week 8: Full-Duplex Testing**
- [ ] Test STT → LM → TTS pipeline
- [ ] Optimize latency
- [ ] Handle overlapping speech
- [ ] User testing

**Deliverable**: Complete STT+TTS system

#### Phase 4: Optimization (2-3 weeks)

**Week 9-10: Performance**
- [ ] Optimize FFT (use vDSP or kissfft)
- [ ] Tune Metal shaders
- [ ] Reduce memory usage
- [ ] Improve batching

**Week 11: Quality**
- [ ] A/B testing vs. MLX baseline
- [ ] Fix any quality regressions
- [ ] Tune quantization levels
- [ ] Speaker similarity testing

**Deliverable**: Production-ready system

### 6.3 Conversion Pipeline: MLX → GGUF

**Challenge**: Moshi models currently in MLX format, need GGUF

**Approach**:
```python
# convert_moshi_to_gguf.py

import mlx.core as mx
import numpy as np
import struct
from pathlib import Path

def convert_mlx_to_gguf(mlx_model_path, output_path):
    # 1. Load MLX weights
    weights = mx.load(mlx_model_path)

    # 2. Create GGUF header
    with open(output_path, 'wb') as f:
        # Magic: GGUF
        f.write(struct.pack('I', 0x46554747))

        # Version: 3
        f.write(struct.pack('I', 3))

        # Metadata
        write_metadata(f, {
            'general.architecture': 'moshi',
            'general.name': 'moshi-mlx-2b',
            'moshi.context_length': 3000,
            'moshi.embedding_length': 4096,
            # ... more metadata
        })

        # 3. Write tensors
        for name, tensor in weights.items():
            # Convert MLX → NumPy → GGUF
            np_array = np.array(tensor)
            write_tensor(f, name, np_array, dtype='f16')

    print(f"Converted {mlx_model_path} → {output_path}")

# Run conversion
convert_mlx_to_gguf(
    'moshi_mlx_2b/model.safetensors',
    'moshi-2b-f16.gguf'
)
```

**Quantization**:
```bash
# Use llama.cpp quantizer
./llama-quantize \
    moshi-2b-f16.gguf \
    moshi-2b-q8_0.gguf \
    q8_0

# Test different levels
for quant in q4_0 q4_k_m q5_k_m q6_k q8_0; do
    ./llama-quantize moshi-2b-f16.gguf moshi-2b-$quant.gguf $quant
    # Run quality tests
    ./test_quality moshi-2b-$quant.gguf
done
```

**Estimated Effort**: 3-5 days (initial), ongoing for testing

---

## 7. Performance Analysis

### 7.1 Latency Budget Breakdown

**Target**: 80ms per frame (12.5Hz)

| Component | Current (MLX) | Target (ggml) | Budget |
|-----------|---------------|---------------|--------|
| **LM (Temporal)** | 80ms | 70ms | 50ms |
| **DepFormer** | 15ms | 15ms | 15ms |
| **Mimi Decode** | 10ms | 10ms | 10ms |
| **FFT/Post** | 2ms | 0.5ms | 2ms |
| **Overhead** | 5ms | 2.5ms | 3ms |
| **TOTAL** | **112ms** | **98ms** | **80ms** |

**Status**: Currently **1.4x realtime**, target is **1.0x realtime**

**Optimization Opportunities**:
1. **LM Quantization**: Q8_0 likely 1.5-2x faster
2. **Metal Optimization**: Tune thread groups, reduce kernel launches
3. **FFT**: Use vDSP (5ms → 0.5ms = 4.5ms saved)
4. **Batching**: Process multiple tokens together

**Achievability**: **High** - ggml Metal backend is mature

### 7.2 Memory Usage

**Current MLX Model** (moshi-mlx-2b):
```
Temporal Transformer: 7B params × 2 bytes (f16) = 14GB
DepFormer: 100M params × 2 bytes = 200MB
KV Cache: 3000 ctx × 4096 dim × 2 bytes × 32 layers = 800MB
Total: ~15GB
```

**With Quantization (Q8_0)**:
```
LM: 7B × 1 byte = 7GB
DepFormer: 100M × 1 byte = 100MB
KV Cache: 800MB (keep f16 for quality)
Total: ~8GB
```

**With Aggressive Quantization (Q4_K_M)**:
```
LM: 7B × 0.5 bytes = 3.5GB
Total: ~4.5GB
```

**Conclusion**: Q8_0 fits comfortably on M1 Pro (16GB), Q4_K_M enables 8GB devices

### 7.3 Quality vs. Performance Trade-offs

| Quantization | Size | Speed | Quality | Use Case |
|--------------|------|-------|---------|----------|
| **F16** | 14GB | 1.0x | 100% | Baseline |
| **Q8_0** | 7GB | 1.5x | 99.5% | Recommended |
| **Q6_K** | 5.5GB | 1.8x | 99% | Good balance |
| **Q5_K_M** | 4.8GB | 2.0x | 98% | Memory-constrained |
| **Q4_K_M** | 4GB | 2.2x | 96% | Fast inference |
| **Q4_0** | 3.7GB | 2.5x | 93% | Edge devices |

**Testing Strategy**:
1. Generate 100 test sentences with MLX F16 (ground truth)
2. Quantize and generate same sentences
3. Measure:
   - MOS (Mean Opinion Score) via listening tests
   - Speaker similarity (cosine distance)
   - Word error rate (WER) via STT
   - Mel spectrogram difference

**Quality Threshold**: >98% similarity → Q5_K_M or better

---

## 8. Implementation Roadmap

### 8.1 Milestone Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                   IMPLEMENTATION TIMELINE                    │
│                      (8-10 weeks total)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: ggml LM Core (3-4 weeks)                          │
│  ├─ Week 1-2: Temporal Transformer port                     │
│  ├─ Week 3: DepFormer implementation                        │
│  └─ Week 4: Testing & tuning                                │
│      Deliverable: ggml LM callable from Rust                │
│                                                              │
│  PHASE 2: Server Integration (1-2 weeks)                    │
│  ├─ Week 5: FFI bridge & streaming                          │
│  └─ Week 6: End-to-end testing                              │
│      Deliverable: Full TTS server with ggml                 │
│                                                              │
│  PHASE 3: STT Addition (1-2 weeks)                          │
│  ├─ Week 7: whisper.cpp integration                         │
│  └─ Week 8: Full-duplex testing                             │
│      Deliverable: Complete STT+TTS system                   │
│                                                              │
│  PHASE 4: Optimization (2-3 weeks)                          │
│  ├─ Week 9-10: Performance tuning                           │
│  └─ Week 11: Quality validation                             │
│      Deliverable: Production-ready system                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Resource Requirements

**Team**:
- 1x ML Engineer (ggml/llama.cpp experience)
- 0.5x Systems Engineer (Rust/FFI)
- 0.5x Audio Engineer (testing/validation)

**Hardware**:
- Mac with M1/M2 (16GB+ RAM) for development
- Linux workstation with NVIDIA GPU for cross-platform testing
- Test devices: iPhone, MacBook Air (8GB), etc.

**Dependencies**:
- llama.cpp (fork and modify)
- whisper.cpp (link as library)
- Existing Moshi codebase
- Testing infrastructure

### 8.3 Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Quality regression** | Medium | High | Extensive A/B testing, quality metrics |
| **Performance miss** | Low | High | Early benchmarking, fallback to MLX |
| **DepFormer complexity** | Medium | Medium | Prototype early, validate architecture |
| **FFI bugs** | Medium | Medium | Comprehensive testing, memory safety |
| **Quantization issues** | Low | Medium | Test multiple levels, conservative choice |
| **Timeline slip** | Medium | Low | Phased approach, clear milestones |

**Contingency Plans**:
- If ggml port too complex: Keep MLX, optimize that path
- If performance insufficient: Use hybrid (ggml LM + Metal kernels)
- If quality issues: Fall back to higher quantization (Q8_0 minimum)

---

## 9. Risk Assessment

### 9.1 Technical Risks

#### High Risk: DepFormer Incompatibility

**Issue**: DepFormer is novel, may not map cleanly to ggml

**Indicators**:
- Unusual tensor shapes (4D: batch × time × 24 × dim)
- Non-standard attention patterns
- Tight coupling with Temporal Transformer

**Mitigation**:
1. **Early prototype** (Week 1)
2. Implement simplified version first
3. Validate numerically against MLX
4. If blocked: Implement as custom ggml operation

**Fallback**: Keep DepFormer in MLX, only port Temporal to ggml

#### Medium Risk: FFT Performance

**Issue**: Spectral processing is slow without optimized FFT

**Current**: Naive DFT in llama.cpp tts.cpp (O(n²))

**Solutions**:
1. **vDSP** (Apple Accelerate): Built-in, highly optimized
2. **kissfft**: BSD license, simple integration
3. **pffft**: Fastest, SIMD-optimized

**Test**:
```c
// Benchmark different FFT implementations
for (int impl = 0; impl < N_IMPLS; impl++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        fft_impls[impl](data, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    printf("FFT %s: %.2f ms\n", names[impl], duration_ms);
}
```

**Threshold**: <1ms per 80ms frame (1.25% overhead)

#### Low Risk: Memory Management

**Issue**: ggml context allocation, Rust FFI lifetime management

**Mitigation**:
- Use ggml_allocr for automatic buffer management
- Rust `Box` and `Arc` for cross-language ownership
- Comprehensive testing with Valgrind / AddressSanitizer

### 9.2 Product Risks

#### High Risk: User Experience Regression

**Concern**: Worse quality or latency than current MLX

**Metrics**:
- **Latency**: Must be ≤ 100ms/frame (vs. 112ms current)
- **Quality**: MOS ≥ 4.0/5.0 (listener tests)
- **Similarity**: Speaker embedding distance ≤ 0.05

**Validation**:
- Blind A/B tests (MLX vs. ggml)
- Automated quality suite (run on every commit)
- User beta testing program

#### Medium Risk: Limited Hardware Support

**Concern**: ggml may not work well on all platforms

**Testing Matrix**:
| Platform | Backend | Priority |
|----------|---------|----------|
| Mac M1/M2 | Metal | P0 |
| Mac Intel | Metal | P1 |
| Linux + NVIDIA | CUDA | P1 |
| Linux + AMD | HIP | P2 |
| Linux CPU | AVX2 | P2 |
| Windows | Vulkan | P3 |

**Strategy**: Focus on P0/P1, community support for P2/P3

---

## 10. Recommendations

### 10.1 Immediate Next Steps

1. **Prototype DepFormer in ggml** (3-5 days)
   - Validate architecture feasibility
   - Measure performance
   - Identify blockers early

2. **Convert MLX model to GGUF** (2-3 days)
   - Write conversion script
   - Validate weight accuracy
   - Test quantization levels

3. **Benchmark ggml vs. MLX** (2 days)
   - Same hardware (M1 Pro)
   - Same operations (matrix mul, attention, etc.)
   - Establish performance baseline

4. **Design C API** (1 day)
   - Define Rust FFI interface
   - Plan memory ownership
   - Document calling conventions

**Total**: ~2 weeks for validation phase

### 10.2 Strategic Recommendations

#### For TTS:

**Primary Path**: Hybrid ggml LM + Candle Mimi codec
- **Rationale**: Focus effort on bottleneck (LM), reuse working codec
- **Timeline**: 6-8 weeks to production
- **Risk**: Low-medium

**Alternative Path**: Keep MLX, optimize further
- **Rationale**: If ggml port proves too complex
- **Effort**: 2-3 weeks (tune Metal kernels, quantization)
- **Risk**: Low

#### For STT:

**Primary Path**: Integrate whisper.cpp
- **Rationale**: Best quality, proven, GGUF-ready
- **Timeline**: 1-2 weeks
- **Risk**: Low

**Alternative Path**: Multimodal LLM (Qwen2.5-Omni)
- **Rationale**: Unified architecture, potentially better streaming
- **Timeline**: 2-3 weeks
- **Risk**: Medium (less proven)

#### For Deployment:

**Phase 1**: ggml TTS only (keep MLX STT placeholder)
**Phase 2**: Add whisper.cpp STT
**Phase 3**: Optimize for edge devices (quantization, etc.)

### 10.3 Long-Term Vision

```
┌─────────────────────────────────────────────────────────────┐
│           MOSHI: Unified Speech Foundation Model             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Core LM (GGUF/ggml)                    │    │
│  │  - 7B parameter transformer                         │    │
│  │  - Multi-backend (Metal, CUDA, CPU, Vulkan)       │    │
│  │  - Quantized (Q4-Q8 range)                         │    │
│  │  - Real-time inference (<80ms/frame)               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────┐       ┌──────────────────┐          │
│  │   STT Module     │       │   TTS Module     │          │
│  │  (whisper.cpp)   │       │  (ggml + Mimi)   │          │
│  │  - Streaming     │       │  - Multi-speaker │          │
│  │  - Multi-lingual │       │  - Voice cloning │          │
│  └──────────────────┘       └──────────────────┘          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Deployment Targets                     │    │
│  │  - Desktop (Mac, Linux, Windows)                   │    │
│  │  - Mobile (iOS, Android via GGML mobile)           │    │
│  │  - Edge (Raspberry Pi, NVIDIA Jetson)             │    │
│  │  - Cloud (Docker containers, Kubernetes)          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- ✅ Unified ecosystem (GGUF for all models)
- ✅ Broad hardware support
- ✅ Easy deployment (single binary)
- ✅ Community contributions (llama.cpp ecosystem)
- ✅ Future-proof (new backends added regularly)

---

## Appendices

### A. GGML Operations Quick Reference

See `/Users/ochafik/github/llama.cpp/docs/ops.md` for full list.

**Essential for Moshi**:
- `ggml_mul_mat` - Matrix multiplication
- `ggml_rope` - Rotary embeddings
- `ggml_flash_attn_ext` - Flash attention
- `ggml_rms_norm` - RMS normalization
- `ggml_silu`, `ggml_swiglu` - Activations
- `ggml_conv_1d`, `ggml_conv_transpose_1d` - Convolutions
- `ggml_concat`, `ggml_reshape` - Tensor manipulation

### B. File Locations Reference

**Moshi**:
- TTS: `/Users/ochafik/github/moshi/moshi_mlx/moshi_mlx/models/tts.py`
- LM: `/Users/ochafik/github/moshi/moshi_mlx/moshi_mlx/models/lm.py`
- Codec: `/Users/ochafik/github/moshi/rust/moshi-core/src/mimi.rs`
- Server: `/Users/ochafik/github/moshi/rust/moshi-server/src/main.rs`

**llama.cpp**:
- TTS: `/Users/ochafik/github/llama.cpp/tools/tts/tts.cpp`
- GGML: `/Users/ochafik/github/llama.cpp/ggml/include/ggml.h`
- Server: `/Users/ochafik/github/llama.cpp/tools/server/server.cpp`
- Audio: `/Users/ochafik/github/llama.cpp/tools/mtmd/mtmd-audio.cpp`

### C. Performance Benchmarks

**To be collected**: Comprehensive benchmarks comparing MLX vs. ggml

**Metrics to track**:
- Latency per frame (ms)
- Throughput (frames/second)
- Memory usage (GB)
- Quality scores (MOS, WER, speaker similarity)
- Power consumption (W)

### D. References

1. **Moshi Paper**: [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/Moshi.pdf)
2. **GGML**: [GGML Documentation](https://github.com/ggml-org/ggml)
3. **llama.cpp**: [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
4. **whisper.cpp**: [whisper.cpp GitHub](https://github.com/ggml-org/whisper.cpp)
5. **OuteTTS**: [OuteTTS Models](https://huggingface.co/OuteAI)
6. **WavTokenizer**: [WavTokenizer Paper](https://arxiv.org/abs/2408.16532)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-03
**Next Review**: After prototype phase (2 weeks)
