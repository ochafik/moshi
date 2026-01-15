# Deep Analysis: whisper.cpp & OuteTTS Integration in llama.cpp

**Author**: Analysis for Moshi TTS/STT Integration
**Date**: 2025-12-03
**Purpose**: Understand proven integration patterns from llama.cpp's audio implementations

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [whisper.cpp Architecture (via llama.cpp mtmd)](#whisper-architecture)
3. [OuteTTS Integration Analysis (PR #10784)](#outetts-integration)
4. [Integration Patterns Comparison](#integration-patterns)
5. [Lessons for Moshi Integration](#lessons-for-moshi)
6. [Recommended Architecture](#recommended-architecture)
7. [Implementation Roadmap](#implementation-roadmap)

---

## 1. Executive Summary

### Key Findings

**whisper.cpp in llama.cpp:**
- **Not a separate dependency**: llama.cpp implements Whisper-compatible audio preprocessing in its multimodal library (`libmtmd`)
- **Architecture**: Encoder-only audio → embeddings → LLM for transcription
- **Performance**: Multi-threaded mel spectrogram computation, optimized FFT
- **Integration**: Unified API alongside vision models

**OuteTTS (PR #10784):**
- **Merged**: December 18, 2024 (commit `0bf2d10c5`)
- **Architecture**: Two-stage pipeline (Text→Codes LLM, Codes→Audio Vocoder)
- **Server Design**: Two separate llama-server instances, not a dedicated endpoint
- **Client-side**: Final audio synthesis (IRFFT) done client-side

### Critical Insights for Moshi

1. **No dedicated audio endpoints**: llama.cpp reuses `/completion` and `/embeddings` with enhancements
2. **Dual-server pattern**: Viable for two-stage pipelines
3. **FFT placement**: Audio post-processing can be client-side (Python/C++)
4. **Multimodal API design**: `libmtmd` provides excellent pattern for unified TTS/STT API
5. **Streaming gap**: Neither implementation supports real-time audio streaming (batch only)

---

## 2. Whisper Architecture (via llama.cpp mtmd)

### 2.1 Overall Structure

**Location**: `/Users/ochafik/github/llama.cpp/tools/mtmd/`

**Key Components**:
```
libmtmd (Multimodal Library)
├── mtmd.cpp/h           # Core API, tokenization
├── mtmd-audio.cpp/h     # Whisper-compatible audio preprocessing
├── mtmd-helper.cpp/h    # High-level helper functions
├── clip.cpp/h           # Vision & audio encoder implementation
└── mtmd-cli.cpp         # Command-line example
```

### 2.2 Audio Preprocessing Pipeline

**File**: `/Users/ochafik/github/llama.cpp/tools/mtmd/mtmd-audio.cpp`

#### Constants (Whisper-Compatible)
```cpp
#define WHISPER_SAMPLE_RATE 16000   // 16kHz input
#define WHISPER_N_FFT       400     // FFT size (25ms @ 16kHz)
#define WHISPER_HOP_LENGTH  160     // Hop size (10ms)
#define WHISPER_CHUNK_SIZE  30      // 30-second chunks
#define WHISPER_N_MEL       128     // 128 mel bins (or 80)
```

#### Processing Steps

**1. Audio Loading** (via miniaudio):
```cpp
// Support: WAV, MP3, FLAC
// Output: PCM F32 mono @ 16kHz
mtmd_bitmap * mtmd_bitmap_init_from_audio(
    size_t n_samples,
    const float * data
);
```

**2. Padding**:
```cpp
// Reflective padding: 200 samples at start/end
// Zero padding: 30 seconds (480,000 samples) at end
n_samples_padded = n_samples + 2*pad + WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SIZE;
```

**3. FFT Computation** (lines 78-124):
```cpp
// Cooley-Tukey FFT algorithm
static void fft(float* in, int N, float* out) {
    // Recursive divide-and-conquer
    // O(N log N) complexity
    // Uses pre-cached sin/cos tables
}
```

**Optimization**: Global sine/cosine cache
```cpp
std::map<int, std::vector<float>> g_fft_sin;
std::map<int, std::vector<float>> g_fft_cos;
```

**4. Mel Spectrogram** (lines 126-189):
```cpp
static void log_mel_spectrogram_worker_thread(
    const float * samples,
    const int n_samples,
    const int frame_size,     // 400
    const int frame_step,     // 160
    const int n_threads,
    const int ith,            // Thread ID
    const whisper_filters & filters,
    whisper_mel & mel
) {
    // For each frame (multi-threaded)
    for (int i = ith; i < n_frames; i += n_threads) {
        // Apply Hann window
        for (int j = 0; j < frame_size; j++) {
            fft_in[j] = hann[j] * samples[i*frame_step + j];
        }

        // Compute FFT
        fft(frame_size, fft_in, fft_out);

        // Power spectrum
        for (int j = 0; j < frame_size/2; j++) {
            float re = fft_out[2*j + 0];
            float im = fft_out[2*j + 1];
            power[j] = re*re + im*im;
        }

        // Apply mel filterbank
        for (int j = 0; j < n_mel; j++) {
            float sum = 0.0f;
            for (int k = 0; k < frame_size/2; k++) {
                sum += power[k] * filters.data[j*frame_size/2 + k];
            }
            mel.data[j*n_frames + i] = sum;
        }
    }
}
```

**5. Log Scaling & Normalization** (lines 270-279):
```cpp
// Log compression
float mmax = -1e20;
for (float val : mel.data) {
    mmax = std::max(mmax, log10f(std::max(val, 1e-10f)));
}

// Normalize: clamp to [mmax-8, mmax], then scale to [-4, 4]
for (float & val : mel.data) {
    float log_val = log10f(std::max(val, 1e-10f));
    log_val = std::max(log_val, mmax - 8.0f);
    val = (log_val + 4.0f) / 4.0f;
}
```

**6. Chunking**:
```cpp
// Split into 3000-frame chunks (30 seconds × 100 fps)
// Each chunk: [3000, 128] array
```

### 2.3 Mel Filterbank Pre-calculation

**File**: `/Users/ochafik/github/llama.cpp/tools/mtmd/mtmd-audio.cpp` (lines 358-767)

**Pre-calculated filters** (128 mel bins):
- Stored as static float array `whisper_mel_filter_128`
- Computed offline using mel scale formula
- Reduces runtime computation significantly

**Alternative**: 80 mel bins (`whisper_mel_filter_80`) for some models

### 2.4 C API Design

**File**: `/Users/ochafik/github/llama.cpp/tools/mtmd/mtmd.h`

#### Core Types
```cpp
// Opaque context type
typedef struct mtmd_context mtmd_context;

// Bitmap type (unified for images/audio)
typedef struct mtmd_bitmap mtmd_bitmap;

// Input text with tokens
struct mtmd_input_text {
    const char * text;
    llama_token * tokens;
    int32_t n_tokens;
    bool add_special;
    bool parse_special;
};

// Input chunk (polymorphic)
enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
};
```

#### Initialization
```cpp
struct mtmd_context_params {
    bool use_gpu;           // Enable GPU acceleration
    bool print_timings;     // Performance metrics
    int n_threads;          // CPU threads
    ggml_log_level verbosity;
    const char * media_marker;  // Default: "<__media__>"
};

mtmd_context_params mtmd_context_params_default();

mtmd_context * mtmd_init_from_file(
    const char * mmproj_fname,           // Encoder model path
    const llama_model * text_model,      // Text model for embedding check
    const mtmd_context_params ctx_params
);

void mtmd_free(mtmd_context * ctx);
```

#### Audio Input
```cpp
// Create bitmap from audio samples
mtmd_bitmap * mtmd_bitmap_init_from_audio(
    size_t n_samples,
    const float * data  // PCM F32 mono
);

// Load from file (auto-detect WAV/MP3/FLAC)
mtmd_bitmap * mtmd_helper_bitmap_init_from_file(
    mtmd_context * ctx,
    const char * fname
);

void mtmd_bitmap_free(mtmd_bitmap * bitmap);
```

#### Tokenization & Encoding
```cpp
// Tokenize text + media into chunks
int32_t mtmd_tokenize(
    mtmd_context * ctx,
    mtmd_input_chunks * output,      // Out: tokenized chunks
    const mtmd_input_text * text,    // Text with <__media__> markers
    const mtmd_bitmap ** bitmaps,    // Array of images/audio
    size_t n_bitmaps
);

// Encode a single chunk to embeddings
int32_t mtmd_encode_chunk(
    mtmd_context * ctx,
    const mtmd_input_chunk * chunk
);

// Get output embeddings
float * mtmd_get_output_embd(mtmd_context * ctx);
```

#### Helper API
```cpp
// High-level evaluation (encode + decode)
int32_t mtmd_helper_eval_chunks(
    mtmd_context * ctx,           // Multimodal context
    llama_context * lctx,         // Text generation context
    const mtmd_input_chunks * chunks,
    llama_pos n_past,             // Current position
    llama_seq_id seq_id,          // Sequence ID
    int32_t n_batch,              // Batch size
    bool logits_last,             // Only last token logits
    llama_pos * new_n_past        // Out: updated position
);
```

### 2.5 Integration with llama-server

**File**: `/Users/ochafik/github/llama.cpp/tools/server/server.cpp`

#### Initialization (lines ~2450-2475)
```cpp
// Global multimodal context
mtmd_context * mctx = nullptr;

// Initialize if mmproj model provided
if (!params_base.mmproj.empty()) {
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params_base.mmproj_use_gpu;
    mparams.print_timings = true;
    mparams.n_threads = params_base.cpuparams.n_threads;
    mparams.verbosity = GGML_LOG_LEVEL_INFO;

    std::string mmproj_path = params_base.mmproj;
    mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);

    if (!mctx) {
        LOG_ERR("Failed to load multimodal projector\n");
        return 1;
    }
}
```

#### Endpoint Support

**Chat Completions** (`/v1/chat/completions`):
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this:"},
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
            ]
        }
    ]
}
```

**Embeddings** (`/v1/embeddings`):
```json
{
    "input": "audio file path or base64",
    "encoding_format": "float"
}
```

#### Audio Processing in Server
```cpp
// Decode base64 audio
std::string audio_data = base64_decode(audio_base64);

// Load as bitmap
mtmd_bitmap * audio_bmp = mtmd_bitmap_init_from_audio(
    n_samples,
    reinterpret_cast<const float*>(audio_data.data())
);

// Add to processing queue
slot.media_bitmaps.push_back(audio_bmp);
```

### 2.6 Memory Management

**RAII Pattern**:
```cpp
// Custom deleters for C types
struct mtmd_context_deleter {
    void operator()(mtmd_context * ctx) {
        mtmd_free(ctx);
    }
};

using mtmd_context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;
```

**Backend Management**:
```cpp
struct clip_ctx {
    ggml_context_ptr ctx_data;               // Tensor storage
    std::vector<ggml_backend_t> backend_ptrs; // GPU/CPU backends
    ggml_backend_sched_ptr sched;            // Backend scheduler
    ggml_backend_t backend;                  // Primary (GPU)
    ggml_backend_t backend_cpu;              // Fallback (CPU)
};
```

### 2.7 Performance Optimizations

**Multi-threading**:
```cpp
// Parallel mel spectrogram computation
std::vector<std::thread> workers;
for (int i = 0; i < n_threads; i++) {
    workers.emplace_back(log_mel_spectrogram_worker_thread,
                         samples, n_samples, frame_size, frame_step,
                         n_threads, i, filters, std::ref(mel));
}
for (auto & w : workers) w.join();
```

**Pre-cached Data**:
- Mel filterbanks (static arrays)
- FFT trig functions (global maps)
- Hann window (computed once per frame size)

**Batching**:
```cpp
// Batch audio tokens for GPU efficiency
for (size_t i = 0; i < audio_tokens.size(); i++) {
    common_batch_add(batch, audio_tokens[i], pos_base + i, seq_ids, false);
}
llama_decode(ctx, batch);
```

---

## 3. OuteTTS Integration (PR #10784)

### 3.1 Timeline & History

**Merged**: December 18, 2024 (commit `0bf2d10c5`)
**Follow-up PRs**:
- #11186 - Guide tokens support (prevent hallucinations)
- #11235 - Python audio synthesis function
- #12048 - Speaker file support
- #12398 - Output file option
- #13713 - Fix n_ubatch, make WavTokenizer cache-less

**Commit History** (relevant):
```
0bf2d10c5 - tts : add OuteTTS support (#10784)
6390a998b - tts : add guide tokens support (#11186)
c43af9276 - tts: add speaker file support (#12048)
b5486956f - added rudimentary support for outetts v0.3 500m and 1b models
8a1d206f1 - tts : fix n_ubatch + make WavTokenizer cache-less (#13713)
```

### 3.2 Architecture Deep Dive

#### Two-Stage Pipeline

**Stage 1: Text-to-Codes (TTC)**
```
Input: "Hello world" + speaker profile
       ↓
    [Text Normalization]
       ↓
    [SentencePiece Tokenization]
       ↓
    [LLM Generation] (OuteTTS-0.2-500M)
       ↓
Output: Audio codes [151672, 151740, 151636, ...]
```

**Stage 2: Codes-to-Speech (CTS)**
```
Input: Audio codes (offset to 0-based)
       ↓
    [WavTokenizer Decoder] (embeddings mode)
       ↓
Output: Embeddings [magnitude + phase spectrogram]
       ↓
    [IRFFT + Overlap-Add] (client-side)
       ↓
Output: PCM audio @ 24kHz
```

### 3.3 File Structure

**Location**: `/Users/ochafik/github/llama.cpp/tools/tts/`

```
tools/tts/
├── tts.cpp                    # Main C++ implementation (1094 lines)
├── tts-outetts.py             # Python server-based version (300 lines)
├── convert_pt_to_hf.py        # PyTorch → HuggingFace converter
├── README.md                  # Documentation
└── CMakeLists.txt             # Build configuration
```

### 3.4 Implementation Analysis: tts.cpp

#### Text Normalization (lines 365-417)

```cpp
static std::string process_text(const std::string & text, outetts_version version) {
    // 1. Number to words
    std::string processed = replace_numbers_with_words(text);
    //    "I have 3 cats" → "I have three cats"

    // 2. Lowercase
    std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);

    // 3. Replace special chars with spaces
    processed = std::regex_replace(processed, std::regex(R"([-_/,\.\\])"), " ");

    // 4. Remove non-alphabetic (except spaces)
    processed = std::regex_replace(processed, std::regex(R"([^a-z\s])"), "");

    // 5. Collapse multiple spaces
    processed = std::regex_replace(processed, std::regex(R"(\s+)"), " ");

    // 6. Trim
    processed = std::regex_replace(processed, std::regex(R"(^\s+|\s+$)"), "");

    // 7. Insert separator tokens
    std::string sep = (version == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed = std::regex_replace(processed, std::regex(R"(\s)"), sep);

    return processed;
}
```

**Number Conversion** (lines 332-363):
```cpp
static std::string replace_numbers_with_words(const std::string & input_text) {
    static const std::vector<std::string> ones = {
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine"
    };
    static const std::vector<std::string> tens = {
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    };
    // ... conversion logic
}
```

#### Speaker Profile Format (lines 419-550)

**JSON Structure**:
```json
{
    "speaker_profile": {
        "text": "Hello this is a test",
        "words": [
            {
                "word": "hello",
                "duration": 0.56,
                "codes": [257, 740, 636, ...]  // ~6-8 codes per word
            },
            {
                "word": "this",
                "duration": 0.24,
                "codes": [173, 289, 941, ...]
            }
        ]
    }
}
```

**Prompt Construction** (lines 669-693):
```cpp
// Build prompt with speaker profile
std::string prompt = "<|im_start|>";

// Add text section
prompt += "<|text_start|>";
for (const auto & word : speaker_words) {
    prompt += word.text + "<|text_sep|>";  // or <|space|> for v0.3
}
prompt += user_text;  // Append new text to synthesize
prompt += "<|text_end|>";

// Add audio section (speaker reference)
prompt += "<|audio_start|>";
for (const auto & word : speaker_words) {
    // Duration marker
    prompt += word.text + "<|t_" + format_duration(word.duration) + "|>";

    // Audio codes
    prompt += "<|code_start|>";
    for (int code : word.codes) {
        prompt += "<|" + std::to_string(code) + "|>";
    }
    prompt += "<|code_end|>";
}

// User input section (to be generated)
for (const auto & word : user_words) {
    prompt += word.text + "<|t_0.00|><|code_start|>";
    // LLM will generate codes here
}
```

#### Guide Tokens (lines 884-893)

**Purpose**: Prevent hallucinations (LLM generating wrong words)

**Implementation**:
```cpp
// Prepare guide tokens from user text
if (params.vocoder.use_guide_tokens) {
    guide_tokens = prepare_guide_tokens(vocab, prompt_clean, tts_version);
}

// During generation
while (n_decode <= n_predict) {
    llama_token new_token_id = common_sampler_sample(smpl, ctx_ttc, i_batch);

    // Override sampled token if it should be guided
    if (!guide_tokens.empty() && next_token_uses_guide_token) {
        new_token_id = guide_tokens[0];
        guide_tokens.erase(guide_tokens.begin());
    }

    common_sampler_accept(smpl, new_token_id, true);
}
```

**Result**: Forces LLM to generate exact text tokens, only allows creativity in audio codes

#### Audio Code Extraction (lines 1002-1013)

```cpp
// Filter tokens to audio codes only
// OuteTTS audio codes: 151672-155772 (4096 codes)
std::vector<llama_token> codes;
std::copy_if(generated_tokens.begin(), generated_tokens.end(),
             std::back_inserter(codes),
             [](llama_token t) { return t >= 151672 && t <= 155772; });

// Convert to 0-based codes
for (auto & token : codes) {
    token -= 151672;  // Now in range [0, 4095]
}
```

#### Vocoder Processing (lines 1019-1043)

```cpp
// Batch all codes for parallel processing
llama_batch batch = llama_batch_init(codes.size(), 0, 1);

for (size_t i = 0; i < codes.size(); ++i) {
    common_batch_add(batch, codes[i], i, {0}, true);
}

// Encode in embeddings mode
if (llama_encode(ctx_cts, batch) != 0) {
    LOG_ERR("Vocoder encoding failed\n");
    return 1;
}

// Get embeddings (magnitude + phase)
const int n_embd = llama_n_embd(model_cts);
const float * embd = llama_get_embeddings(ctx_cts);

// Convert to audio
std::vector<float> audio = embd_to_audio(embd, codes.size(), n_embd, n_thread);
```

#### Spectral Processing (lines 200-280)

**Parameters**:
```cpp
const int n_fft = 1280;   // FFT size
const int n_hop = 320;    // Hop length (4× upsampling from 75 Hz)
const int n_win = 1280;   // Window size
const int n_pad = 480;    // Padding
const int n_sr = 24000;   // Sample rate
```

**IRFFT Implementation** (lines 126-163):
```cpp
static void irfft(int n, const float * inp_cplx, float * out_real) {
    // Naive DFT implementation O(n²)
    // TODO: Replace with optimized FFT (kissfft, pffft, vDSP)

    for (int k = 0; k < n; ++k) {
        float sum_r = 0.0f;
        for (int j = 0; j < n/2 + 1; ++j) {
            float theta = 2.0f * M_PI * j * k / n;
            float re = inp_cplx[2*j + 0];  // Real part
            float im = inp_cplx[2*j + 1];  // Imaginary part
            sum_r += re * cosf(theta) - im * sinf(theta);
        }
        out_real[k] = sum_r / n;
    }
}
```

**Overlap-Add (Fold)** (lines 165-198):
```cpp
static void fold(
    const std::vector<float> & data,
    int n_out,
    int n_win,
    int n_hop,
    int n_pad,
    std::vector<float> & output
) {
    // Initialize output with zeros
    output.resize(n_out, 0.0f);
    std::vector<float> window_env(n_out, 0.0f);

    // Overlap-add reconstruction
    for (size_t i = 0; i < data.size() / n_win; ++i) {
        int offset = i * n_hop - n_pad;

        for (int j = 0; j < n_win; ++j) {
            int out_idx = offset + j;
            if (out_idx >= 0 && out_idx < n_out) {
                output[out_idx] += data[i*n_win + j];
                window_env[out_idx] += hann_window[j];
            }
        }
    }

    // Normalize by window envelope
    for (int i = 0; i < n_out; ++i) {
        if (window_env[i] > 1e-8f) {
            output[i] /= window_env[i];
        }
    }
}
```

**Complete Pipeline** (lines 201-280):
```cpp
static std::vector<float> embd_to_audio(
    const float * embd,
    const int n_codes,
    const int n_embd,
    const int n_thread
) {
    // 1. Separate magnitude and phase
    std::vector<float> mag(n_codes * n_embd/2);
    std::vector<float> phase(n_codes * n_embd/2);

    for (int i = 0; i < n_codes; i++) {
        for (int j = 0; j < n_embd/2; j++) {
            mag[j*n_codes + i] = exp(embd[j*n_codes + i]);  // Log mag → mag
            phase[j*n_codes + i] = embd[(j + n_embd/2)*n_codes + i];
        }
    }

    // 2. Create complex spectrogram
    std::vector<float> S(n_codes * n_embd);  // Complex (real, imag pairs)
    for (int i = 0; i < n_codes * n_embd/2; i++) {
        S[2*i + 0] = mag[i] * cosf(phase[i]);  // Real
        S[2*i + 1] = mag[i] * sinf(phase[i]);  // Imaginary
    }

    // 3. IRFFT + windowing
    std::vector<float> frames(n_codes * n_fft);
    for (int i = 0; i < n_codes; i++) {
        irfft(n_fft, S.data() + i*n_embd, frames.data() + i*n_fft);

        // Apply Hann window
        for (int j = 0; j < n_fft; j++) {
            frames[i*n_fft + j] *= hann_window[j];
        }
    }

    // 4. Overlap-add
    int n_out = (n_codes - 1) * n_hop + n_win;
    std::vector<float> audio;
    fold(frames, n_out, n_win, n_hop, n_pad, audio);

    return audio;
}
```

### 3.5 Python Server Implementation

**File**: `/Users/ochafik/github/llama.cpp/tools/tts/tts-outetts.py`

#### Architecture

**Two-Server Pattern**:
```python
# Usage
python tts-outetts.py http://localhost:8020 http://localhost:8021 "Hello world"

# Server 1: LLM (port 8020)
./llama-server -m outetts-0.2-0.5B-q8_0.gguf --port 8020

# Server 2: Vocoder (port 8021)
./llama-server -m wavtokenizer-large-75-f16.gguf --port 8021 \
               --embeddings --pooling none
```

#### Text-to-Codes Request (lines 244-263)

```python
def generate_audio_codes(llm_url, prompt, prefix, suffix):
    response = requests.post(f"{llm_url}/completion", json={
        "prompt": [prefix + prompt, *suffix],
        "n_predict": 1024,
        "cache_prompt": True,
        "return_tokens": True,     # NEW: Get raw token IDs
        "samplers": ["top_k"],
        "top_k": 16,
        "seed": 1003,
        "stream": False
    })

    data = response.json()
    tokens = data["tokens"]

    # Filter to audio codes (151672-155772)
    audio_codes = [t for t in tokens if 151672 <= t <= 155772]

    # Offset to 0-based
    audio_codes = [t - 151672 for t in audio_codes]

    return audio_codes
```

#### Codes-to-Embeddings Request (lines 268-285)

```python
def codes_to_embeddings(decoder_url, audio_codes):
    response = requests.post(f"{decoder_url}/embeddings", json={
        "input": audio_codes,
        "encoding_format": "float"
    })

    data = response.json()

    # Response format with pooling=none:
    # [{"embedding": [[...], [...], ...]}]
    # Each inner array is one token's embedding

    embeddings = data[0]["embedding"]
    return embeddings
```

#### Audio Synthesis (lines 287-315)

```python
def embd_to_audio(embd, n_codes, n_embd):
    """Convert embeddings to audio waveform"""
    import numpy as np
    from scipy.fft import irfft

    n_fft = 1280
    n_hop = 320
    n_win = 1280
    n_pad = 480

    # Separate magnitude and phase
    mag = np.exp(embd[:n_embd//2])     # Log mag → mag
    phase = embd[n_embd//2:]           # Phase

    # Create complex spectrogram
    S = mag * np.exp(1j * phase)

    # IRFFT
    frames = []
    for i in range(n_codes):
        frame = irfft(S[:, i], n=n_fft)
        # Apply Hann window
        frame *= np.hanning(n_fft)
        frames.append(frame)

    # Overlap-add
    audio = overlap_add(frames, n_hop, n_pad)

    return audio
```

**Note**: The Python version is **incomplete** in the repository - the `embd_to_audio` function was added in PR #11235 after initial merge.

### 3.6 Server Enhancements

#### Enhancement 1: `return_tokens` Parameter

**Commit**: `0e70ba686`
**File**: `/Users/ochafik/github/llama.cpp/tools/server/server.cpp`

**Change**: Added ability to return raw token IDs from `/completion`

```cpp
// Request parameter
struct completion_request {
    // ... existing fields
    bool return_tokens = false;  // NEW
};

// Response JSON
if (params.return_tokens) {
    res_json["tokens"] = slot.generated_token_ids;
}
```

**Benefit**: Allows extracting audio codes without parsing text

#### Enhancement 2: `pooling=none` for Embeddings

**Commit**: `152610eda`
**File**: `/Users/ochafik/github/llama.cpp/tools/server/server.cpp`

**Change**: Return per-token embeddings instead of pooled

```cpp
// Start server with --pooling none
./llama-server --embeddings --pooling none

// Response format changes:
// Before (pooled): {"embedding": [0.1, 0.2, ...]}  // Single vector
// After (none):    {"embedding": [[0.1, ...], [0.2, ...], ...]}  // Per-token
```

**Benefit**: Vocoder needs embeddings for each audio code separately

### 3.7 WavTokenizer Architecture

**Added to llama.cpp** in `src/llama.cpp`

**Components**:
```cpp
struct llm_arch_wavtokenizer_dec {
    // 1. Token embedding
    struct ggml_tensor * token_embd;

    // 2. Conv1D downsampling
    struct ggml_tensor * conv1d_0_w;
    struct ggml_tensor * conv1d_0_b;

    // 3. ConvNeXT blocks (multiple layers)
    struct wavtokenizer_convnext_layer {
        ggml_tensor * dw_conv_w, * dw_conv_b;  // Depthwise conv
        ggml_tensor * gn_w, * gn_b;            // Group norm
        ggml_tensor * pw1_w, * pw1_b;          // Pointwise conv 1
        ggml_tensor * pw2_w, * pw2_b;          // Pointwise conv 2
        ggml_tensor * gamma;                   // Learnable scaling
    };

    // 4. PosNet blocks (ResNet-style)
    struct wavtokenizer_posnet_layer {
        ggml_tensor * conv_w, * conv_b;        // Convolution
        ggml_tensor * attn_q, * attn_k, * attn_v;  // Multi-head attention
        ggml_tensor * gn_w, * gn_b;            // Group norm
    };

    // 5. Output layer
    struct ggml_tensor * out_w;
    struct ggml_tensor * out_b;
};
```

**Graph Building** (simplified):
```cpp
struct ggml_cgraph * build_wavtokenizer_dec(
    llama_context & lctx,
    const llama_batch & batch
) {
    // 1. Embed tokens
    cur = ggml_get_rows(ctx, token_embd, batch.token);

    // 2. Conv1D
    cur = ggml_conv_1d(ctx, conv1d_0_w, cur, ...);
    cur = ggml_add(ctx, cur, conv1d_0_b);

    // 3. ConvNeXT blocks
    for (auto & layer : convnext_layers) {
        inpL = cur;

        // Depthwise conv
        cur = ggml_conv_1d_dw(ctx, layer.dw_conv_w, cur, ...);
        cur = ggml_group_norm(ctx, cur, ...);

        // Pointwise convs with GELU
        cur = ggml_conv_1d(ctx, layer.pw1_w, cur, ...);
        cur = ggml_gelu(ctx, cur);
        cur = ggml_conv_1d(ctx, layer.pw2_w, cur, ...);

        // Residual + scaling
        cur = ggml_scale(ctx, cur, layer.gamma);
        cur = ggml_add(ctx, cur, inpL);
    }

    // 4. PosNet blocks
    for (auto & layer : posnet_layers) {
        // Multi-head attention
        cur = multi_head_attention(ctx, cur, layer);
        // ResNet connection
        cur = ggml_add(ctx, cur, inpL);
    }

    // 5. Output projection
    cur = ggml_mul_mat(ctx, out_w, cur);
    cur = ggml_add(ctx, cur, out_b);

    return gf;
}
```

### 3.8 Configuration & Usage

#### Command-Line Options

```bash
# Standalone TTS tool
llama-tts \
    -m outetts-0.2-0.5B-q8_0.gguf \      # Text-to-codes model
    -mv wavtokenizer-large-75-f16.gguf \  # Vocoder model
    -p "Hello world" \                     # Text to synthesize
    -o output.wav \                        # Output file
    --speaker-file speaker.json \          # Optional speaker profile
    --use-guide-tokens \                   # Prevent hallucinations
    -t 8                                   # CPU threads
```

#### Model Sources

**Text-to-Codes Models**:
- `OuteAI/OuteTTS-0.2-500M-GGUF` (Q2_K - Q8_0, F16)
- `OuteAI/OuteTTS-0.3-500M-GGUF` (newer version)
- `OuteAI/OuteTTS-0.3-1B-GGUF` (larger model)

**Vocoder Models**:
- `OuteAI/WavTokenizer-large-75-GGUF`
- `OuteAI/WavTokenizer-medium-320-GGUF`

#### Performance Characteristics

**Model Sizes**:
- OuteTTS-0.2-500M: ~500MB (Q8_0), ~250MB (Q4_0)
- WavTokenizer-large: ~250MB (F16)

**Generation Speed** (M1 Pro):
- ~54 seconds of audio per 4096-token context
- ~75 audio codes per second (WavTokenizer frame rate)
- Text-to-codes: ~2-3 seconds for short sentence
- Codes-to-audio: ~1-2 seconds

**Quality**:
- Supports English (v0.2) and English+French (some models)
- v0.3 adds punctuation support
- Speaker cloning from reference audio

---

## 4. Integration Patterns Comparison

### 4.1 Audio Input (STT-style) vs. Audio Output (TTS-style)

| Aspect | Audio Input (mtmd) | Audio Output (OuteTTS) |
|--------|-------------------|------------------------|
| **Direction** | Audio → Text | Text → Audio |
| **Preprocessing** | Mel spectrogram (server) | Text normalization (client/server) |
| **Postprocessing** | None (text output) | IRFFT + fold (client-side) |
| **Model Count** | 1 (encoder + LLM) | 2 (LLM + vocoder) |
| **Server Instances** | 1 (unified) | 2 (separate) |
| **Streaming** | Batch chunks | No streaming |
| **Endpoints** | `/v1/chat/completions` | `/completion` + `/embeddings` |
| **Integration** | Deep (libmtmd) | Shallow (enhanced endpoints) |

### 4.2 API Design Patterns

#### Pattern 1: Dedicated Library (mtmd)

**Characteristics**:
- Separate library (`libmtmd`)
- C API with opaque types
- Unified handling of multiple modalities
- Deep integration with server

**Pros**:
- Clean abstraction
- Reusable across tools
- Well-tested
- Extensible

**Cons**:
- Larger codebase
- More complex build
- Tighter coupling

**Best for**: Core functionality, multiple tools using same code

#### Pattern 2: Enhanced Endpoints (OuteTTS)

**Characteristics**:
- Reuse existing endpoints
- Generic enhancements (`return_tokens`, `pooling=none`)
- Client-side coordination
- Minimal server changes

**Pros**:
- Lightweight integration
- Flexible architecture
- Independent scaling
- Easy to experiment

**Cons**:
- Two-server complexity
- Network latency
- Client-side complexity
- No atomic transactions

**Best for**: Prototyping, optional features, distributed systems

### 4.3 FFT Placement Strategies

#### Client-Side (OuteTTS Approach)

```python
# Python client handles IRFFT
def embd_to_audio(embd):
    S = mag * np.exp(1j * phase)
    audio = scipy.fft.irfft(S)
    return audio
```

**Pros**:
- Offload computation from server
- Flexibility (different FFT libraries)
- Easy to iterate

**Cons**:
- Every client needs implementation
- Duplicated code
- Potential quality variations

#### Server-Side (Potential mtmd Approach)

```cpp
// Server returns ready-to-play audio
std::vector<float> audio = vocoder_decode(codes);
return wav_encode(audio);
```

**Pros**:
- Consistent quality
- Simpler clients
- Centralized optimization

**Cons**:
- Server CPU load
- Larger responses (PCM vs embeddings)
- Less flexible

### 4.4 Memory Management Patterns

#### RAII with Custom Deleters

```cpp
struct mtmd_context_deleter {
    void operator()(mtmd_context * ctx) { mtmd_free(ctx); }
};

using mtmd_context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;
```

**Benefits**:
- Automatic cleanup
- Exception-safe
- Prevents leaks
- Easy to reason about lifetimes

#### Backend Scheduler Pattern

```cpp
struct clip_ctx {
    std::vector<ggml_backend_t> backend_ptrs;
    ggml_backend_sched_ptr sched;

    // Scheduler automatically chooses GPU/CPU
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_sched_graph_compute(sched, gf);
};
```

**Benefits**:
- Heterogeneous execution
- Automatic memory placement
- Optimal backend selection
- Load balancing

---

## 5. Lessons for Moshi Integration

### 5.1 Critical Takeaways

#### 1. Don't Reinvent Audio Preprocessing

**Lesson**: llama.cpp's mel spectrogram implementation is production-ready

**For Moshi STT**:
- Reuse `mtmd-audio.cpp` preprocessing
- Already Whisper-compatible
- Multi-threaded, optimized
- Pre-calculated filters

**Action**: Link against `libmtmd` or copy audio preprocessing code

#### 2. Two-Server Pattern is Viable

**Lesson**: OuteTTS successfully uses dual servers

**For Moshi TTS**:
- Separate LM server (text → codes)
- Separate codec server (codes → audio)
- Client orchestrates the pipeline

**Consideration**: Moshi needs streaming, OuteTTS doesn't - may need different approach

#### 3. Client-Side Post-Processing is Acceptable

**Lesson**: OuteTTS does IRFFT client-side successfully

**For Moshi**:
- Codec can return embeddings
- Client does final audio synthesis
- Reduces server complexity

**Trade-off**: Every client needs audio synthesis code vs. server CPU load

#### 4. Generic Endpoint Enhancements > Dedicated Endpoints

**Lesson**: `return_tokens` and `pooling=none` benefit all models

**For Moshi**:
- Add `/v1/audio/speech` if truly needed
- Otherwise, enhance `/completion` and `/embeddings`
- Make changes broadly useful

#### 5. Guide Tokens Improve Quality

**Lesson**: Forcing LLM to generate correct text tokens prevents hallucinations

**For Moshi TTS**:
- Implement similar guidance mechanism
- Separate text generation from prosody generation
- Quality improvement > strict end-to-end

#### 6. RAII + Custom Deleters for C/Rust FFI

**Lesson**: Clean C API with C++ wrappers works well

**For Moshi**:
```rust
// Rust FFI pattern
pub struct MoshiContext(*mut moshi_sys::moshi_context);

impl Drop for MoshiContext {
    fn drop(&mut self) {
        unsafe { moshi_sys::moshi_free(self.0); }
    }
}
```

### 5.2 Streaming Considerations

**Key Gap**: Neither implementation supports real-time audio streaming

**Moshi Requirement**: 12.5Hz frame rate (80ms latency)

**Implications**:
1. **Can't use OuteTTS pattern directly** - it's batch-only
2. **Need incremental generation** - token by token with KV cache
3. **Need frame-based API** - not full-sequence generation
4. **Codec state management** - Mimi maintains state across frames

**Solution**: Hybrid approach
- Use mtmd patterns for API structure
- Add streaming mode to generation loop
- Maintain codec state between frames
- Return audio incrementally

### 5.3 Performance Optimization Lessons

#### Multi-threading is Essential

**mtmd approach**:
```cpp
// Parallel mel spectrogram computation
for (int ith = 0; ith < n_threads; ith++) {
    workers.push_back(std::thread(worker_fn, ith));
}
```

**For Moshi**:
- Parallelize mel spectrogram (STT input)
- Parallelize IRFFT (TTS output)
- Use thread pool (don't spawn per-request)

#### Pre-calculation Saves Time

**mtmd approach**:
- Static mel filterbanks
- Global FFT trig tables

**For Moshi**:
- Pre-calculate Mimi codec filters
- Cache voice embeddings
- Pre-compute Hann windows

#### Batching is Critical

**OuteTTS approach**:
```cpp
// Batch all audio codes
llama_batch batch = llama_batch_init(codes.size(), 0, 1);
for (size_t i = 0; i < codes.size(); i++) {
    common_batch_add(batch, codes[i], i, {0}, true);
}
```

**For Moshi**:
- Batch audio tokens across time
- Batch multiple codebooks together
- Minimize decode calls

---

## 6. Recommended Architecture for Moshi

### 6.1 Hybrid Approach: Best of Both Worlds

```
┌─────────────────────────────────────────────────────────────────┐
│                    MOSHI + llama.cpp INTEGRATION                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  STT Pipeline (Whisper-compatible)                     │    │
│  │                                                         │    │
│  │  Audio Input (16kHz PCM)                              │    │
│  │       ↓                                                 │    │
│  │  [libmtmd Audio Preprocessing]                        │    │
│  │       ├─ Mel spectrogram (multi-threaded)            │    │
│  │       ├─ Pre-cached filterbanks                       │    │
│  │       └─ Chunking (3000 frames)                       │    │
│  │       ↓                                                 │    │
│  │  [Whisper Encoder] (ggml GGUF)                        │    │
│  │       ↓                                                 │    │
│  │  [Moshi LM] (cross-attention with encoder)           │    │
│  │       ↓                                                 │    │
│  │  Text Output                                           │    │
│  │                                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  TTS Pipeline (Moshi Architecture)                     │    │
│  │                                                         │    │
│  │  Text Input                                            │    │
│  │       ↓                                                 │    │
│  │  [Text Normalization] (client/server)                 │    │
│  │       ↓                                                 │    │
│  │  [Moshi LM - ggml] (Temporal + DepFormer)            │    │
│  │       ├─ Cross-attention voice conditioning           │    │
│  │       ├─ Guide tokens (prevent hallucinations)        │    │
│  │       └─ Streaming generation (12.5Hz)                │    │
│  │       ↓                                                 │    │
│  │  Audio Codes (8 codebooks)                            │    │
│  │       ↓                                                 │    │
│  │  [Mimi Codec - Candle/MLX] (keep current impl)       │    │
│  │       ↓                                                 │    │
│  │  PCM Output (24kHz)                                    │    │
│  │                                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Server Integration (llama-server + Rust)              │    │
│  │                                                         │    │
│  │  Endpoints:                                            │    │
│  │    - /v1/audio/transcriptions (STT)                   │    │
│  │    - /v1/audio/speech (TTS)                           │    │
│  │    - /v1/chat/completions (full-duplex dialog)        │    │
│  │                                                         │    │
│  │  State Management:                                     │    │
│  │    - Per-connection KV cache                          │    │
│  │    - Codec state (Mimi streaming)                     │    │
│  │    - Voice profile cache                              │    │
│  │                                                         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Breakdown

#### STT: Use libmtmd Pattern

**C API** (`moshi_stt.h`):
```cpp
typedef struct moshi_stt_context moshi_stt_context;

// Initialize with Whisper encoder
moshi_stt_context * moshi_stt_init(
    const char * whisper_model_path,
    const llama_model * text_model,
    const moshi_stt_params params
);

// Transcribe audio chunk
int moshi_stt_transcribe(
    moshi_stt_context * ctx,
    const float * audio,      // 16kHz PCM
    size_t n_samples,
    char * output_text,       // Out: transcription
    size_t max_len
);

void moshi_stt_free(moshi_stt_context * ctx);
```

**Implementation Strategy**:
1. Copy `mtmd-audio.cpp` preprocessing
2. Load Whisper encoder GGUF
3. Integrate with Moshi LM
4. Return text transcription

**Effort**: 1-2 weeks

#### TTS: Hybrid ggml LM + Native Codec

**C API** (`moshi_tts.h`):
```cpp
typedef struct moshi_tts_context moshi_tts_context;

struct moshi_tts_params {
    const char * lm_model_path;      // ggml GGUF
    const char * codec_model_path;   // Candle/MLX (current)
    const char * voice_profile;      // JSON
    float temperature;
    int n_threads;
};

moshi_tts_context * moshi_tts_init(const moshi_tts_params params);

// Streaming generation
int moshi_tts_step(
    moshi_tts_context * ctx,
    const char * text_chunk,  // Incremental text
    float * audio_output,     // Out: 1920 samples (80ms @ 24kHz)
    size_t * n_samples        // Out: actual samples generated
);

void moshi_tts_free(moshi_tts_context * ctx);
```

**Implementation Strategy**:
1. Port Temporal Transformer to ggml (3 weeks)
2. Port DepFormer to ggml (1 week)
3. Create FFI wrapper for Rust (1 week)
4. Keep Mimi codec in Candle/Rust (no change)
5. Implement streaming state management (1 week)

**Effort**: 6-8 weeks

### 6.3 Server Endpoints

#### STT Endpoint: `/v1/audio/transcriptions`

**Request** (OpenAI-compatible):
```json
POST /v1/audio/transcriptions
Content-Type: multipart/form-data

file: audio.wav
model: whisper-large-v3
language: en
```

**Response**:
```json
{
    "text": "Hello, how are you today?"
}
```

**Implementation**:
```rust
async fn audio_transcriptions(
    State(state): State<ServerState>,
    multipart: Multipart
) -> Result<Json<TranscriptionResponse>> {
    // 1. Parse multipart form data
    let audio_data = extract_audio_file(multipart).await?;

    // 2. Decode audio (WAV/MP3/FLAC)
    let pcm_samples = decode_audio(&audio_data)?;

    // 3. Call moshi_stt_transcribe via FFI
    let text = unsafe {
        moshi_stt_transcribe(
            state.stt_ctx,
            pcm_samples.as_ptr(),
            pcm_samples.len()
        )
    };

    // 4. Return transcription
    Ok(Json(TranscriptionResponse { text }))
}
```

#### TTS Endpoint: `/v1/audio/speech`

**Request** (OpenAI-compatible):
```json
POST /v1/audio/speech
{
    "model": "moshi-tts",
    "input": "Hello world",
    "voice": "alloy",
    "response_format": "wav"
}
```

**Response**: Binary WAV file

**Streaming Version**:
```json
POST /v1/audio/speech
{
    "model": "moshi-tts",
    "input": "Hello world",
    "voice": "alloy",
    "response_format": "pcm",
    "stream": true
}
```

**Response**: Chunked PCM data (80ms chunks)

**Implementation**:
```rust
async fn audio_speech(
    State(state): State<ServerState>,
    Json(req): Json<SpeechRequest>
) -> Result<Response> {
    // 1. Normalize text
    let text = normalize_text(&req.input);

    // 2. Load voice profile
    let voice_profile = load_voice_profile(&req.voice)?;

    if req.stream {
        // Streaming response
        let stream = async_stream::stream! {
            // Generate audio in 80ms chunks
            for chunk in text.split_sentences() {
                let mut audio_chunk = vec![0.0f32; 1920];
                let mut n_samples = 0;

                unsafe {
                    moshi_tts_step(
                        state.tts_ctx,
                        chunk.as_ptr(),
                        audio_chunk.as_mut_ptr(),
                        &mut n_samples
                    );
                }

                yield Ok::<_, Error>(audio_chunk.into());
            }
        };

        Ok(Response::new(Body::wrap_stream(stream)))
    } else {
        // Batch response (generate full audio)
        let audio = generate_full_audio(state.tts_ctx, &text)?;
        let wav_data = encode_wav(&audio, 24000)?;
        Ok(Response::new(Body::from(wav_data)))
    }
}
```

#### Full-Duplex Endpoint: `/v1/chat/completions` (WebSocket)

**Upgrade to WebSocket**:
```
GET /v1/chat/completions
Upgrade: websocket
Connection: Upgrade
```

**Message Format**:
```json
// Client → Server (audio input)
{
    "type": "audio_input",
    "data": [0.1, 0.2, ...],  // PCM F32 samples
    "sample_rate": 16000
}

// Server → Client (transcription)
{
    "type": "transcription",
    "text": "Hello"
}

// Server → Client (audio output)
{
    "type": "audio_output",
    "data": [0.1, 0.2, ...],  // PCM F32 samples
    "sample_rate": 24000
}

// Server → Client (LM response)
{
    "type": "text_response",
    "text": "Hello! How can I help?"
}
```

### 6.4 State Management

#### Per-Connection State

```rust
struct ConnectionState {
    // STT state
    stt_buffer: AudioRingBuffer,  // Accumulate audio for chunking
    stt_ctx: *mut moshi_stt_context,

    // TTS state
    tts_ctx: *mut moshi_tts_context,
    tts_queue: VecDeque<String>,  // Text to synthesize

    // LM state
    kv_cache: KvCache,            // Conversation history
    seq_id: u32,                  // Sequence ID

    // Voice state
    voice_profile: VoiceProfile,
    voice_embd: Vec<f32>,         // Cached embeddings
}
```

#### Codec Streaming State

**Challenge**: Mimi codec maintains internal state for continuity

**Solution**:
```rust
struct MimiStreamingState {
    encoder_state: Vec<f32>,  // Conv state
    decoder_state: Vec<f32>,  // Conv state
    prev_codes: [i32; 8],     // Previous codebook values
}

impl MoshiTtsContext {
    pub fn step_with_state(
        &mut self,
        text: &str,
        state: &mut MimiStreamingState
    ) -> Vec<f32> {
        // Generate codes
        let codes = self.lm_generate(text);

        // Decode with state
        let audio = self.mimi.decode_streaming(&codes, state);

        audio
    }
}
```

---

## 7. Implementation Roadmap

### Phase 1: STT Integration (2-3 weeks)

**Week 1: Audio Preprocessing**
- [ ] Copy `mtmd-audio.cpp` to Moshi
- [ ] Create `moshi_audio.h` C API
- [ ] Test mel spectrogram computation
- [ ] Benchmark vs. mtmd (ensure parity)

**Week 2: Whisper Integration**
- [ ] Download Whisper GGUF models
- [ ] Create Rust FFI bindings
- [ ] Implement `moshi_stt_init()`
- [ ] Implement `moshi_stt_transcribe()`
- [ ] Unit tests

**Week 3: Server Integration**
- [ ] Add `/v1/audio/transcriptions` endpoint
- [ ] Handle multipart form data
- [ ] Audio format detection & decoding
- [ ] Response formatting
- [ ] Integration tests

**Deliverable**: Working STT endpoint

### Phase 2: TTS ggml Port (5-6 weeks)

**Weeks 4-5: Temporal Transformer**
- [ ] Define ggml graph structure
- [ ] Port attention layers
- [ ] Port feedforward layers
- [ ] Implement RoPE
- [ ] Cross-attention for voice
- [ ] Test numerically vs. MLX

**Week 6: DepFormer**
- [ ] Port depth transformer
- [ ] Handle 24-level architecture
- [ ] Integrate with Temporal output
- [ ] Test numerical accuracy

**Week 7: Model Conversion**
- [ ] Write MLX → GGUF converter
- [ ] Convert Moshi 2B model
- [ ] Validate weight loading
- [ ] Test quantization (Q8_0, Q4_K_M)

**Week 8: Integration**
- [ ] Create C API (`moshi_tts.h`)
- [ ] Rust FFI bindings
- [ ] Connect ggml LM → Candle Mimi
- [ ] Test end-to-end generation

**Week 9: Optimization**
- [ ] Benchmark latency
- [ ] Optimize batch size
- [ ] Tune quantization
- [ ] Profile bottlenecks

**Deliverable**: Working TTS with ggml LM

### Phase 3: Server Streaming (2-3 weeks)

**Week 10: Streaming Infrastructure**
- [ ] Implement `moshi_tts_step()`
- [ ] Frame-based generation loop
- [ ] State management (codec, KV cache)
- [ ] Buffer management

**Week 11: Endpoint Implementation**
- [ ] Add `/v1/audio/speech` (batch)
- [ ] Add streaming mode
- [ ] Chunked response encoding
- [ ] Error handling

**Week 12: WebSocket Full-Duplex**
- [ ] Upgrade HTTP → WebSocket
- [ ] Bidirectional message handling
- [ ] STT + TTS coordination
- [ ] Latency optimization

**Deliverable**: Production-ready streaming TTS/STT

### Phase 4: Testing & Optimization (2-3 weeks)

**Week 13: Quality Testing**
- [ ] A/B tests (ggml vs. MLX)
- [ ] MOS (Mean Opinion Score)
- [ ] Speaker similarity metrics
- [ ] WER (Word Error Rate) for STT

**Week 14: Performance Tuning**
- [ ] Optimize FFT (vDSP integration)
- [ ] Multi-threading tuning
- [ ] Memory profiling
- [ ] Reduce latency

**Week 15: Documentation**
- [ ] API documentation
- [ ] Integration guide
- [ ] Performance benchmarks
- [ ] Example code

**Deliverable**: Production-ready system

**Total**: 12-15 weeks (3-4 months)

---

## Conclusion

The analysis of whisper.cpp (via libmtmd) and OuteTTS integration reveals proven patterns for audio processing in llama.cpp:

**Key Patterns**:
1. **Unified multimodal API** (libmtmd) - excellent for STT
2. **Dual-server architecture** (OuteTTS) - viable but not ideal for streaming
3. **Client-side post-processing** - acceptable trade-off
4. **Generic endpoint enhancements** - better than dedicated endpoints
5. **RAII + custom deleters** - clean C/Rust FFI pattern

**For Moshi**:
- **STT**: Adopt libmtmd pattern directly (2-3 weeks)
- **TTS**: Hybrid ggml LM + native Mimi codec (6-8 weeks)
- **Server**: Streaming WebSocket for full-duplex (2-3 weeks)

**Total Implementation**: 12-15 weeks to production-ready system

The recommended architecture balances:
- **Reuse**: Leverage llama.cpp's audio preprocessing
- **Focus**: Port only the bottleneck (LM) to ggml
- **Pragmatism**: Keep working components (Mimi codec)
- **Quality**: Real-time streaming with <100ms latency

This approach provides the best effort/benefit ratio while maintaining Moshi's core full-duplex capabilities.
