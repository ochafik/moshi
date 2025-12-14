// Moshi Speech-to-Text Native Implementation
// Header file with public API
//
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MOSHI_STT_H
#define MOSHI_STT_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Forward declarations
struct ggml_context;
struct gguf_context;
struct ggml_tensor;

namespace moshi {

// Model hyperparameters
struct stt_hparams {
    int32_t dim = 2048;           // embedding_length
    int32_t n_layers = 16;        // block_count
    int32_t n_heads = 16;         // attention.head_count
    int32_t n_ctx = 750;          // context_length
    int32_t n_vocab_text = 8000;  // vocab_size_text
    int32_t n_vocab_audio = 2048; // vocab_size_audio
    int32_t n_codebooks = 32;     // audio_codebooks
    float rope_base = 100000.0f;  // rope.freq_base

    int32_t head_dim() const { return dim / n_heads; }
};

// Transcription result
struct stt_result {
    std::vector<int32_t> all_tokens;    // All output tokens (including padding)
    std::vector<int32_t> text_tokens;   // Valid text tokens only (>=4)
    int n_frames;                       // Number of input frames processed
};

// STT Model class
class STTModel {
public:
    STTModel();
    ~STTModel();

    // Load model from GGUF file
    bool load(const std::string & path);

    // Transcribe audio tokens to text tokens
    // Input: frames[t][q] where t=time frame, q=codebook index
    // Returns transcription result with token IDs
    stt_result transcribe(const std::vector<std::vector<int32_t>> & frames, bool verbose = false);

    // Get model hyperparameters
    const stt_hparams & get_hparams() const { return hparams_; }

    // Check if model is loaded
    bool is_loaded() const { return loaded_; }

private:
    struct layer {
        struct ggml_tensor * norm1_alpha;
        struct ggml_tensor * norm2_alpha;
        struct ggml_tensor * attn_in_proj;
        struct ggml_tensor * attn_out_proj;
        struct ggml_tensor * ffn_linear_in;
        struct ggml_tensor * ffn_linear_out;
    };

    stt_hparams hparams_;
    bool loaded_ = false;

    // Embeddings
    std::vector<struct ggml_tensor *> emb_;
    struct ggml_tensor * text_emb_ = nullptr;

    // Transformer layers
    std::vector<layer> layers_;

    // Output
    struct ggml_tensor * out_norm_alpha_ = nullptr;
    struct ggml_tensor * text_linear_ = nullptr;

    // GGML context
    struct ggml_context * ctx_ = nullptr;
    struct gguf_context * gguf_ctx_ = nullptr;
};

// Utility functions

// Parse audio tokens from JSON file
// Input format: {"audio_tokens": [[codebook0], [codebook1], ...]}
// Returns: frames[t][q] transposed format
std::vector<std::vector<int32_t>> parse_audio_tokens_json(const std::string & json_content, int n_codebooks = 32);

// Parse audio tokens from JSON file path
std::vector<std::vector<int32_t>> load_audio_tokens(const std::string & path, int n_codebooks = 32);

} // namespace moshi

#endif // MOSHI_STT_H
