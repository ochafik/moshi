// Mimi Audio Codec - Native C++ Implementation
// Encoder: raw audio -> discrete tokens (32 codebooks)
// Decoder: discrete tokens -> raw audio
//
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MIMI_H
#define MIMI_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Forward declarations
struct ggml_context;
struct gguf_context;
struct ggml_tensor;

namespace mimi {

// Mimi hyperparameters
struct mimi_hparams {
    int32_t sample_rate = 24000;      // Audio sample rate
    float frame_rate = 12.5f;         // Output frame rate
    int32_t hop_length = 1920;        // Samples per frame (sample_rate / frame_rate)
    int32_t encoder_dim = 512;        // Encoder hidden dimension
    int32_t encoder_layers = 8;       // Transformer layers
    int32_t n_codebooks = 32;         // Number of RVQ codebooks
    int32_t codebook_size = 2048;     // Entries per codebook
    int32_t codebook_dim = 256;       // Codebook embedding dimension

    // SEANet encoder config
    int32_t seanet_channels = 1;      // Input audio channels (mono)
    int32_t seanet_nfilters = 64;     // Initial filter count
    int32_t seanet_ratios[4] = {4, 5, 6, 8};  // Downsampling ratios

    // Derived
    int32_t total_stride() const { return seanet_ratios[0] * seanet_ratios[1] * seanet_ratios[2] * seanet_ratios[3]; }
};

// Encode result: audio tokens for each frame
struct encode_result {
    std::vector<std::vector<int32_t>> tokens;  // tokens[frame][codebook]
    int32_t n_frames;
    float duration;  // Audio duration in seconds
};

// Decode result: raw audio samples
struct decode_result {
    std::vector<float> audio;  // Mono audio samples
    int32_t sample_rate;
};

// Mimi Encoder class
class MimiEncoder {
public:
    MimiEncoder();
    ~MimiEncoder();

    // Load encoder model from GGUF file
    bool load(const std::string & path);

    // Encode raw audio to tokens
    // Input: mono audio samples at 24kHz
    // Output: tokens[frame][codebook] for 32 codebooks
    encode_result encode(const float * audio, int n_samples, bool verbose = false);

    // Convenience: encode from vector
    encode_result encode(const std::vector<float> & audio, bool verbose = false) {
        return encode(audio.data(), audio.size(), verbose);
    }

    // Get model hyperparameters
    const mimi_hparams & get_hparams() const { return hparams_; }

    // Check if model is loaded
    bool is_loaded() const { return loaded_; }

private:
    // SEANet encoder layers
    struct seanet_layer {
        // Residual block
        struct ggml_tensor * res_conv1_w = nullptr;
        struct ggml_tensor * res_conv1_b = nullptr;
        struct ggml_tensor * res_conv2_w = nullptr;
        struct ggml_tensor * res_conv2_b = nullptr;
        // Downsampling
        struct ggml_tensor * down_conv_w = nullptr;
        struct ggml_tensor * down_conv_b = nullptr;
    };

    // Transformer layer
    struct transformer_layer {
        // Attention
        struct ggml_tensor * attn_norm_w = nullptr;
        struct ggml_tensor * attn_norm_b = nullptr;
        struct ggml_tensor * attn_in_proj = nullptr;
        struct ggml_tensor * attn_out_proj = nullptr;
        struct ggml_tensor * attn_layer_scale = nullptr;
        // FFN
        struct ggml_tensor * ffn_norm_w = nullptr;
        struct ggml_tensor * ffn_norm_b = nullptr;
        struct ggml_tensor * ffn_linear1 = nullptr;
        struct ggml_tensor * ffn_linear2 = nullptr;
        struct ggml_tensor * ffn_layer_scale = nullptr;
    };

    mimi_hparams hparams_;
    bool loaded_ = false;

    // SEANet encoder weights
    struct ggml_tensor * init_conv_w_ = nullptr;
    struct ggml_tensor * init_conv_b_ = nullptr;
    std::vector<seanet_layer> seanet_layers_;
    struct ggml_tensor * final_conv_w_ = nullptr;
    struct ggml_tensor * final_conv_b_ = nullptr;

    // Downsample layer (connects SEANet to transformer)
    struct ggml_tensor * downsample_conv_w_ = nullptr;

    // Encoder transformer
    std::vector<transformer_layer> transformer_layers_;

    // Quantizer
    struct ggml_tensor * quantizer_input_proj_[2] = {nullptr, nullptr};  // rvq_first, rvq_rest
    // Codebooks stored as EMA statistics (embedding = embedding_sum / cluster_usage)
    std::vector<struct ggml_tensor *> embedding_sums_;  // 32 codebooks embedding_sum
    std::vector<struct ggml_tensor *> cluster_usages_;  // 32 codebooks cluster_usage

    // GGML context
    struct ggml_context * ctx_ = nullptr;
    struct gguf_context * gguf_ctx_ = nullptr;

    // Internal methods
    void seanet_encode(const float * audio, int n_samples, std::vector<float> & out);
    void transformer_encode(std::vector<float> & latents, int n_frames);
    void quantize(const std::vector<float> & latents, int n_frames, std::vector<std::vector<int32_t>> & tokens);
};

// Mimi Decoder class (for audio generation)
class MimiDecoder {
public:
    MimiDecoder();
    ~MimiDecoder();

    // Load decoder model from GGUF file
    bool load(const std::string & path);

    // Decode tokens to raw audio
    // Input: tokens[frame][codebook] for 32 codebooks
    // Output: mono audio samples at 24kHz
    decode_result decode(const std::vector<std::vector<int32_t>> & tokens, bool verbose = false);

    // Get model hyperparameters
    const mimi_hparams & get_hparams() const { return hparams_; }

    bool is_loaded() const { return loaded_; }

private:
    mimi_hparams hparams_;
    bool loaded_ = false;

    // Decoder weights (similar structure to encoder, but reversed)
    // TODO: Add decoder tensors

    struct ggml_context * ctx_ = nullptr;
    struct gguf_context * gguf_ctx_ = nullptr;
};

// Utility functions

// Load audio from WAV file (returns mono 24kHz float samples)
std::vector<float> load_wav(const std::string & path, int target_sample_rate = 24000);

// Save audio to WAV file
bool save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate = 24000);

} // namespace mimi

#endif // MIMI_H
