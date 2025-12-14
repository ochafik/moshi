// Mimi Audio Codec - Native C++ Implementation
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#include "mimi.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>

#include "ggml.h"
#include "gguf.h"

namespace mimi {

// ============================================================================
// Math primitives
// ============================================================================

static inline float elu(float x, float alpha = 1.0f) {
    return x >= 0 ? x : alpha * (expf(x) - 1.0f);
}

static void layer_norm(float * out, const float * in, const float * gamma, const float * beta,
                       int dim, float eps = 1e-5f) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = in[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

static void matvec(float * out, const float * W, const float * in, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += W[o * in_dim + i] * in[i];
        }
        out[o] = sum;
    }
}

static void softmax(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Causal Conv1d: output[t] = sum_k(input[t-k] * weight[k])
// Weight shape: [in_channels, kernel_size, out_channels] (GGUF layout)
static void causal_conv1d(float * out, const float * in, const float * weight, const float * bias,
                          int in_channels, int out_channels, int kernel_size, int stride,
                          int in_len, int & out_len) {
    // Causal padding: pad left with (kernel_size - 1) zeros
    int pad_left = kernel_size - 1;
    int padded_len = in_len + pad_left;

    out_len = (padded_len - kernel_size) / stride + 1;

    for (int t = 0; t < out_len; t++) {
        int in_start = t * stride - pad_left;

        for (int oc = 0; oc < out_channels; oc++) {
            float sum = bias ? bias[oc] : 0.0f;

            for (int k = 0; k < kernel_size; k++) {
                int in_pos = in_start + k;
                if (in_pos >= 0 && in_pos < in_len) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        // Weight layout: [in_channels, kernel_size, out_channels]
                        sum += weight[(ic * kernel_size + k) * out_channels + oc] * in[in_pos * in_channels + ic];
                    }
                }
                // else: pad with zeros (implicit)
            }

            out[t * out_channels + oc] = sum;
        }
    }
}

// Apply ELU activation in-place
static void apply_elu(float * data, int n, float alpha = 1.0f) {
    for (int i = 0; i < n; i++) {
        data[i] = elu(data[i], alpha);
    }
}

// Find nearest codebook entry (L2 distance)
static int find_nearest(const float * query, const float * codebook, int codebook_size, int dim) {
    int best_idx = 0;
    float best_dist = 1e30f;

    for (int i = 0; i < codebook_size; i++) {
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = query[d] - codebook[i * dim + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

// ============================================================================
// MimiEncoder implementation
// ============================================================================

MimiEncoder::MimiEncoder() = default;

MimiEncoder::~MimiEncoder() {
    if (ctx_) ggml_free(ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
}

bool MimiEncoder::load(const std::string & path) {
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx_,
    };

    gguf_ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_ctx_) {
        fprintf(stderr, "Error: failed to load GGUF file: %s\n", path.c_str());
        return false;
    }

    // Read hyperparameters
    auto get_u32 = [&](const char * key, int32_t def) {
        int64_t id = gguf_find_key(gguf_ctx_, key);
        return id >= 0 ? gguf_get_val_u32(gguf_ctx_, id) : def;
    };
    auto get_f32 = [&](const char * key, float def) {
        int64_t id = gguf_find_key(gguf_ctx_, key);
        return id >= 0 ? gguf_get_val_f32(gguf_ctx_, id) : def;
    };

    hparams_.sample_rate = get_u32("mimi.sample_rate", 24000);
    hparams_.frame_rate = get_f32("mimi.frame_rate", 12.5f);
    hparams_.hop_length = get_u32("mimi.hop_length", 1920);
    hparams_.encoder_dim = get_u32("mimi.seanet.dimension", 512);
    hparams_.encoder_layers = get_u32("mimi.transformer.num_layers", 8);
    hparams_.n_codebooks = get_u32("mimi.quantizer.nq", 32);
    hparams_.codebook_size = get_u32("mimi.quantizer.bins", 2048);
    hparams_.codebook_dim = get_u32("mimi.quantizer.dim", 256);
    hparams_.seanet_nfilters = get_u32("mimi.seanet.nfilters", 64);

    // Build tensor map
    std::map<std::string, struct ggml_tensor *> tensors;
    int n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx_, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_, name);
        if (tensor) tensors[name] = tensor;
    }

    auto get_tensor = [&](const std::string & name) -> struct ggml_tensor * {
        // Try with .weight suffix
        auto it = tensors.find(name + ".weight");
        if (it != tensors.end()) return it->second;
        // Try without suffix
        it = tensors.find(name);
        if (it != tensors.end()) return it->second;
        return nullptr;
    };

    auto get_tensor_bias = [&](const std::string & name) -> struct ggml_tensor * {
        auto it = tensors.find(name + ".bias");
        if (it != tensors.end()) return it->second;
        return nullptr;
    };

    // Load SEANet encoder weights
    init_conv_w_ = get_tensor("seanet_enc.init_conv1d.conv.conv");
    init_conv_b_ = get_tensor_bias("seanet_enc.init_conv1d.conv.conv");

    seanet_layers_.resize(4);  // 4 downsampling layers
    for (int i = 0; i < 4; i++) {
        char buf[128];
        auto & L = seanet_layers_[i];

        snprintf(buf, sizeof(buf), "seanet_enc.layers.%d.residuals.0.block.0.conv.conv", i);
        L.res_conv1_w = get_tensor(buf);
        L.res_conv1_b = get_tensor_bias(buf);

        snprintf(buf, sizeof(buf), "seanet_enc.layers.%d.residuals.0.block.1.conv.conv", i);
        L.res_conv2_w = get_tensor(buf);
        L.res_conv2_b = get_tensor_bias(buf);

        snprintf(buf, sizeof(buf), "seanet_enc.layers.%d.downsample.conv.conv", i);
        L.down_conv_w = get_tensor(buf);
        L.down_conv_b = get_tensor_bias(buf);
    }

    final_conv_w_ = get_tensor("seanet_enc.final_conv1d.conv.conv");
    final_conv_b_ = get_tensor_bias("seanet_enc.final_conv1d.conv.conv");

    // Downsample layer
    downsample_conv_w_ = get_tensor("downsample.conv.conv.conv");

    // Load encoder transformer weights
    transformer_layers_.resize(hparams_.encoder_layers);
    for (int i = 0; i < hparams_.encoder_layers; i++) {
        char buf[128];
        auto & L = transformer_layers_[i];

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.norm1", i);
        L.attn_norm_w = get_tensor(buf);
        L.attn_norm_b = get_tensor_bias(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.self_attn.in_proj", i);
        L.attn_in_proj = get_tensor(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.self_attn.out_proj", i);
        L.attn_out_proj = get_tensor(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.layer_scale_1.scale", i);
        L.attn_layer_scale = get_tensor(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.norm2", i);
        L.ffn_norm_w = get_tensor(buf);
        L.ffn_norm_b = get_tensor_bias(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.gating.linear1", i);
        L.ffn_linear1 = get_tensor(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.gating.linear2", i);
        L.ffn_linear2 = get_tensor(buf);

        snprintf(buf, sizeof(buf), "enc_transformer.transformer.layers.%d.layer_scale_2.scale", i);
        L.ffn_layer_scale = get_tensor(buf);
    }

    // Load quantizer weights
    quantizer_input_proj_[0] = get_tensor("quantizer.rvq_first.input_proj");
    quantizer_input_proj_[1] = get_tensor("quantizer.rvq_rest.input_proj");

    // Load codebook EMA statistics (embedding = embedding_sum / cluster_usage)
    embedding_sums_.resize(hparams_.n_codebooks);
    cluster_usages_.resize(hparams_.n_codebooks);
    for (int i = 0; i < hparams_.n_codebooks; i++) {
        char buf_sum[128], buf_usage[128];
        if (i == 0) {
            snprintf(buf_sum, sizeof(buf_sum), "quantizer.rvq_first.vq.layers.0.codebook.embedding_sum");
            snprintf(buf_usage, sizeof(buf_usage), "quantizer.rvq_first.vq.layers.0.codebook.cluster_usage");
        } else {
            snprintf(buf_sum, sizeof(buf_sum), "quantizer.rvq_rest.vq.layers.%d.codebook.embedding_sum", i - 1);
            snprintf(buf_usage, sizeof(buf_usage), "quantizer.rvq_rest.vq.layers.%d.codebook.cluster_usage", i - 1);
        }
        embedding_sums_[i] = get_tensor(buf_sum);
        cluster_usages_[i] = get_tensor(buf_usage);
    }

    loaded_ = true;
    return true;
}

void MimiEncoder::seanet_encode(const float * audio, int n_samples, std::vector<float> & out) {
    // SEANet encoder: series of conv + residual + downsample blocks
    // Initial channels: 1 (mono audio)
    // Layer 0: 1 -> 64, stride 4
    // Layer 1: 64 -> 128, stride 5
    // Layer 2: 128 -> 256, stride 6
    // Layer 3: 256 -> 512, stride 8
    // Final: 512 -> 512 (hidden)
    // Total stride: 4 * 5 * 6 * 8 = 960

    int channels = 1;
    int len = n_samples;

    // Working buffers
    std::vector<float> buf1(n_samples * 1024);  // Max size needed
    std::vector<float> buf2(n_samples * 1024);

    float * cur = buf1.data();
    float * next = buf2.data();

    // Copy input audio
    memcpy(cur, audio, n_samples * sizeof(float));

    // Initial conv: [1, 7, 64] = [in_ch, kernel, out_ch] -> 64 channels
    if (init_conv_w_) {
        // GGUF tensor shape: [in_ch, kernel, out_ch]
        int in_channels = init_conv_w_->ne[0];   // 1
        int kernel_size = init_conv_w_->ne[1];   // 7
        int out_channels = init_conv_w_->ne[2];  // 64

        int out_len;
        causal_conv1d(next, cur, (float *)init_conv_w_->data,
                      init_conv_b_ ? (float *)init_conv_b_->data : nullptr,
                      in_channels, out_channels, kernel_size, 1, len, out_len);

        std::swap(cur, next);
        channels = out_channels;
        len = out_len;
    }

    // 4 downsample blocks
    int strides[4] = {4, 5, 6, 8};
    for (int layer = 0; layer < 4; layer++) {
        auto & L = seanet_layers_[layer];

        // Residual block
        if (L.res_conv1_w && L.res_conv2_w) {
            // ELU activation
            apply_elu(cur, len * channels);

            // First conv (compress: channels -> channels/2)
            // GGUF tensor shape: [in_ch, kernel, out_ch]
            int res1_in_ch = L.res_conv1_w->ne[0];
            int kernel1 = L.res_conv1_w->ne[1];
            int mid_channels = L.res_conv1_w->ne[2];
            int out_len1;
            causal_conv1d(next, cur, (float *)L.res_conv1_w->data,
                         L.res_conv1_b ? (float *)L.res_conv1_b->data : nullptr,
                         res1_in_ch, mid_channels, kernel1, 1, len, out_len1);

            // ELU
            apply_elu(next, out_len1 * mid_channels);

            // Second conv (expand: channels/2 -> channels)
            // GGUF tensor shape: [in_ch, kernel, out_ch]
            int res2_in_ch = L.res_conv2_w->ne[0];
            int kernel2 = L.res_conv2_w->ne[1];
            int out_channels2 = L.res_conv2_w->ne[2];
            std::vector<float> residual(out_len1 * out_channels2);
            int out_len2;
            causal_conv1d(residual.data(), next, (float *)L.res_conv2_w->data,
                         L.res_conv2_b ? (float *)L.res_conv2_b->data : nullptr,
                         res2_in_ch, out_channels2, kernel2, 1, out_len1, out_len2);

            // Add residual (true_skip = identity)
            for (int i = 0; i < out_len2 * out_channels2; i++) {
                cur[i] += residual[i];
            }
            len = out_len2;
        }

        // Downsample conv
        if (L.down_conv_w) {
            apply_elu(cur, len * channels);

            // GGUF tensor shape: [in_ch, kernel, out_ch]
            int down_in_ch = L.down_conv_w->ne[0];
            int kernel_size = L.down_conv_w->ne[1];
            int out_channels = L.down_conv_w->ne[2];
            int stride = strides[layer];
            int out_len;
            causal_conv1d(next, cur, (float *)L.down_conv_w->data,
                         L.down_conv_b ? (float *)L.down_conv_b->data : nullptr,
                         down_in_ch, out_channels, kernel_size, stride, len, out_len);

            std::swap(cur, next);
            channels = out_channels;
            len = out_len;
        }
    }

    // Final conv
    if (final_conv_w_) {
        apply_elu(cur, len * channels);

        // GGUF tensor shape: [in_ch, kernel, out_ch]
        int final_in_ch = final_conv_w_->ne[0];
        int kernel_size = final_conv_w_->ne[1];
        int out_channels = final_conv_w_->ne[2];
        int out_len;
        causal_conv1d(next, cur, (float *)final_conv_w_->data,
                     final_conv_b_ ? (float *)final_conv_b_->data : nullptr,
                     final_in_ch, out_channels, kernel_size, 1, len, out_len);

        std::swap(cur, next);
        channels = out_channels;
        len = out_len;
    }

    // Copy output
    out.resize(len * channels);
    memcpy(out.data(), cur, len * channels * sizeof(float));
}

void MimiEncoder::transformer_encode(std::vector<float> & latents, int n_frames) {
    const int dim = hparams_.encoder_dim;
    const int n_heads = 8;  // From config
    const int head_dim = dim / n_heads;
    const int ffn_dim = 2048;  // From config

    std::vector<float> normed(dim), qkv(3 * dim), attn_out(dim);
    std::vector<float> ffn_hidden(ffn_dim), ffn_out(dim);

    // KV cache for all layers and positions
    std::vector<std::vector<std::vector<float>>> k_cache(hparams_.encoder_layers,
        std::vector<std::vector<float>>(n_frames, std::vector<float>(dim)));
    std::vector<std::vector<std::vector<float>>> v_cache(hparams_.encoder_layers,
        std::vector<std::vector<float>>(n_frames, std::vector<float>(dim)));

    for (int t = 0; t < n_frames; t++) {
        float * x = latents.data() + t * dim;

        for (int layer = 0; layer < hparams_.encoder_layers; layer++) {
            auto & L = transformer_layers_[layer];

            // Self-attention with pre-norm
            if (L.attn_norm_w && L.attn_in_proj && L.attn_out_proj) {
                layer_norm(normed.data(), x, (float *)L.attn_norm_w->data,
                          L.attn_norm_b ? (float *)L.attn_norm_b->data : nullptr, dim);

                matvec(qkv.data(), (float *)L.attn_in_proj->data, normed.data(), 3 * dim, dim);

                float * Q = qkv.data();
                float * K = qkv.data() + dim;
                float * V = qkv.data() + 2 * dim;

                // Cache K and V
                std::copy(K, K + dim, k_cache[layer][t].begin());
                std::copy(V, V + dim, v_cache[layer][t].begin());

                // Causal attention
                std::fill(attn_out.begin(), attn_out.end(), 0.0f);
                float scale = 1.0f / sqrtf((float)head_dim);

                for (int h = 0; h < n_heads; h++) {
                    float * q_h = Q + h * head_dim;
                    std::vector<float> scores(t + 1);

                    for (int s = 0; s <= t; s++) {
                        float * k_s = k_cache[layer][s].data() + h * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            dot += q_h[d] * k_s[d];
                        }
                        scores[s] = dot * scale;
                    }

                    softmax(scores.data(), t + 1);

                    for (int s = 0; s <= t; s++) {
                        float * v_s = v_cache[layer][s].data() + h * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            attn_out[h * head_dim + d] += scores[s] * v_s[d];
                        }
                    }
                }

                // Output projection
                std::vector<float> proj_out(dim);
                matvec(proj_out.data(), (float *)L.attn_out_proj->data, attn_out.data(), dim, dim);

                // Layer scale and residual
                float scale_val = L.attn_layer_scale ? ((float *)L.attn_layer_scale->data)[0] : 1.0f;
                for (int d = 0; d < dim; d++) {
                    x[d] += scale_val * proj_out[d];
                }
            }

            // FFN with pre-norm
            if (L.ffn_norm_w && L.ffn_linear1 && L.ffn_linear2) {
                layer_norm(normed.data(), x, (float *)L.ffn_norm_w->data,
                          L.ffn_norm_b ? (float *)L.ffn_norm_b->data : nullptr, dim);

                matvec(ffn_hidden.data(), (float *)L.ffn_linear1->data, normed.data(), ffn_dim, dim);

                // GELU activation
                for (int i = 0; i < ffn_dim; i++) {
                    float v = ffn_hidden[i];
                    ffn_hidden[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                }

                matvec(ffn_out.data(), (float *)L.ffn_linear2->data, ffn_hidden.data(), dim, ffn_dim);

                // Layer scale and residual
                float scale_val = L.ffn_layer_scale ? ((float *)L.ffn_layer_scale->data)[0] : 1.0f;
                for (int d = 0; d < dim; d++) {
                    x[d] += scale_val * ffn_out[d];
                }
            }
        }
    }
}

void MimiEncoder::quantize(const std::vector<float> & latents, int n_frames,
                           std::vector<std::vector<int32_t>> & tokens) {
    const int dim = hparams_.encoder_dim;
    const int codebook_dim = hparams_.codebook_dim;
    const int codebook_size = hparams_.codebook_size;

    tokens.resize(n_frames);
    for (int t = 0; t < n_frames; t++) {
        tokens[t].resize(hparams_.n_codebooks);
    }

    // Project to quantizer dimension
    std::vector<float> projected(n_frames * codebook_dim);

    // For each frame
    for (int t = 0; t < n_frames; t++) {
        const float * frame = latents.data() + t * dim;
        float * proj = projected.data() + t * codebook_dim;

        // Input projection (512 -> 256)
        if (quantizer_input_proj_[0]) {
            matvec(proj, (float *)quantizer_input_proj_[0]->data, frame, codebook_dim, dim);
        } else {
            // Fallback: just copy first codebook_dim elements
            memcpy(proj, frame, codebook_dim * sizeof(float));
        }
    }

    // Helper to get embedding from EMA statistics: embedding = embedding_sum / cluster_usage
    auto get_embedding = [&](int q, int code, std::vector<float> & out) {
        if (!embedding_sums_[q] || !cluster_usages_[q]) {
            out.assign(codebook_dim, 0.0f);
            return;
        }

        const float * emb_sum = (const float *)embedding_sums_[q]->data;
        const float * usage = (const float *)cluster_usages_[q]->data;

        out.resize(codebook_dim);
        float u = usage[code];
        if (u < 1.0f) u = 1.0f;  // Avoid division by zero

        // embedding_sum shape: [codebook_dim, codebook_size]
        for (int d = 0; d < codebook_dim; d++) {
            out[d] = emb_sum[d * codebook_size + code] / u;
        }
    };

    // Find nearest codebook entry using L2 distance
    auto find_nearest_ema = [&](int q, const float * query) -> int {
        if (!embedding_sums_[q] || !cluster_usages_[q]) return 0;

        const float * emb_sum = (const float *)embedding_sums_[q]->data;
        const float * usage = (const float *)cluster_usages_[q]->data;

        int best_idx = 0;
        float best_dist = 1e30f;

        for (int i = 0; i < codebook_size; i++) {
            float u = usage[i];
            if (u < 1.0f) u = 1.0f;

            float dist = 0.0f;
            for (int d = 0; d < codebook_dim; d++) {
                float emb = emb_sum[d * codebook_size + i] / u;
                float diff = query[d] - emb;
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }
        return best_idx;
    };

    // Residual quantization through all codebooks
    std::vector<float> residual = projected;
    std::vector<float> entry(codebook_dim);

    for (int q = 0; q < hparams_.n_codebooks; q++) {
        if (!embedding_sums_[q] || !cluster_usages_[q]) continue;

        for (int t = 0; t < n_frames; t++) {
            float * res = residual.data() + t * codebook_dim;

            // Find nearest codebook entry
            int idx = find_nearest_ema(q, res);
            tokens[t][q] = idx;

            // Get embedding and subtract from residual
            get_embedding(q, idx, entry);
            for (int d = 0; d < codebook_dim; d++) {
                res[d] -= entry[d];
            }
        }
    }
}

encode_result MimiEncoder::encode(const float * audio, int n_samples, bool verbose) {
    encode_result result;
    result.duration = (float)n_samples / hparams_.sample_rate;

    if (!loaded_ || n_samples == 0) {
        return result;
    }

    if (verbose) {
        fprintf(stderr, "Encoding %.2f seconds of audio (%d samples)...\n", result.duration, n_samples);
    }

    // Step 1: SEANet encode
    std::vector<float> seanet_out;
    seanet_encode(audio, n_samples, seanet_out);

    int n_frames = seanet_out.size() / hparams_.encoder_dim;
    if (verbose) {
        fprintf(stderr, "SEANet output: %d frames x %d channels\n", n_frames, hparams_.encoder_dim);
    }

    // Step 2: Transformer encode
    transformer_encode(seanet_out, n_frames);

    // Step 3: Quantize to discrete tokens
    quantize(seanet_out, n_frames, result.tokens);
    result.n_frames = n_frames;

    if (verbose) {
        fprintf(stderr, "Quantized to %d frames x %d codebooks\n", n_frames, hparams_.n_codebooks);
    }

    return result;
}

// ============================================================================
// MimiDecoder implementation (stub for now)
// ============================================================================

MimiDecoder::MimiDecoder() = default;
MimiDecoder::~MimiDecoder() {
    if (ctx_) ggml_free(ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
}

bool MimiDecoder::load(const std::string & path) {
    // TODO: Implement decoder loading
    fprintf(stderr, "MimiDecoder::load not yet implemented\n");
    return false;
}

decode_result MimiDecoder::decode(const std::vector<std::vector<int32_t>> & tokens, bool verbose) {
    // TODO: Implement decoder
    decode_result result;
    return result;
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<float> load_wav(const std::string & path, int target_sample_rate) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: cannot open WAV file: %s\n", path.c_str());
        return {};
    }

    // Read WAV header
    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "Error: not a valid WAV file\n");
        return {};
    }

    uint32_t file_size;
    file.read((char *)&file_size, 4);

    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: not a valid WAV file\n");
        return {};
    }

    // Find fmt chunk
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;

    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;
        file.read(chunk_id, 4);
        file.read((char *)&chunk_size, 4);

        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read((char *)&audio_format, 2);
            file.read((char *)&num_channels, 2);
            file.read((char *)&sample_rate, 4);
            file.seekg(6, std::ios::cur);  // Skip byte_rate and block_align
            file.read((char *)&bits_per_sample, 2);
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            // Read audio data
            int bytes_per_sample = bits_per_sample / 8;
            int n_samples = chunk_size / (bytes_per_sample * num_channels);

            std::vector<float> audio(n_samples);

            for (int i = 0; i < n_samples; i++) {
                float sample = 0.0f;
                for (int ch = 0; ch < num_channels; ch++) {
                    if (bits_per_sample == 16) {
                        int16_t s;
                        file.read((char *)&s, 2);
                        sample += s / 32768.0f;
                    } else if (bits_per_sample == 32 && audio_format == 3) {
                        // Float32
                        float s;
                        file.read((char *)&s, 4);
                        sample += s;
                    } else if (bits_per_sample == 32) {
                        int32_t s;
                        file.read((char *)&s, 4);
                        sample += s / 2147483648.0f;
                    }
                }
                audio[i] = sample / num_channels;  // Mix to mono
            }

            // TODO: Resample if sample_rate != target_sample_rate
            if (sample_rate != (uint32_t)target_sample_rate) {
                fprintf(stderr, "Warning: sample rate mismatch (%d vs %d), resampling not implemented\n",
                        sample_rate, target_sample_rate);
            }

            return audio;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    return {};
}

bool save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    int n_samples = audio.size();
    int bytes_per_sample = 2;  // 16-bit
    int data_size = n_samples * bytes_per_sample;
    int file_size = 36 + data_size;

    // RIFF header
    file.write("RIFF", 4);
    file.write((char *)&file_size, 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    file.write((char *)&fmt_size, 4);
    uint16_t audio_format = 1;  // PCM
    file.write((char *)&audio_format, 2);
    uint16_t num_channels = 1;
    file.write((char *)&num_channels, 2);
    file.write((char *)&sample_rate, 4);
    uint32_t byte_rate = sample_rate * bytes_per_sample;
    file.write((char *)&byte_rate, 4);
    uint16_t block_align = bytes_per_sample;
    file.write((char *)&block_align, 2);
    uint16_t bits_per_sample = 16;
    file.write((char *)&bits_per_sample, 2);

    // data chunk
    file.write("data", 4);
    file.write((char *)&data_size, 4);

    for (float sample : audio) {
        int16_t s = (int16_t)(sample * 32767.0f);
        file.write((char *)&s, 2);
    }

    return true;
}

} // namespace mimi
