// Moshi Speech-to-Text Native Implementation
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#include "moshi_stt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>

#include "ggml.h"
#include "gguf.h"

namespace moshi {

// ============================================================================
// Math primitives
// ============================================================================

static void rms_norm(float * out, const float * in, const float * alpha, int dim) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += in[i] * in[i];
    float scale = 1.0f / sqrtf(sum_sq / dim + 1e-8f);
    for (int i = 0; i < dim; i++) out[i] = in[i] * scale * alpha[i];
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

static void apply_rope_qk(float * q, float * k, int head_dim, int pos, float base) {
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(base, (float)(2 * i) / head_dim);
        float theta = pos * freq;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float q0 = q[2*i], q1 = q[2*i + 1];
        q[2*i]     = q0 * cos_t - q1 * sin_t;
        q[2*i + 1] = q0 * sin_t + q1 * cos_t;

        float k0 = k[2*i], k1 = k[2*i + 1];
        k[2*i]     = k0 * cos_t - k1 * sin_t;
        k[2*i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ============================================================================
// STTModel implementation
// ============================================================================

STTModel::STTModel() = default;

STTModel::~STTModel() {
    if (ctx_) ggml_free(ctx_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
}

bool STTModel::load(const std::string & path) {
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

    hparams_.dim = get_u32("moshi-stt.embedding_length", 2048);
    hparams_.n_layers = get_u32("moshi-stt.block_count", 16);
    hparams_.n_heads = get_u32("moshi-stt.attention.head_count", 16);
    hparams_.n_ctx = get_u32("moshi-stt.context_length", 750);
    hparams_.n_vocab_text = get_u32("moshi-stt.vocab_size_text", 8000);
    hparams_.n_vocab_audio = get_u32("moshi-stt.vocab_size_audio", 2048);
    hparams_.n_codebooks = get_u32("moshi-stt.audio_codebooks", 32);
    hparams_.rope_base = get_f32("moshi-stt.rope.freq_base", 100000.0f);

    // Build tensor map
    std::map<std::string, struct ggml_tensor *> tensors;
    int n_tensors = gguf_get_n_tensors(gguf_ctx_);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx_, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_, name);
        if (tensor) tensors[name] = tensor;
    }

    auto get_tensor = [&](const std::string & name) {
        auto it = tensors.find(name);
        return it != tensors.end() ? it->second : nullptr;
    };

    // Load embeddings
    emb_.resize(hparams_.n_codebooks);
    for (int i = 0; i < hparams_.n_codebooks; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "emb.%d.weight", i);
        emb_[i] = get_tensor(buf);
    }
    text_emb_ = get_tensor("text_emb.weight");

    // Load transformer layers
    layers_.resize(hparams_.n_layers);
    for (int i = 0; i < hparams_.n_layers; i++) {
        char buf[128];
        auto & L = layers_[i];

        snprintf(buf, sizeof(buf), "transformer.layers.%d.norm1.alpha", i);
        L.norm1_alpha = get_tensor(buf);
        snprintf(buf, sizeof(buf), "transformer.layers.%d.norm2.alpha", i);
        L.norm2_alpha = get_tensor(buf);
        snprintf(buf, sizeof(buf), "transformer.layers.%d.self_attn.in_proj_weight", i);
        L.attn_in_proj = get_tensor(buf);
        snprintf(buf, sizeof(buf), "transformer.layers.%d.self_attn.out_proj.weight", i);
        L.attn_out_proj = get_tensor(buf);
        snprintf(buf, sizeof(buf), "transformer.layers.%d.gating.linear_in.weight", i);
        L.ffn_linear_in = get_tensor(buf);
        snprintf(buf, sizeof(buf), "transformer.layers.%d.gating.linear_out.weight", i);
        L.ffn_linear_out = get_tensor(buf);
    }

    out_norm_alpha_ = get_tensor("out_norm.alpha");
    text_linear_ = get_tensor("text_linear.weight");

    // Debug: print embedding shapes and values
    if (text_emb_) {
        fprintf(stderr, "text_emb shape: [%d, %d]\n", (int)text_emb_->ne[0], (int)text_emb_->ne[1]);
        const float * d = (const float *)text_emb_->data;
        int dim = (int)text_emb_->ne[0];
        // Token 0 embedding: first `dim` values
        fprintf(stderr, "  Token 0 first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                d[0], d[1], d[2], d[3], d[4]);
        // Token 8000 (BOS) embedding
        fprintf(stderr, "  Token 8000 first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                d[8000*dim], d[8000*dim+1], d[8000*dim+2], d[8000*dim+3], d[8000*dim+4]);
    }
    if (emb_[0]) {
        fprintf(stderr, "audio_emb.0 shape: [%d, %d]\n", (int)emb_[0]->ne[0], (int)emb_[0]->ne[1]);
        const float * d = (const float *)emb_[0]->data;
        int dim = (int)emb_[0]->ne[0];
        // Token 2048 (init) embedding
        fprintf(stderr, "  Audio init (2048) first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                d[2048*dim], d[2048*dim+1], d[2048*dim+2], d[2048*dim+3], d[2048*dim+4]);
        // Token 1174 embedding
        fprintf(stderr, "  Token 1174 first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                d[1174*dim], d[1174*dim+1], d[1174*dim+2], d[1174*dim+3], d[1174*dim+4]);
    }
    if (text_linear_) {
        fprintf(stderr, "text_linear shape: [%d, %d]\n", (int)text_linear_->ne[0], (int)text_linear_->ne[1]);
    }

    loaded_ = true;
    return true;
}

stt_result STTModel::transcribe(const std::vector<std::vector<int32_t>> & frames, bool verbose) {
    stt_result result;
    result.n_frames = frames.size();

    if (!loaded_ || frames.empty()) {
        return result;
    }

    const int T = frames.size();
    const int dim = hparams_.dim;
    const int n_heads = hparams_.n_heads;
    const int head_dim = hparams_.head_dim();

    // Text embedding tokens
    const float * text_emb_w = text_emb_ ? (const float *)text_emb_->data : nullptr;
    const int text_initial_token = hparams_.n_vocab_text;  // 8000 - initial token for frame 0
    const int text_pad_token = 3;  // Padding token for subsequent frames
    const int audio_initial_token = hparams_.n_vocab_audio;  // 2048 - initial audio token

    // KV cache for all layers and all positions
    // We process T positions: [initial, audio[0], ..., audio[T-2]]
    std::vector<std::vector<std::vector<float>>> k_cache(hparams_.n_layers,
        std::vector<std::vector<float>>(T, std::vector<float>(dim)));
    std::vector<std::vector<std::vector<float>>> v_cache(hparams_.n_layers,
        std::vector<std::vector<float>>(T, std::vector<float>(dim)));

    result.all_tokens.resize(T);
    std::vector<float> x(dim), normed(dim), qkv(3 * dim), attn_out(dim);
    std::vector<float> ffn_tmp(2 * 5632), ffn_out_buf(dim);

    const float * out_norm_a = (const float *)out_norm_alpha_->data;
    const float * text_linear_w = (const float *)text_linear_->data;

    // Process T positions: [initial, audio[0], ..., audio[T-2]]
    // Position t uses: initial tokens if t==0, else audio frame t-1
    // Output at position t predicts text for frame t
    // IMPORTANT: Text is AUTOREGRESSIVE - we feed back the predicted text token
    int prev_text_token = text_initial_token;  // Start with initial token

    for (int t = 0; t < T; t++) {
        std::fill(x.begin(), x.end(), 0.0f);

        // Frame 0: text = initial token (8000)
        // Frames 1+: text = PREDICTED token from previous step (autoregressive)
        int text_token = prev_text_token;
        if (text_emb_w) {
            for (int d = 0; d < dim; d++) {
                x[d] += text_emb_w[text_token * dim + d];
            }
        }

        // Audio embeddings
        if (verbose && t <= 1) {
            int audio_tok = (t == 0) ? audio_initial_token : frames[t-1][0];
            fprintf(stderr, "  [debug] Position %d: text=%d, audio=[%d", t, text_token, audio_tok);
            if (t > 0) {
                for (int q = 1; q < std::min(4, hparams_.n_codebooks); q++) {
                    fprintf(stderr, ",%d", frames[t-1][q]);
                }
            }
            fprintf(stderr, "...]\n");
        }

        for (int q = 0; q < hparams_.n_codebooks; q++) {
            if (!emb_[q]) continue;
            int token;
            if (t == 0) {
                token = audio_initial_token;  // 2048 for all codebooks
            } else {
                token = frames[t-1][q];  // Actual audio token from frame t-1
                if (token < 0 || token >= hparams_.n_vocab_audio) {
                    token = audio_initial_token;
                }
            }
            const float * emb_w = (const float *)emb_[q]->data;
            for (int d = 0; d < dim; d++) {
                x[d] += emb_w[token * dim + d];
            }
        }

        // Debug: print x after embedding
        if (verbose && t == 0) {
            float x_norm = 0.0f;
            for (int d = 0; d < dim; d++) x_norm += x[d] * x[d];
            x_norm = sqrtf(x_norm);
            fprintf(stderr, "  [debug] After embed: norm=%.4f, first5=[%.5f,%.5f,%.5f,%.5f,%.5f]\n",
                    x_norm, x[0], x[1], x[2], x[3], x[4]);
        }

        // 2. Run through all transformer layers
        for (int layer = 0; layer < hparams_.n_layers; layer++) {
            auto & L = layers_[layer];

            const float * norm1_a = (const float *)L.norm1_alpha->data;
            const float * norm2_a = (const float *)L.norm2_alpha->data;
            const float * in_proj = (const float *)L.attn_in_proj->data;
            const float * out_proj = (const float *)L.attn_out_proj->data;
            const float * ffn_in = (const float *)L.ffn_linear_in->data;
            const float * ffn_out_w = (const float *)L.ffn_linear_out->data;
            int ffn_dim = L.ffn_linear_out->ne[0];

            // Self-attention
            rms_norm(normed.data(), x.data(), norm1_a, dim);

            // Debug after norm1 for layer 0, frame 0
            if (verbose && t == 0 && layer == 0) {
                float n_norm = 0.0f;
                for (int d = 0; d < dim; d++) n_norm += normed[d] * normed[d];
                n_norm = sqrtf(n_norm);
                fprintf(stderr, "  [debug] After norm1: norm=%.4f, first5=[%.5f,%.5f,%.5f,%.5f,%.5f]\n",
                        n_norm, normed[0], normed[1], normed[2], normed[3], normed[4]);
            }

            matvec(qkv.data(), in_proj, normed.data(), 3 * dim, dim);

            // Debug QKV for layer 0, frame 0
            if (verbose && t == 0 && layer == 0) {
                fprintf(stderr, "  [debug] Q first5=[%.5f,%.5f,%.5f,%.5f,%.5f]\n",
                        qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]);
                fprintf(stderr, "  [debug] K first5=[%.5f,%.5f,%.5f,%.5f,%.5f]\n",
                        qkv[dim], qkv[dim+1], qkv[dim+2], qkv[dim+3], qkv[dim+4]);
            }

            float * Q = qkv.data();
            float * K = qkv.data() + dim;
            float * V = qkv.data() + 2 * dim;

            // Apply RoPE to Q and K
            for (int h = 0; h < n_heads; h++) {
                apply_rope_qk(Q + h * head_dim, K + h * head_dim, head_dim, t, hparams_.rope_base);
            }

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

            // Output projection and residual
            std::vector<float> proj_out(dim);
            matvec(proj_out.data(), out_proj, attn_out.data(), dim, dim);
            for (int d = 0; d < dim; d++) {
                x[d] += proj_out[d];
            }

            // FFN
            rms_norm(normed.data(), x.data(), norm2_a, dim);
            matvec(ffn_tmp.data(), ffn_in, normed.data(), 2 * ffn_dim, dim);

            for (int i = 0; i < ffn_dim; i++) {
                ffn_tmp[i] = silu(ffn_tmp[i]) * ffn_tmp[ffn_dim + i];
            }

            matvec(ffn_out_buf.data(), ffn_out_w, ffn_tmp.data(), dim, ffn_dim);
            for (int d = 0; d < dim; d++) {
                x[d] += ffn_out_buf[d];
            }
        }

        // 3. Final norm and output projection
        rms_norm(normed.data(), x.data(), out_norm_a, dim);

        std::vector<float> logits(hparams_.n_vocab_text);
        matvec(logits.data(), text_linear_w, normed.data(), hparams_.n_vocab_text, dim);

        // Argmax
        int best = 0;
        for (int i = 1; i < hparams_.n_vocab_text; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        result.all_tokens[t] = best;

        // Feed back predicted token for next step (autoregressive)
        prev_text_token = best;

        if (verbose && (t < 3 || best >= 4 || t == T - 1)) {
            fprintf(stderr, "  Frame %d: token %d (logit=%.4f)", t, best, logits[best]);
            // Show top 5 logits for debugging
            std::vector<std::pair<float, int>> sorted_logits;
            for (int i = 0; i < std::min(hparams_.n_vocab_text, 100); i++) {
                sorted_logits.push_back({logits[i], i});
            }
            std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<>());
            fprintf(stderr, " top5: ");
            for (int i = 0; i < std::min(5, (int)sorted_logits.size()); i++) {
                fprintf(stderr, "%d=%.2f ", sorted_logits[i].second, sorted_logits[i].first);
            }
            fprintf(stderr, "\n");
        }
    }

    // Collect valid text tokens (>=4, excluding padding tokens 0-3)
    for (int token : result.all_tokens) {
        if (token >= 4) {
            result.text_tokens.push_back(token);
        }
    }

    return result;
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<std::vector<int32_t>> parse_audio_tokens_json(const std::string & content, int n_codebooks) {
    std::vector<std::vector<int32_t>> codebook_tokens;
    size_t pos = content.find("\"audio_tokens\"");
    if (pos == std::string::npos) return {};

    pos = content.find('[', pos);
    if (pos == std::string::npos) return {};

    int depth = 0;
    std::vector<int32_t> current;
    std::string num_str;

    for (size_t i = pos; i < content.size(); i++) {
        char c = content[i];
        if (c == '[') {
            depth++;
            if (depth == 2) current.clear();
        } else if (c == ']') {
            if (!num_str.empty()) {
                current.push_back(std::stoi(num_str));
                num_str.clear();
            }
            if (depth == 2 && !current.empty()) {
                codebook_tokens.push_back(current);
            }
            depth--;
            if (depth == 0) break;
        } else if (c == ',') {
            if (!num_str.empty()) {
                current.push_back(std::stoi(num_str));
                num_str.clear();
            }
        } else if (c >= '0' && c <= '9') {
            num_str += c;
        }
    }

    // Transpose [n_q][T] -> [T][n_q]
    if (!codebook_tokens.empty() && codebook_tokens.size() == (size_t)n_codebooks) {
        int T = codebook_tokens[0].size();
        std::vector<std::vector<int32_t>> frames(T);
        for (int t = 0; t < T; t++) {
            frames[t].resize(n_codebooks);
            for (int q = 0; q < n_codebooks; q++) {
                frames[t][q] = codebook_tokens[q][t];
            }
        }
        return frames;
    }
    return codebook_tokens;
}

std::vector<std::vector<int32_t>> load_audio_tokens(const std::string & path, int n_codebooks) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    return parse_audio_tokens_json(content, n_codebooks);
}

} // namespace moshi
