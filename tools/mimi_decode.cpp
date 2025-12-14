// Mimi Neural Audio Codec Decoder
// Standalone tool to decode audio tokens from JSON to WAV
// Uses ggml for tensor operations

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

// Mimi model parameters
struct mimi_hparams {
    int32_t n_codebooks = 8;
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 256;
    int32_t d_model = 512;
    int32_t n_heads = 8;
    int32_t n_layers = 8;
    int32_t dim_feedforward = 2048;
    int32_t sample_rate = 24000;
    float frame_rate = 12.5f;

    // SEANet params
    int32_t seanet_dim = 512;
    int32_t seanet_nfilters = 64;
    std::vector<int32_t> ratios = {8, 6, 5, 4};  // Upsampling ratios
};

// Mimi model weights
struct mimi_model {
    mimi_hparams hparams;

    // Quantizer - first codebook
    struct ggml_tensor * rvq_first_input_proj;   // [codebook_dim, d_model]
    struct ggml_tensor * rvq_first_output_proj;  // [d_model, codebook_dim]
    struct ggml_tensor * rvq_first_embedding_sum;   // [codebook_dim, codebook_size]
    struct ggml_tensor * rvq_first_cluster_usage;   // [codebook_size]

    // Quantizer - rest codebooks (layers 0-6 for codebooks 1-7)
    std::vector<struct ggml_tensor *> rvq_rest_embedding_sum;   // 7 x [codebook_dim, codebook_size]
    std::vector<struct ggml_tensor *> rvq_rest_cluster_usage;   // 7 x [codebook_size]
    struct ggml_tensor * rvq_rest_input_proj;   // [codebook_dim, d_model]
    struct ggml_tensor * rvq_rest_output_proj;  // [d_model, codebook_dim]

    // Decoder transformer (8 layers)
    struct layer {
        struct ggml_tensor * norm1_w;
        struct ggml_tensor * norm1_b;
        struct ggml_tensor * norm2_w;
        struct ggml_tensor * norm2_b;
        struct ggml_tensor * attn_in_proj;    // [3*d_model, d_model]
        struct ggml_tensor * attn_out_proj;   // [d_model, d_model]
        struct ggml_tensor * ffn_linear1;     // [dim_ff, d_model]
        struct ggml_tensor * ffn_linear2;     // [d_model, dim_ff]
        struct ggml_tensor * layer_scale_1;
        struct ggml_tensor * layer_scale_2;
    };
    std::vector<layer> layers;

    // SEANet decoder
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    struct seanet_block {
        struct ggml_tensor * upsample_w;
        struct ggml_tensor * upsample_b;
        struct ggml_tensor * res_conv1_w;
        struct ggml_tensor * res_conv1_b;
        struct ggml_tensor * res_conv2_w;
        struct ggml_tensor * res_conv2_b;
    };
    std::vector<seanet_block> seanet_blocks;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    // GGML context
    struct ggml_context * ctx;
    struct gguf_context * gguf_ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Simple JSON parsing for audio tokens
static std::vector<std::vector<int32_t>> parse_audio_tokens(const std::string & json_path) {
    std::ifstream f(json_path);
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

    std::vector<std::vector<int32_t>> tokens;
    size_t pos = content.find("\"audio_tokens\"");
    if (pos == std::string::npos) return tokens;

    pos = content.find('[', pos);
    if (pos == std::string::npos) return tokens;

    while (true) {
        pos = content.find('[', pos + 1);
        if (pos == std::string::npos) break;

        size_t check = content.find(']', pos);
        if (check != std::string::npos && content.find('[', pos + 1) > check) {
            std::vector<int32_t> frame;
            size_t end = content.find(']', pos);
            std::string frame_str = content.substr(pos + 1, end - pos - 1);

            size_t num_start = 0;
            while (num_start < frame_str.size()) {
                while (num_start < frame_str.size() &&
                       (frame_str[num_start] == ' ' || frame_str[num_start] == ',')) {
                    num_start++;
                }
                if (num_start >= frame_str.size()) break;

                size_t num_end = num_start;
                while (num_end < frame_str.size() &&
                       (frame_str[num_end] >= '0' && frame_str[num_end] <= '9')) {
                    num_end++;
                }
                if (num_end > num_start) {
                    frame.push_back(std::stoi(frame_str.substr(num_start, num_end - num_start)));
                }
                num_start = num_end;
            }

            if (frame.size() == 8) {
                tokens.push_back(frame);
            }
            pos = end;
        }

        size_t next_bracket = content.find('[', pos + 1);
        size_t end_array = content.find(']', pos + 1);
        if (next_bracket == std::string::npos || (end_array < next_bracket)) {
            break;
        }
    }

    return tokens;
}

// Save WAV file
static void save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s for writing\n", path.c_str());
        return;
    }

    std::vector<int16_t> audio_i16(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        float s = audio[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        audio_i16[i] = (int16_t)(s * 32767.0f);
    }

    uint32_t data_size = audio_i16.size() * 2;
    uint32_t file_size = 36 + data_size;

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);

    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t byte_rate = sample_rate * 2;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;

    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(audio_i16.data(), 2, audio_i16.size(), f);

    fclose(f);
}

// Load Mimi model from GGUF
static bool load_model(mimi_model & model, const std::string & path) {
    printf("Loading model from %s...\n", path.c_str());

    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &model.ctx,
    };

    model.gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (!model.gguf_ctx) {
        fprintf(stderr, "Error: failed to load GGUF file\n");
        return false;
    }

    // Read hyperparameters
    int64_t key_id;

    key_id = gguf_find_key(model.gguf_ctx, "mimi.quantizer.nq");
    if (key_id >= 0) model.hparams.n_codebooks = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.quantizer.bins");
    if (key_id >= 0) model.hparams.codebook_size = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.quantizer.dim");
    if (key_id >= 0) model.hparams.codebook_dim = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.embedding_length");
    if (key_id >= 0) model.hparams.d_model = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.block_count");
    if (key_id >= 0) model.hparams.n_layers = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.attention.head_count");
    if (key_id >= 0) model.hparams.n_heads = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.feed_forward_length");
    if (key_id >= 0) model.hparams.dim_feedforward = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.sample_rate");
    if (key_id >= 0) model.hparams.sample_rate = gguf_get_val_u32(model.gguf_ctx, key_id);

    key_id = gguf_find_key(model.gguf_ctx, "mimi.frame_rate");
    if (key_id >= 0) model.hparams.frame_rate = gguf_get_val_f32(model.gguf_ctx, key_id);

    printf("Model params:\n");
    printf("  n_codebooks: %d\n", model.hparams.n_codebooks);
    printf("  codebook_size: %d\n", model.hparams.codebook_size);
    printf("  codebook_dim: %d\n", model.hparams.codebook_dim);
    printf("  d_model: %d\n", model.hparams.d_model);
    printf("  n_layers: %d\n", model.hparams.n_layers);
    printf("  n_heads: %d\n", model.hparams.n_heads);
    printf("  sample_rate: %d\n", model.hparams.sample_rate);
    printf("  frame_rate: %.1f\n", model.hparams.frame_rate);

    // Build tensor map
    int n_tensors = gguf_get_n_tensors(model.gguf_ctx);
    printf("Loading %d tensors...\n", n_tensors);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(model.gguf_ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.ctx, name);
        if (tensor) {
            model.tensors[name] = tensor;
        }
    }

    // Get key tensors
    auto get_tensor = [&](const std::string & name) -> struct ggml_tensor * {
        auto it = model.tensors.find(name);
        if (it == model.tensors.end()) {
            fprintf(stderr, "Warning: tensor not found: %s\n", name.c_str());
            return nullptr;
        }
        return it->second;
    };

    // Quantizer - first codebook
    model.rvq_first_input_proj = get_tensor("quantizer.rvq_first.input_proj.weight");
    model.rvq_first_output_proj = get_tensor("quantizer.rvq_first.output_proj.weight");
    model.rvq_first_embedding_sum = get_tensor("quantizer.rvq_first.vq.layers.0.codebook.embedding_sum.weight");
    model.rvq_first_cluster_usage = get_tensor("quantizer.rvq_first.vq.layers.0.codebook.cluster_usage.weight");

    // Quantizer - rest codebooks (7 codebooks, layers 0-6)
    model.rvq_rest_input_proj = get_tensor("quantizer.rvq_rest.input_proj.weight");
    model.rvq_rest_output_proj = get_tensor("quantizer.rvq_rest.output_proj.weight");
    model.rvq_rest_embedding_sum.resize(7);
    model.rvq_rest_cluster_usage.resize(7);
    for (int i = 0; i < 7; i++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "quantizer.rvq_rest.vq.layers.%d.codebook.embedding_sum.weight", i);
        model.rvq_rest_embedding_sum[i] = get_tensor(buf);
        snprintf(buf, sizeof(buf), "quantizer.rvq_rest.vq.layers.%d.codebook.cluster_usage.weight", i);
        model.rvq_rest_cluster_usage[i] = get_tensor(buf);
    }

    // Transformer layers
    model.layers.resize(model.hparams.n_layers);
    for (int i = 0; i < model.hparams.n_layers; i++) {
        char buf[256];
        auto & layer = model.layers[i];

        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.norm1.weight", i);
        layer.norm1_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.norm1.bias", i);
        layer.norm1_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.norm2.weight", i);
        layer.norm2_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.norm2.bias", i);
        layer.norm2_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.self_attn.in_proj.weight", i);
        layer.attn_in_proj = get_tensor(buf);
        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.self_attn.out_proj.weight", i);
        layer.attn_out_proj = get_tensor(buf);

        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.gating.linear1.weight", i);
        layer.ffn_linear1 = get_tensor(buf);
        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.gating.linear2.weight", i);
        layer.ffn_linear2 = get_tensor(buf);

        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.layer_scale_1.scale.weight", i);
        layer.layer_scale_1 = get_tensor(buf);
        snprintf(buf, sizeof(buf), "dec_transformer.transformer.layers.%d.layer_scale_2.scale.weight", i);
        layer.layer_scale_2 = get_tensor(buf);
    }

    // SEANet decoder
    model.init_conv_w = get_tensor("seanet_dec.init_conv1d.conv.conv.weight");
    model.init_conv_b = get_tensor("seanet_dec.init_conv1d.conv.conv.bias");

    model.seanet_blocks.resize(4);
    for (int i = 0; i < 4; i++) {
        char buf[256];
        auto & block = model.seanet_blocks[i];

        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.upsample.convtr.convtr.weight", i);
        block.upsample_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.upsample.convtr.convtr.bias", i);
        block.upsample_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.residuals.0.block.0.conv.conv.weight", i);
        block.res_conv1_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.residuals.0.block.0.conv.conv.bias", i);
        block.res_conv1_b = get_tensor(buf);

        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.residuals.0.block.1.conv.conv.weight", i);
        block.res_conv2_w = get_tensor(buf);
        snprintf(buf, sizeof(buf), "seanet_dec.layers.%d.residuals.0.block.1.conv.conv.bias", i);
        block.res_conv2_b = get_tensor(buf);
    }

    model.final_conv_w = get_tensor("seanet_dec.final_conv1d.conv.conv.weight");
    model.final_conv_b = get_tensor("seanet_dec.final_conv1d.conv.conv.bias");

    printf("Model loaded successfully\n");
    fflush(stdout);
    return true;
}

// ELU activation (used in SEANet)
static inline float elu(float x) {
    return x >= 0.0f ? x : (expf(x) - 1.0f);
}

// GELU activation (used in transformer)
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// LayerNorm
static void layer_norm(float * out, const float * in, const float * gamma, const float * beta,
                       int n_features, int seq_len, float eps = 1e-5f) {
    for (int t = 0; t < seq_len; t++) {
        // Compute mean and variance
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < n_features; i++) {
            mean += in[t * n_features + i];
        }
        mean /= n_features;

        for (int i = 0; i < n_features; i++) {
            float diff = in[t * n_features + i] - mean;
            var += diff * diff;
        }
        var /= n_features;

        float std_inv = 1.0f / sqrtf(var + eps);

        for (int i = 0; i < n_features; i++) {
            float norm = (in[t * n_features + i] - mean) * std_inv;
            out[t * n_features + i] = gamma[i] * norm + beta[i];
        }
    }
}

// Matrix multiply: C = A @ B^T (A is [M, K], B is [N, K], C is [M, N])
static void matmul(float * C, const float * A, const float * B, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// Softmax over last dimension
static void softmax(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// Transposed 1D convolution (upsampling)
static void conv_transpose1d(
    std::vector<float> & out, const std::vector<float> & in,
    const float * weight, const float * bias,
    int in_ch, int out_ch, int kernel, int stride, int seq_len
) {
    int out_len = (seq_len - 1) * stride + kernel;
    out.assign(out_ch * out_len, 0.0f);

    // Add bias
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int t = 0; t < out_len; t++) {
                out[oc * out_len + t] = bias[oc];
            }
        }
    }

    // Scatter-add
    // Weight shape from GGUF: [in_ch, kernel, out_ch]
    for (int ic = 0; ic < in_ch; ic++) {
        for (int t = 0; t < seq_len; t++) {
            float in_val = in[ic * seq_len + t];
            int t_out = t * stride;

            for (int k = 0; k < kernel; k++) {
                for (int oc = 0; oc < out_ch; oc++) {
                    int w_idx = ic * kernel * out_ch + k * out_ch + oc;
                    out[oc * out_len + t_out + k] += in_val * weight[w_idx];
                }
            }
        }
    }
}

// Regular 1D convolution
static void conv1d(
    std::vector<float> & out, const std::vector<float> & in,
    const float * weight, const float * bias,
    int in_ch, int out_ch, int kernel, int seq_len, int dilation = 1
) {
    int pad = (kernel - 1) * dilation;  // Causal padding
    int out_len = seq_len;  // Same length output with causal padding
    out.assign(out_ch * out_len, 0.0f);

    // Weight shape from GGUF: [in_ch, kernel, out_ch]
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < out_len; t++) {
            float sum = bias ? bias[oc] : 0.0f;

            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int t_in = t - pad + k * dilation;
                    if (t_in >= 0 && t_in < seq_len) {
                        int w_idx = ic * kernel * out_ch + k * out_ch + oc;
                        sum += in[ic * seq_len + t_in] * weight[w_idx];
                    }
                }
            }
            out[oc * out_len + t] = sum;
        }
    }
}

// Apply ELU activation in-place
static void apply_elu(std::vector<float> & x) {
    for (auto & v : x) {
        v = elu(v);
    }
}

// Decode audio tokens to waveform
static std::vector<float> decode(mimi_model & model, const std::vector<std::vector<int32_t>> & tokens) {
    const auto & hp = model.hparams;
    const int T = tokens.size();  // Number of frames

    printf("Decoding %d frames...\n", T);
    fflush(stdout);

    // Step 1: Quantizer decode - lookup embeddings and sum
    // The quantizer uses embedding = embedding_sum / cluster_usage
    // Output: [d_model, T]
    std::vector<float> x(hp.d_model * T, 0.0f);

    if (!model.rvq_first_embedding_sum || !model.rvq_first_cluster_usage) {
        fprintf(stderr, "Error: first quantizer weights not loaded\n");
        return {};
    }

    // Helper to compute embedding from embedding_sum / cluster_usage
    auto get_embedding = [&](const float * emb_sum, const float * usage, int code, std::vector<float> & out) {
        if (code < 0 || code >= hp.codebook_size) code = 0;
        float u = usage[code];
        if (u < 1.0f) u = 1.0f;  // Avoid division by zero

        out.resize(hp.codebook_dim);
        // embedding_sum shape: [codebook_dim, codebook_size]
        for (int d = 0; d < hp.codebook_dim; d++) {
            out[d] = emb_sum[d * hp.codebook_size + code] / u;
        }
    };

    // First codebook
    const float * first_emb_sum = (const float *)model.rvq_first_embedding_sum->data;
    const float * first_usage = (const float *)model.rvq_first_cluster_usage->data;
    const float * first_out_proj = model.rvq_first_output_proj ? (const float *)model.rvq_first_output_proj->data : nullptr;

    std::vector<float> emb(hp.codebook_dim);
    for (int t = 0; t < T; t++) {
        int code = tokens[t][0];
        get_embedding(first_emb_sum, first_usage, code, emb);

        // Project through output_proj: [d_model, codebook_dim] @ [codebook_dim] -> [d_model]
        if (first_out_proj) {
            for (int d = 0; d < hp.d_model; d++) {
                float sum = 0.0f;
                for (int k = 0; k < hp.codebook_dim; k++) {
                    sum += first_out_proj[d * hp.codebook_dim + k] * emb[k];
                }
                x[d * T + t] += sum;
            }
        } else {
            for (int d = 0; d < hp.codebook_dim && d < hp.d_model; d++) {
                x[d * T + t] += emb[d];
            }
        }
    }

    // Rest codebooks (indices 1-7, using rvq_rest.vq.layers.0-6)
    const float * rest_out_proj = model.rvq_rest_output_proj ? (const float *)model.rvq_rest_output_proj->data : nullptr;

    for (int q = 1; q < hp.n_codebooks; q++) {
        int layer_idx = q - 1;  // layers 0-6 for codebooks 1-7
        if (layer_idx >= (int)model.rvq_rest_embedding_sum.size()) {
            fprintf(stderr, "Warning: missing codebook %d\n", q);
            continue;
        }

        if (!model.rvq_rest_embedding_sum[layer_idx] || !model.rvq_rest_cluster_usage[layer_idx]) {
            fprintf(stderr, "Warning: codebook %d weights not loaded\n", q);
            continue;
        }

        const float * rest_emb_sum = (const float *)model.rvq_rest_embedding_sum[layer_idx]->data;
        const float * rest_usage = (const float *)model.rvq_rest_cluster_usage[layer_idx]->data;

        for (int t = 0; t < T; t++) {
            int code = tokens[t][q];
            get_embedding(rest_emb_sum, rest_usage, code, emb);

            if (rest_out_proj) {
                for (int d = 0; d < hp.d_model; d++) {
                    float sum = 0.0f;
                    for (int k = 0; k < hp.codebook_dim; k++) {
                        sum += rest_out_proj[d * hp.codebook_dim + k] * emb[k];
                    }
                    x[d * T + t] += sum;
                }
            } else {
                for (int d = 0; d < hp.codebook_dim && d < hp.d_model; d++) {
                    x[d * T + t] += emb[d];
                }
            }
        }
    }

    printf("  Quantizer decode done, x shape: [%d, %d]\n", hp.d_model, T);
    fflush(stdout);

    // Step 2: Decoder transformer
    // For simplicity, skip transformer for now (it refines the embeddings)
    // In a full implementation, we'd run 8 layers of attention + FFN
    printf("  Skipping transformer (simplified decode)...\n");

    // Step 3: SEANet decoder
    // init_conv: [d_model, T] -> [1024, T]
    std::vector<float> y;

    if (model.init_conv_w) {
        const float * w = (const float *)model.init_conv_w->data;
        const float * b = model.init_conv_b ? (const float *)model.init_conv_b->data : nullptr;

        // Weight shape: [in_ch=512, kernel=7, out_ch=1024]
        conv1d(y, x, w, b, hp.d_model, 1024, 7, T);
        apply_elu(y);
        printf("  init_conv: [%d, %d] -> [1024, %d]\n", hp.d_model, T, T);
        fflush(stdout);
    } else {
        fprintf(stderr, "Warning: init_conv not found, using identity\n");
        y = x;
    }

    int cur_len = T;
    int cur_ch = 1024;

    // 4 upsampling blocks with ratios [8, 6, 5, 4]
    int ratios[] = {8, 6, 5, 4};
    int channels[] = {1024, 512, 256, 128, 64};  // Channels after each block

    for (int i = 0; i < 4; i++) {
        auto & block = model.seanet_blocks[i];
        int ratio = ratios[i];
        int out_ch = channels[i + 1];
        int in_ch = channels[i];

        if (block.upsample_w) {
            const float * w = (const float *)block.upsample_w->data;
            const float * b = block.upsample_b ? (const float *)block.upsample_b->data : nullptr;

            // Kernel size = 2 * ratio
            int kernel = 2 * ratio;
            std::vector<float> upsampled;
            conv_transpose1d(upsampled, y, w, b, in_ch, out_ch, kernel, ratio, cur_len);

            int new_len = (cur_len - 1) * ratio + kernel;
            apply_elu(upsampled);

            // Residual block
            if (block.res_conv1_w && block.res_conv2_w) {
                std::vector<float> res;
                const float * w1 = (const float *)block.res_conv1_w->data;
                const float * b1 = block.res_conv1_b ? (const float *)block.res_conv1_b->data : nullptr;

                // First conv: compress channels
                conv1d(res, upsampled, w1, b1, out_ch, out_ch / 2, 3, new_len);
                apply_elu(res);

                const float * w2 = (const float *)block.res_conv2_w->data;
                const float * b2 = block.res_conv2_b ? (const float *)block.res_conv2_b->data : nullptr;

                // Second conv: expand channels back
                std::vector<float> res2;
                conv1d(res2, res, w2, b2, out_ch / 2, out_ch, 1, new_len);

                // Add residual
                for (size_t j = 0; j < upsampled.size(); j++) {
                    upsampled[j] += res2[j];
                }
            }

            y = std::move(upsampled);
            cur_len = (cur_len - 1) * ratio + kernel;
            cur_ch = out_ch;

            printf("  upsample[%d]: ratio=%d, [%d, %d] -> [%d, %d]\n",
                   i, ratio, in_ch, (cur_len - kernel + ratio - 1) / ratio + 1, out_ch, cur_len);
        }
    }

    // Final conv: [64, L] -> [1, L]
    std::vector<float> audio;
    if (model.final_conv_w) {
        const float * w = (const float *)model.final_conv_w->data;
        const float * b = model.final_conv_b ? (const float *)model.final_conv_b->data : nullptr;

        conv1d(audio, y, w, b, 64, 1, 3, cur_len);
        printf("  final_conv: [64, %d] -> [1, %d]\n", cur_len, cur_len);
    } else {
        // Just take first channel
        audio.resize(cur_len);
        for (int t = 0; t < cur_len; t++) {
            audio[t] = y[t];
        }
    }

    // Apply tanh to clip output
    for (auto & s : audio) {
        s = tanhf(s);
    }

    printf("Decode complete, %zu samples\n", audio.size());
    return audio;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <audio_tokens.json> [-o output.wav] [-m mimi.gguf]\n", argv[0]);
        return 1;
    }

    std::string json_path = argv[1];
    std::string output_path = "";
    std::string model_path = "/tmp/mimi-decoder.gguf";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (output_path.empty()) {
        output_path = json_path;
        size_t pos = output_path.rfind('.');
        if (pos != std::string::npos) {
            output_path = output_path.substr(0, pos);
        }
        output_path += "_native.wav";
    }

    // Parse audio tokens
    printf("Loading audio tokens from %s...\n", json_path.c_str());
    auto tokens = parse_audio_tokens(json_path);
    if (tokens.empty()) {
        fprintf(stderr, "Error: no audio tokens found\n");
        return 1;
    }
    printf("Loaded %zu frames\n", tokens.size());

    // Load model
    mimi_model model;
    if (!load_model(model, model_path)) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }

    // Decode
    auto audio = decode(model, tokens);
    if (audio.empty()) {
        fprintf(stderr, "Error: decoding failed\n");
        return 1;
    }

    // Save output
    save_wav(output_path, audio, model.hparams.sample_rate);
    printf("Saved audio to %s\n", output_path.c_str());
    printf("Duration: %.2f seconds\n", (float)audio.size() / model.hparams.sample_rate);

    // Cleanup
    if (model.ctx) ggml_free(model.ctx);
    if (model.gguf_ctx) gguf_free(model.gguf_ctx);

    return 0;
}
