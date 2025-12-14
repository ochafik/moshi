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
#include "ggml-cpu.h"
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

    // Top-level upsample (2x before transformer)
    struct ggml_tensor * upsample_w;  // [512, 4, 1] - transposed conv for 2x upsample

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
    fprintf(stderr, "  DEBUG: opening file...\n");
    std::ifstream f(json_path);
    if (!f.good()) {
        fprintf(stderr, "  ERROR: cannot open file\n");
        return {};
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    fprintf(stderr, "  DEBUG: file read, %zu bytes\n", content.size());

    std::vector<std::vector<int32_t>> tokens;
    size_t pos = content.find("\"audio_tokens\"");
    if (pos == std::string::npos) {
        fprintf(stderr, "  ERROR: audio_tokens not found\n");
        return tokens;
    }
    fprintf(stderr, "  DEBUG: found audio_tokens at pos %zu\n", pos);

    pos = content.find('[', pos);
    if (pos == std::string::npos) return tokens;

    int frame_count = 0;
    int loop_count = 0;
    while (true) {
        loop_count++;
        if (loop_count > 1000) {
            fprintf(stderr, "  ERROR: infinite loop detected at pos %zu\n", pos);
            break;
        }

        pos = content.find('[', pos + 1);
        if (pos == std::string::npos) break;

        size_t check = content.find(']', pos);
        size_t next_open = content.find('[', pos + 1);
        fprintf(stderr, "    loop %d: pos=%zu, check=%zu, next_open=%zu\n", loop_count, pos, check, next_open);
        if (check != std::string::npos && next_open > check) {
            std::vector<int32_t> frame;
            size_t end = content.find(']', pos);
            std::string frame_str = content.substr(pos + 1, end - pos - 1);
            fprintf(stderr, "      parsing frame: '%s'\n", frame_str.c_str());

            size_t num_start = 0;
            while (num_start < frame_str.size()) {
                while (num_start < frame_str.size() &&
                       (frame_str[num_start] == ' ' || frame_str[num_start] == ',' || frame_str[num_start] == '\n')) {
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

            fprintf(stderr, "      frame has %zu values\n", frame.size());
            if (frame.size() == 8) {
                tokens.push_back(frame);
                frame_count++;
            }
            pos = end;
        }

        size_t next_bracket = content.find('[', pos + 1);
        size_t end_array = content.find(']', pos + 1);
        fprintf(stderr, "    after frame: next_bracket=%zu, end_array=%zu\n", next_bracket, end_array);
        if (next_bracket == std::string::npos || (end_array != std::string::npos && end_array < next_bracket)) {
            fprintf(stderr, "    breaking loop\n");
            break;
        }
    }

    fprintf(stderr, "  DEBUG: parsed %d frames\n", frame_count);
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
    printf("  dim_feedforward: %d\n", model.hparams.dim_feedforward);
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

    // Top-level upsample (2x before transformer)
    model.upsample_w = get_tensor("upsample.convtr.convtr.convtr.weight");

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
// trim_end: number of samples to remove from the end (like unpad1d(y, (0, trim_end)))
static void conv_transpose1d(
    std::vector<float> & out, const std::vector<float> & in,
    const float * weight, const float * bias,
    int in_ch, int out_ch, int kernel, int stride, int seq_len, int trim_end = 0
) {
    // Full output length before trimming
    int full_len = (seq_len - 1) * stride + kernel;
    // Final output length after trimming from the end only
    int out_len = full_len - trim_end;

    // Work in full buffer first, then trim
    std::vector<float> full_out(out_ch * full_len, 0.0f);

    // Add bias to full buffer
    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int t = 0; t < full_len; t++) {
                full_out[oc * full_len + t] = bias[oc];
            }
        }
    }

    // Scatter-add
    // GGUF weight shape: [K, OC, IC] in GGML ne[] order (matches ggml_conv_transpose_1d)
    // PyTorch ConvTranspose1d weight was [IC, OC, K], stored directly without transpose
    // To access w[k][oc][ic]: index = k + oc * kernel + ic * kernel * out_ch
    for (int ic = 0; ic < in_ch; ic++) {
        for (int t = 0; t < seq_len; t++) {
            float in_val = in[ic * seq_len + t];
            int t_out = t * stride;

            for (int k = 0; k < kernel; k++) {
                for (int oc = 0; oc < out_ch; oc++) {
                    int w_idx = k + oc * kernel + ic * kernel * out_ch;
                    full_out[oc * full_len + t_out + k] += in_val * weight[w_idx];
                }
            }
        }
    }

    // Extract from start, trimming from the end only
    out.resize(out_ch * out_len);
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < out_len; t++) {
            out[oc * out_len + t] = full_out[oc * full_len + t];
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

    // GGUF weight shape: [K, IC, OC] in GGML ne[] order (matches ggml_conv_1d)
    // PyTorch weight was [OC, IC, K], stored directly without transpose
    // To access w[k][ic][oc]: index = k + ic * kernel + oc * kernel * in_ch
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < out_len; t++) {
            float sum = bias ? bias[oc] : 0.0f;

            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int t_in = t - pad + k * dilation;
                    if (t_in >= 0 && t_in < seq_len) {
                        int w_idx = k + ic * kernel + oc * kernel * in_ch;
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
    // IMPORTANT: The output_proj is applied ONCE after summing embeddings, not per-codebook
    // Output: [d_model, T]
    std::vector<float> x(hp.d_model * T, 0.0f);

    if (!model.rvq_first_embedding_sum || !model.rvq_first_cluster_usage) {
        fprintf(stderr, "Error: first quantizer weights not loaded\n");
        return {};
    }

    // Helper to compute embedding from embedding_sum / cluster_usage
    // Note: epsilon=1e-5 matches Python's EuclideanCodebook
    auto get_embedding = [&](const float * emb_sum, const float * usage, int code, std::vector<float> & out) {
        if (code < 0 || code >= hp.codebook_size) code = 0;
        float u = usage[code];
        if (u < 1e-5f) u = 1e-5f;  // Match Python epsilon

        out.resize(hp.codebook_dim);
        // Data layout: [codebook_size, codebook_dim] = [2048, 256]
        for (int d = 0; d < hp.codebook_dim; d++) {
            out[d] = emb_sum[code * hp.codebook_dim + d] / u;
        }
    };

    // Helper to apply output projection: Conv1d(256, 512, kernel=1)
    // PyTorch weight: [out, in, kernel] = [512, 256, 1], index: w[d * 256 + k]
    auto apply_out_proj = [&](const float * proj, const std::vector<float> & in, std::vector<float> & out) {
        out.resize(hp.d_model);
        for (int d = 0; d < hp.d_model; d++) {
            float sum = 0.0f;
            for (int k = 0; k < hp.codebook_dim; k++) {
                sum += proj[d * hp.codebook_dim + k] * in[k];
            }
            out[d] = sum;
        }
    };

    const float * first_emb_sum = (const float *)model.rvq_first_embedding_sum->data;
    const float * first_usage = (const float *)model.rvq_first_cluster_usage->data;
    const float * first_out_proj = model.rvq_first_output_proj ? (const float *)model.rvq_first_output_proj->data : nullptr;
    const float * rest_out_proj = model.rvq_rest_output_proj ? (const float *)model.rvq_rest_output_proj->data : nullptr;

    std::vector<float> emb(hp.codebook_dim);
    std::vector<float> emb_sum_first(hp.codebook_dim);
    std::vector<float> emb_sum_rest(hp.codebook_dim);
    std::vector<float> proj_first(hp.d_model);
    std::vector<float> proj_rest(hp.d_model);

    for (int t = 0; t < T; t++) {
        // First codebook (rvq_first): 1 codebook, apply output_proj once
        std::fill(emb_sum_first.begin(), emb_sum_first.end(), 0.0f);
        get_embedding(first_emb_sum, first_usage, tokens[t][0], emb);
        for (int d = 0; d < hp.codebook_dim; d++) {
            emb_sum_first[d] += emb[d];
        }

        // Apply output projection for rvq_first
        if (first_out_proj) {
            apply_out_proj(first_out_proj, emb_sum_first, proj_first);
        } else {
            proj_first = emb_sum_first;
            proj_first.resize(hp.d_model, 0.0f);
        }

        // Rest codebooks (rvq_rest): 7 codebooks, sum embeddings first, then apply output_proj once
        std::fill(emb_sum_rest.begin(), emb_sum_rest.end(), 0.0f);
        for (int q = 1; q < hp.n_codebooks; q++) {
            int layer_idx = q - 1;
            if (layer_idx >= (int)model.rvq_rest_embedding_sum.size()) continue;
            if (!model.rvq_rest_embedding_sum[layer_idx]) continue;

            const float * rest_emb_sum = (const float *)model.rvq_rest_embedding_sum[layer_idx]->data;
            const float * rest_usage = (const float *)model.rvq_rest_cluster_usage[layer_idx]->data;

            get_embedding(rest_emb_sum, rest_usage, tokens[t][q], emb);
            for (int d = 0; d < hp.codebook_dim; d++) {
                emb_sum_rest[d] += emb[d];
            }
        }

        // Apply output projection for rvq_rest
        if (rest_out_proj) {
            apply_out_proj(rest_out_proj, emb_sum_rest, proj_rest);
        } else {
            proj_rest = emb_sum_rest;
            proj_rest.resize(hp.d_model, 0.0f);
        }

        // Sum both contributions
        for (int d = 0; d < hp.d_model; d++) {
            x[d * T + t] = proj_first[d] + proj_rest[d];
        }

    }

    printf("  Quantizer decode done, x shape: [%d, %d]\n", hp.d_model, T);
    printf("  Quantizer output range: [");
    float qmin = x[0], qmax = x[0];
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] < qmin) qmin = x[i];
        if (x[i] > qmax) qmax = x[i];
    }
    printf("%.6f, %.6f]\n", qmin, qmax);
    printf("  First 8 values at t=0: ");
    for (int d = 0; d < 8 && d < hp.d_model; d++) {
        printf("%.6f ", x[d * T + 0]);
    }
    printf("\n");
    fflush(stdout);

    // Step 2: Apply 2x upsample before transformer
    // This is a depthwise transposed conv with kernel=4, stride=2
    // Input: [d_model, T], Output: [d_model, T*2]
    int T_up = T * 2;  // Output length after 2x upsample
    if (model.upsample_w) {
        printf("  Applying 2x upsample: [%d, %d] -> [%d, %d]\n", hp.d_model, T, hp.d_model, T_up);
        fflush(stdout);

        // upsample_w GGUF shape: [K=4, OC=1, IC=512] in GGML ne[] order
        // PyTorch weight was [IC=512, OC=1, K=4] for depthwise groups
        // Access: w[k + c * kernel] for channel c, kernel position k
        const float * w = (const float *)model.upsample_w->data;
        const int kernel = 4;
        const int stride = 2;

        std::vector<float> x_up(hp.d_model * T_up, 0.0f);

        // Depthwise transposed conv: each channel processed independently
        // out[c, t_out] = sum_k(w[c, k] * in[c, t_in]) where t_in = (t_out - k) / stride
        for (int c = 0; c < hp.d_model; c++) {
            for (int t_in = 0; t_in < T; t_in++) {
                float val = x[c * T + t_in];
                for (int k = 0; k < kernel; k++) {
                    int t_out = t_in * stride + k;
                    if (t_out >= 0 && t_out < T_up) {
                        // GGML ne[] = [K, OC=1, IC], access w[k][0][c] = k + c * kernel
                        x_up[c * T_up + t_out] += val * w[k + c * kernel];
                    }
                }
            }
        }
        x = std::move(x_up);

        printf("  Upsample output range: [");
        float umin = x[0], umax = x[0];
        for (size_t i = 0; i < x.size(); i++) {
            if (x[i] < umin) umin = x[i];
            if (x[i] > umax) umax = x[i];
        }
        printf("%.6f, %.6f]\n", umin, umax);
        printf("  First 8 upsample values at t=0: ");
        for (int d = 0; d < 8 && d < hp.d_model; d++) {
            printf("%.6f ", x[d * T_up + 0]);
        }
        printf("\n");
        fflush(stdout);
    } else {
        printf("  Warning: upsample weights not found, skipping upsample\n");
        T_up = T;
    }

    // Step 3: Decoder transformer (8 layers) using ggml computation graph
    // Use ggml_backend for proper multi-context handling
    bool skip_transformer = getenv("SKIP_TRANSFORMER") != nullptr;
    int n_tokens_final = T_up;  // Will be used for SEANet

    if (skip_transformer) {
        printf("  SKIPPING decoder transformer (SKIP_TRANSFORMER set)\n");
        fflush(stdout);
    } else {
    printf("  Running decoder transformer (%d layers) with ggml backend...\n", hp.n_layers);
    fflush(stdout);

    const int64_t n_embd = hp.d_model;
    const int64_t n_head = hp.n_heads;
    const int64_t n_tokens = T_up;  // Use upsampled token count
    const int64_t n_embd_head = n_embd / n_head;

    printf("  DEBUG: n_embd=%lld, n_head=%lld, n_tokens=%lld, n_embd_head=%lld\n",
           (long long)n_embd, (long long)n_head, (long long)n_tokens, (long long)n_embd_head);
    fflush(stdout);

    // Transpose input: x is [d_model, T_up], we work with [n_embd, n_tokens]
    std::vector<float> x_transposed(n_tokens * n_embd);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int d = 0; d < hp.d_model; d++) {
            x_transposed[t * n_embd + d] = x[d * n_tokens + t];
        }
    }

    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Error: failed to create CPU backend\n");
        return {};
    }
    printf("  CPU backend initialized\n");
    fflush(stdout);

    for (int il = 0; il < hp.n_layers; il++) {
        auto & L = model.layers[il];

        if (!L.norm1_w || !L.attn_in_proj || !L.attn_out_proj ||
            !L.norm2_w || !L.ffn_linear1 || !L.ffn_linear2) {
            printf("    Layer %d: missing weights, skipping\n", il);
            continue;
        }


        // Create context for graph structure only (no tensor data allocation)
        size_t ctx_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
        struct ggml_init_params ctx_params = {
            .mem_size   = ctx_size,
            .mem_buffer = nullptr,
            .no_alloc   = true,  // Don't allocate tensor data, just structure
        };
        struct ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) {
            fprintf(stderr, "Error: failed to create context for layer %d\n", il);
            ggml_backend_free(backend);
            return {};
        }

        // Create input tensor (will be allocated by backend)
        struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inp, "input");
        ggml_set_input(inp);

        // Create position tensor for RoPE
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(inp_pos, "pos");
        ggml_set_input(inp_pos);

        struct ggml_tensor * cur = inp;  // cur will be reassigned through graph
        struct ggml_tensor * inpSA = cur;

        // === Self-attention ===
        // Create all weight tensors in compute context
        struct ggml_tensor * w_norm1 = ggml_dup_tensor(ctx, L.norm1_w);
        struct ggml_tensor * w_in_proj = ggml_dup_tensor(ctx, L.attn_in_proj);
        struct ggml_tensor * w_out_proj = ggml_dup_tensor(ctx, L.attn_out_proj);
        struct ggml_tensor * w_norm2 = ggml_dup_tensor(ctx, L.norm2_w);
        struct ggml_tensor * w_ffn1 = ggml_dup_tensor(ctx, L.ffn_linear1);
        struct ggml_tensor * w_ffn2 = ggml_dup_tensor(ctx, L.ffn_linear2);

        // Optional tensors (biases, layer scales)
        struct ggml_tensor * b_norm1 = L.norm1_b ? ggml_dup_tensor(ctx, L.norm1_b) : nullptr;
        struct ggml_tensor * b_norm2 = L.norm2_b ? ggml_dup_tensor(ctx, L.norm2_b) : nullptr;
        struct ggml_tensor * w_scale1 = L.layer_scale_1 ? ggml_dup_tensor(ctx, L.layer_scale_1) : nullptr;
        struct ggml_tensor * w_scale2 = L.layer_scale_2 ? ggml_dup_tensor(ctx, L.layer_scale_2) : nullptr;

        // Mark as inputs
        ggml_set_input(w_norm1);
        ggml_set_input(w_in_proj);
        ggml_set_input(w_out_proj);
        ggml_set_input(w_norm2);
        ggml_set_input(w_ffn1);
        ggml_set_input(w_ffn2);
        if (b_norm1) ggml_set_input(b_norm1);
        if (b_norm2) ggml_set_input(b_norm2);
        if (w_scale1) ggml_set_input(w_scale1);
        if (w_scale2) ggml_set_input(w_scale2);

        // LayerNorm1
        struct ggml_tensor * norm1 = ggml_norm(ctx, cur, 1e-5f);
        norm1 = ggml_mul(ctx, norm1, w_norm1);
        if (b_norm1) {
            norm1 = ggml_add(ctx, norm1, b_norm1);
        }

        // QKV projection
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, w_in_proj, norm1);

        // Split Q, K, V - use cont to make them contiguous
        struct ggml_tensor * Qcur = ggml_cont(ctx, ggml_view_2d(ctx, qkv, n_embd, n_tokens, qkv->nb[1], 0));
        struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, qkv, n_embd, n_tokens, qkv->nb[1], n_embd * sizeof(float)));
        struct ggml_tensor * Vcur = ggml_cont(ctx, ggml_view_2d(ctx, qkv, n_embd, n_tokens, qkv->nb[1], 2 * n_embd * sizeof(float)));

        // Reshape for multi-head
        Qcur = ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, n_embd_head, n_head, n_tokens);

        // Apply RoPE
        Qcur = ggml_rope_ext(ctx, Qcur, inp_pos, nullptr,
                n_embd_head, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                n_embd_head, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for attention: [head_dim, n_head, n_tokens] -> [head_dim, n_tokens, n_head]
        struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * v = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        // Compute attention: K @ Q^T
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        // Scale and softmax
        float kq_scale = 1.0f / sqrtf((float)n_embd_head);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, kq_scale, 0.0f);

        // Attention output: softmax @ V (transpose V for correct dimensions)
        struct ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
        struct ggml_tensor * attn = ggml_mul_mat(ctx, v_t, kq);

        // Permute back and reshape
        attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx, attn, n_embd, n_tokens);

        // Output projection
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, w_out_proj, attn);

        // Layer scale
        if (w_scale1) {
            attn_out = ggml_mul(ctx, attn_out, w_scale1);
        }

        // Residual
        cur = ggml_add(ctx, attn_out, inpSA);

        // === FFN ===
        struct ggml_tensor * ffn_inp = cur;

        // LayerNorm2
        struct ggml_tensor * norm2 = ggml_norm(ctx, cur, 1e-5f);
        norm2 = ggml_mul(ctx, norm2, w_norm2);
        if (b_norm2) {
            norm2 = ggml_add(ctx, norm2, b_norm2);
        }

        // FFN: up -> GELU -> down
        struct ggml_tensor * ffn_hidden = ggml_mul_mat(ctx, w_ffn1, norm2);
        ffn_hidden = ggml_gelu_erf(ctx, ffn_hidden);  // Use erf GELU to match PyTorch default
        struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, w_ffn2, ffn_hidden);

        // Layer scale
        if (w_scale2) {
            ffn_out = ggml_mul(ctx, ffn_out, w_scale2);
        }

        // Residual
        cur = ggml_add(ctx, ffn_out, ffn_inp);
        ggml_set_name(cur, "output");
        ggml_set_output(cur);

        // Build graph
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, cur);

        // Allocate backend buffer for compute tensors
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "Error: failed to allocate backend buffer for layer %d\n", il);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return {};
        }

        // Copy input data to the INPUT tensor (not cur, which has been reassigned)
        ggml_backend_tensor_set(inp, x_transposed.data(), 0, n_tokens * n_embd * sizeof(float));

        // Set position indices
        std::vector<int32_t> positions(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            positions[t] = t;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, n_tokens * sizeof(int32_t));

        // Copy model weights to compute tensors
        ggml_backend_tensor_set(w_norm1, L.norm1_w->data, 0, ggml_nbytes(L.norm1_w));
        ggml_backend_tensor_set(w_in_proj, L.attn_in_proj->data, 0, ggml_nbytes(L.attn_in_proj));
        ggml_backend_tensor_set(w_out_proj, L.attn_out_proj->data, 0, ggml_nbytes(L.attn_out_proj));
        ggml_backend_tensor_set(w_norm2, L.norm2_w->data, 0, ggml_nbytes(L.norm2_w));
        ggml_backend_tensor_set(w_ffn1, L.ffn_linear1->data, 0, ggml_nbytes(L.ffn_linear1));
        ggml_backend_tensor_set(w_ffn2, L.ffn_linear2->data, 0, ggml_nbytes(L.ffn_linear2));

        // Copy optional weights
        if (b_norm1) ggml_backend_tensor_set(b_norm1, L.norm1_b->data, 0, ggml_nbytes(L.norm1_b));
        if (b_norm2) ggml_backend_tensor_set(b_norm2, L.norm2_b->data, 0, ggml_nbytes(L.norm2_b));
        if (w_scale1) ggml_backend_tensor_set(w_scale1, L.layer_scale_1->data, 0, ggml_nbytes(L.layer_scale_1));
        if (w_scale2) ggml_backend_tensor_set(w_scale2, L.layer_scale_2->data, 0, ggml_nbytes(L.layer_scale_2));

        // Compute
        enum ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "Error: backend compute failed for layer %d, status=%d\n", il, (int)status);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return {};
        }

        // Copy result back
        ggml_backend_tensor_get(cur, x_transposed.data(), 0, n_tokens * n_embd * sizeof(float));

        // Debug: print layer output
        float lmin = x_transposed[0], lmax = x_transposed[0];
        for (size_t i = 0; i < x_transposed.size(); i++) {
            if (x_transposed[i] < lmin) lmin = x_transposed[i];
            if (x_transposed[i] > lmax) lmax = x_transposed[i];
        }
        printf("  Layer %d: range [%.4f, %.4f], first 4 at t=0: %.4f %.4f %.4f %.4f\n",
               il, lmin, lmax, x_transposed[0], x_transposed[1], x_transposed[2], x_transposed[3]);
        fflush(stdout);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);

    // Transpose back to [d_model, n_tokens]
    x.resize(hp.d_model * n_tokens);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int d = 0; d < hp.d_model; d++) {
            x[d * n_tokens + t] = x_transposed[t * n_embd + d];
        }
    }

    printf("  Decoder transformer done\n");
    printf("  Transformer output range: [");
    float tmin = x[0], tmax = x[0];
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] < tmin) tmin = x[i];
        if (x[i] > tmax) tmax = x[i];
    }
    printf("%.6f, %.6f]\n", tmin, tmax);
    printf("  First 8 transformer values at t=0: ");
    for (int d = 0; d < 8 && d < hp.d_model; d++) {
        printf("%.6f ", x[d * n_tokens + 0]);
    }
    printf("\n");
    fflush(stdout);

    n_tokens_final = n_tokens;  // Update for SEANet
    } // End else block for transformer

    // Step 4: SEANet decoder
    // init_conv: [d_model, n_tokens] -> [1024, n_tokens]
    std::vector<float> y;
    int T_final = n_tokens_final;  // Token count after transformer (T_up)

    if (model.init_conv_w) {
        const float * w = (const float *)model.init_conv_w->data;
        const float * b = model.init_conv_b ? (const float *)model.init_conv_b->data : nullptr;

        // Weight shape: [in_ch=512, kernel=7, out_ch=1024]
        conv1d(y, x, w, b, hp.d_model, 1024, 7, T_final);
        float ic_min = y[0], ic_max = y[0];
        for (size_t i = 0; i < y.size(); i++) {
            if (y[i] < ic_min) ic_min = y[i];
            if (y[i] > ic_max) ic_max = y[i];
        }
        printf("  init_conv: [%d, %d] -> [1024, %d], range [%.4f, %.4f]\n",
               hp.d_model, T_final, T_final, ic_min, ic_max);
        printf("    First 4 at ch=0, t=0: %.4f %.4f %.4f %.4f\n",
               y[0], y[1 * T_final], y[2 * T_final], y[3 * T_final]);
        apply_elu(y);
        fflush(stdout);
    } else {
        fprintf(stderr, "Warning: init_conv not found, using identity\n");
        y = x;
    }

    int cur_len = T_final;
    int cur_ch = 1024;

    // 4 upsampling blocks with ratios [8, 6, 5, 4]
    int ratios[] = {8, 6, 5, 4};
    int channels[] = {1024, 512, 256, 128, 64};  // Channels after each block

    for (int i = 0; i < 4; i++) {
        auto & block = model.seanet_blocks[i];
        int ratio = ratios[i];
        int out_ch = channels[i + 1];
        int in_ch = channels[i];

        if (i == 0) {
            // Debug: print input to first upsample (after init_conv + ELU)
            float y_min = y[0], y_max = y[0];
            for (size_t j = 0; j < y.size(); j++) {
                if (y[j] < y_min) y_min = y[j];
                if (y[j] > y_max) y_max = y[j];
            }
            printf("  Before upsample[0]: range [%.4f, %.4f]\n", y_min, y_max);
            printf("    First 4 at ch=0, t=0: %.4f %.4f %.4f %.4f\n",
                   y[0], y[1 * cur_len], y[2 * cur_len], y[3 * cur_len]);
            printf("    First 4 at ch=0, t=1: %.4f %.4f %.4f %.4f\n",
                   y[1], y[1 * cur_len + 1], y[2 * cur_len + 1], y[3 * cur_len + 1]);
        }

        if (block.upsample_w) {
            const float * w = (const float *)block.upsample_w->data;
            const float * b = block.upsample_b ? (const float *)block.upsample_b->data : nullptr;

            // Kernel size = 2 * ratio
            int kernel = 2 * ratio;
            std::vector<float> upsampled;
            // Trim = kernel - stride (like PyTorch's unpad1d(y, (0, K-S)))
            int trim_end = kernel - ratio;
            conv_transpose1d(upsampled, y, w, b, in_ch, out_ch, kernel, ratio, cur_len, trim_end);

            // Output length: (in-1)*stride + kernel - trim_end = (in-1)*stride + stride = in*stride
            int new_len = cur_len * ratio;

            // Debug: print range and first values after upsample
            if (i == 0) {
                float up_min = upsampled[0], up_max = upsampled[0];
                for (size_t j = 0; j < upsampled.size(); j++) {
                    if (upsampled[j] < up_min) up_min = upsampled[j];
                    if (upsampled[j] > up_max) up_max = upsampled[j];
                }
                printf("    After upsample[0]: range [%.4f, %.4f]\n", up_min, up_max);
                printf("      First 4 at ch=0, t=0: %.4f %.4f %.4f %.4f\n",
                       upsampled[0], upsampled[1 * new_len], upsampled[2 * new_len], upsampled[3 * new_len]);
            }

            // NOTE: No ELU here! ELU is inside the ResBlock

            // Residual block: shortcut(x) + block(ELU→Conv3→ELU→Conv1)
            if (block.res_conv1_w && block.res_conv2_w) {
                // Block path: ELU -> Conv3 -> ELU -> Conv1
                std::vector<float> block_out = upsampled;  // Copy for block path
                apply_elu(block_out);  // ELU before first conv

                const float * w1 = (const float *)block.res_conv1_w->data;
                const float * b1 = block.res_conv1_b ? (const float *)block.res_conv1_b->data : nullptr;

                // First conv: compress channels (out_ch -> out_ch/2)
                std::vector<float> res;
                conv1d(res, block_out, w1, b1, out_ch, out_ch / 2, 3, new_len);

                if (i == 0) {
                    float r_min = res[0], r_max = res[0];
                    for (size_t j = 0; j < res.size(); j++) {
                        if (res[j] < r_min) r_min = res[j];
                        if (res[j] > r_max) r_max = res[j];
                    }
                    printf("      After Conv3: range [%.4f, %.4f]\n", r_min, r_max);
                    printf("        First 4: %.4f %.4f %.4f %.4f\n",
                           res[0], res[1 * new_len], res[2 * new_len], res[3 * new_len]);
                }

                apply_elu(res);  // ELU after first conv

                const float * w2 = (const float *)block.res_conv2_w->data;
                const float * b2 = block.res_conv2_b ? (const float *)block.res_conv2_b->data : nullptr;

                // Second conv: expand channels back (out_ch/2 -> out_ch)
                std::vector<float> res2;
                conv1d(res2, res, w2, b2, out_ch / 2, out_ch, 1, new_len);

                if (i == 0) {
                    float r_min = res2[0], r_max = res2[0];
                    for (size_t j = 0; j < res2.size(); j++) {
                        if (res2[j] < r_min) r_min = res2[j];
                        if (res2[j] > r_max) r_max = res2[j];
                    }
                    printf("      After Conv1: range [%.4f, %.4f]\n", r_min, r_max);
                    printf("        First 4: %.4f %.4f %.4f %.4f\n",
                           res2[0], res2[1 * new_len], res2[2 * new_len], res2[3 * new_len]);
                }

                // Add residual: shortcut (upsampled) + block (res2)
                for (size_t j = 0; j < upsampled.size(); j++) {
                    upsampled[j] += res2[j];
                }

                if (i == 0) {
                    float r_min = upsampled[0], r_max = upsampled[0];
                    for (size_t j = 0; j < upsampled.size(); j++) {
                        if (upsampled[j] < r_min) r_min = upsampled[j];
                        if (upsampled[j] > r_max) r_max = upsampled[j];
                    }
                    printf("      After residual add: range [%.4f, %.4f]\n", r_min, r_max);
                    printf("        First 4: %.4f %.4f %.4f %.4f\n",
                           upsampled[0], upsampled[1 * new_len], upsampled[2 * new_len], upsampled[3 * new_len]);
                }
            }

            // ELU after the entire resblock
            apply_elu(upsampled);

            y = std::move(upsampled);
            cur_len = new_len;
            cur_ch = out_ch;

            printf("  upsample[%d]: ratio=%d, [%d, %d] -> [%d, %d]\n",
                   i, ratio, in_ch, cur_len / ratio, out_ch, cur_len);
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

    // Trim to expected length: T * (sample_rate / frame_rate)
    int expected_samples = (int)(T * hp.sample_rate / hp.frame_rate);
    if ((int)audio.size() > expected_samples) {
        printf("  Trimming from %zu to %d samples\n", audio.size(), expected_samples);
        audio.resize(expected_samples);
    }

    printf("Decode complete, %zu samples\n", audio.size());
    return audio;
}

int main(int argc, char ** argv) {
    // Disable buffering for immediate output
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    fprintf(stderr, "DEBUG: mimi_decode starting...\n");

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
    printf("Loading model from %s...\n", model_path.c_str());
    mimi_model model;
    if (!load_model(model, model_path)) {
        fprintf(stderr, "Error: failed to load model\n");
        return 1;
    }
    printf("Model loaded successfully\n");

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
