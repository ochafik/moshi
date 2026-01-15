// Mimi Neural Audio Codec Decoder
// Standalone tool to decode audio tokens from JSON to WAV
// Uses ggml for tensor operations (including full SEANet with conv1d, conv_transpose1d, elu)

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

std::vector<std::vector<int32_t>> parse_audio_tokens(const std::string & json_path);
void save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate);

// Note: F32->F16 conversion is done using ggml_cast() in the graph

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

// ============================================================================
// GGML-based SEANet decoder
// ============================================================================

// Helper: Causal pad input on the left (for causal convolution)
// Input: [seq_len, channels, batch, 1], Output: [seq_len + pad, channels, batch, 1]
// Uses ggml_pad_ext which pads with zeros: lp0=left pad dim0, rp0=right pad dim0
static struct ggml_tensor * causal_pad_left(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    int pad
) {
    if (pad <= 0) return x;

    // ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    // For 4D tensor [seq_len, channels, batch, 1], pad dimension 0 on the left
    return ggml_pad_ext(ctx, x, pad, 0, 0, 0, 0, 0, 0, 0);
}

// Helper: Conv1d with causal padding using GGML
// Input: [seq_len, in_ch, N, 1], Weight: F16 [K, in_ch, out_ch], Bias: F32 [out_ch]
// Output: [out_seq, out_ch, N, 1] (same length due to causal padding)
static struct ggml_tensor * ggml_conv1d_causal(
    struct ggml_context * ctx,
    struct ggml_tensor * x,       // F32 [seq_len, in_ch, N, 1]
    struct ggml_tensor * w_f16,   // F16 [K, in_ch, out_ch]
    struct ggml_tensor * bias,    // F32 [out_ch] or nullptr
    int kernel,
    int dilation
) {
    int64_t seq_len = x->ne[0];
    int64_t out_ch = w_f16->ne[2];
    int64_t batch = x->ne[2];

    // Causal padding: all padding on the left
    int pad = (kernel - 1) * dilation;
    struct ggml_tensor * x_padded = causal_pad_left(ctx, x, pad);

    // Conv1d with no padding (padding already applied manually)
    // Output length: seq_len + pad - dilation*(kernel-1) - 1 + 1 = seq_len
    // Output shape: [out_seq, out_ch, N, 1]
    struct ggml_tensor * y = ggml_conv_1d(ctx, w_f16, x_padded, 1, 0, dilation);

    // Add bias: y is [out_seq, out_ch, N, 1], bias is [out_ch]
    if (bias) {
        // For 4D output, need to broadcast bias across seq_len, batch dims
        // ggml_add broadcasts along dims where bias has size 1
        struct ggml_tensor * bias_4d = ggml_reshape_4d(ctx, bias, 1, out_ch, 1, 1);
        y = ggml_add(ctx, y, bias_4d);
    }

    return y;
}

// Helper: ConvTranspose1d with output trimming using GGML
// Input: [seq_len, in_ch], Weight: F32 [K, out_ch, in_ch]
// Output: [seq_len * stride, out_ch] (after trimming kernel-stride samples from end)
static struct ggml_tensor * ggml_conv_transpose1d_trimmed(
    struct ggml_context * ctx,
    struct ggml_tensor * x,       // F32 [seq_len, in_ch]
    struct ggml_tensor * weight,  // F32 [K, out_ch, in_ch]
    struct ggml_tensor * bias,    // F32 [out_ch] or nullptr
    int kernel,
    int stride
) {
    int64_t seq_len = x->ne[0];
    int64_t out_ch = weight->ne[1];

    // ConvTranspose1d: output_len = (seq_len - 1) * stride + kernel
    // ggml_conv_transpose_1d requires p0=0
    struct ggml_tensor * y = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);

    // Full output length
    int64_t full_len = (seq_len - 1) * stride + kernel;

    // Trim from end: remove (kernel - stride) samples
    int trim = kernel - stride;
    int64_t out_len = full_len - trim;  // = seq_len * stride

    // Use view to trim (only take first out_len samples)
    y = ggml_view_2d(ctx, y, out_len, out_ch, y->nb[1], 0);
    y = ggml_cont(ctx, y);

    // Add bias
    if (bias) {
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
        y = ggml_add(ctx, y, bias);
        y = ggml_cont(ctx, ggml_transpose(ctx, y));
    }

    return y;
}

// Run full SEANet decoder using GGML
// Input: x [d_model, seq_len] layout in memory (will be transposed internally)
// Output: audio [1, samples]
static std::vector<float> seanet_decode_ggml(
    mimi_model & model,
    const std::vector<float> & x_in,  // [d_model, seq_len] row-major
    int seq_len
) {
    const auto & hp = model.hparams;
    int d_model = hp.d_model;

    printf("  Running SEANet decoder with GGML...\n");
    fflush(stdout);

    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Error: failed to create CPU backend for SEANet\n");
        return {};
    }

    // Calculate total output samples after all upsampling: seq_len * 8 * 6 * 5 * 4 = seq_len * 960
    int final_len = seq_len;
    for (int r : {8, 6, 5, 4}) final_len *= r;

    // Create context for graph
    // Need enough space for many tensors (convs, activations, residuals)
    size_t ctx_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create GGML context for SEANet\n");
        ggml_backend_free(backend);
        return {};
    }

    // Input tensor: [seq_len, d_model, 1, 1]
    // GGML conv_1d expects [IL, IC, batch, 1] - 4D with ne[3]=1
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, seq_len, d_model, 1, 1);
    ggml_set_name(inp, "seanet_input");
    ggml_set_input(inp);
    ggml_set_output(inp);  // Also set as output to verify data

    // Weight tensors (F32, will be cast to F16 in graph for conv1d)
    // init_conv: [K=7, IC=512, OC=1024]
    int init_k = 7, init_ic = d_model, init_oc = 1024;
    struct ggml_tensor * init_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, init_k, init_ic, init_oc);
    struct ggml_tensor * init_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, init_oc);
    ggml_set_input(init_w);
    ggml_set_input(init_b);

    // Cast init_conv weight to F16 for ggml_conv_1d (required by im2col)
    struct ggml_tensor * init_w_f16 = ggml_cast(ctx, init_w, GGML_TYPE_F16);
    ggml_set_name(init_w_f16, "debug_init_w_f16");
    ggml_set_output(init_w_f16);  // Add as output so we can verify the cast

    // Current tensor - start with input
    struct ggml_tensor * cur = inp;

    // init_conv: Conv1d with causal padding (all on left)
    int causal_pad = init_k - 1;  // kernel=7 -> pad=6

    // Apply causal padding manually (pad left), then conv with no padding
    struct ggml_tensor * cur_padded = causal_pad_left(ctx, cur, causal_pad);
    struct ggml_tensor * conv_out = ggml_conv_1d(ctx, init_w_f16, cur_padded, 1, 0, 1);

    // Add bias
    struct ggml_tensor * bias_4d = ggml_reshape_4d(ctx, init_b, 1, init_oc, 1, 1);
    struct ggml_tensor * with_bias = ggml_add(ctx, conv_out, bias_4d);

    // Apply ELU
    cur = ggml_elu(ctx, with_bias);

    // SEANet blocks: 4 upsampling blocks with ratios [8, 6, 5, 4]
    int ratios[] = {8, 6, 5, 4};
    int channels[] = {1024, 512, 256, 128, 64};

    // Pre-declare weight tensors for all blocks (F32, cast to F16 in graph)
    struct {
        struct ggml_tensor * up_w;       // ConvTranspose weight (F32)
        struct ggml_tensor * up_b;       // ConvTranspose bias
        struct ggml_tensor * res1_w;     // Residual conv1 weight (F32)
        struct ggml_tensor * res1_b;     // Residual conv1 bias
        struct ggml_tensor * res2_w;     // Residual conv2 weight (F32)
        struct ggml_tensor * res2_b;     // Residual conv2 bias
        // Cast versions for conv1d
        struct ggml_tensor * res1_w_f16;
        struct ggml_tensor * res2_w_f16;
    } block_weights[4];

    for (int i = 0; i < 4; i++) {
        int ratio = ratios[i];
        int in_ch = channels[i];
        int out_ch = channels[i + 1];
        int up_kernel = 2 * ratio;

        // ConvTranspose weight: [K, out_ch, in_ch] - stays F32
        block_weights[i].up_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, up_kernel, out_ch, in_ch);
        block_weights[i].up_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);
        ggml_set_input(block_weights[i].up_w);
        ggml_set_input(block_weights[i].up_b);

        // Residual conv1: kernel=3, out_ch -> out_ch/2 (F32 input, cast to F16)
        block_weights[i].res1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, out_ch, out_ch / 2);
        block_weights[i].res1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch / 2);
        ggml_set_input(block_weights[i].res1_w);
        ggml_set_input(block_weights[i].res1_b);
        block_weights[i].res1_w_f16 = ggml_cast(ctx, block_weights[i].res1_w, GGML_TYPE_F16);

        // Residual conv2: kernel=1, out_ch/2 -> out_ch (F32 input, cast to F16)
        block_weights[i].res2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, out_ch / 2, out_ch);
        block_weights[i].res2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);
        ggml_set_input(block_weights[i].res2_w);
        ggml_set_input(block_weights[i].res2_b);
        block_weights[i].res2_w_f16 = ggml_cast(ctx, block_weights[i].res2_w, GGML_TYPE_F16);
    }

    // Build graph for all 4 blocks
    for (int i = 0; i < 4; i++) {
        int ratio = ratios[i];
        int out_ch = channels[i + 1];
        int up_kernel = 2 * ratio;

        // Upsample with ConvTranspose1d
        cur = ggml_conv_transpose1d_trimmed(ctx, cur, block_weights[i].up_w, block_weights[i].up_b, up_kernel, ratio);

        // Save for residual connection
        struct ggml_tensor * shortcut = cur;

        // Residual block: ELU -> Conv3 -> ELU -> Conv1
        struct ggml_tensor * blk = ggml_elu(ctx, cur);
        blk = ggml_conv1d_causal(ctx, blk, block_weights[i].res1_w_f16, block_weights[i].res1_b, 3, 1);
        blk = ggml_elu(ctx, blk);
        blk = ggml_conv1d_causal(ctx, blk, block_weights[i].res2_w_f16, block_weights[i].res2_b, 1, 1);

        // Residual add
        cur = ggml_add(ctx, shortcut, blk);

        // ELU after residual
        cur = ggml_elu(ctx, cur);
    }

    // final_conv: [64, L] -> [1, L] with kernel=3
    struct ggml_tensor * final_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 64, 1);
    struct ggml_tensor * final_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(final_w);
    ggml_set_input(final_b);
    struct ggml_tensor * final_w_f16 = ggml_cast(ctx, final_w, GGML_TYPE_F16);

    cur = ggml_conv1d_causal(ctx, cur, final_w_f16, final_b, 3, 1);

    // tanh activation
    cur = ggml_tanh(ctx, cur);

    ggml_set_name(cur, "seanet_output");
    ggml_set_output(cur);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    printf("  DEBUG: graph has %d nodes\n", ggml_graph_n_nodes(gf));

    // Allocate backend buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "Error: failed to allocate backend buffer for SEANet\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return {};
    }

    // Set input data: convert from [d_model, seq_len] row-major to GGML [seq_len, d_model, 1, 1] layout
    // GGML 4D tensor has ne[0]=seq_len, ne[1]=d_model, ne[2]=1, ne[3]=1
    // Memory layout: data[t + d * seq_len] = value at time t, dimension d
    // x_in is in [d_model, seq_len] row-major: x_in[d * seq_len + t] = channel d, time t
    size_t inp_nbytes = ggml_nbytes(inp);
    size_t expected_bytes = seq_len * d_model * sizeof(float);
    printf("  DEBUG: inp tensor nbytes=%zu, expected=%zu, nelements=%lld\n",
           inp_nbytes, expected_bytes, ggml_nelements(inp));

    std::vector<float> inp_transposed(seq_len * d_model);
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_model; d++) {
            // GGML layout: index = t + d * seq_len
            inp_transposed[t + d * seq_len] = x_in[d * seq_len + t];
        }
    }
    ggml_backend_tensor_set(inp, inp_transposed.data(), 0, inp_transposed.size() * sizeof(float));

    // Verify the data was set by reading it back
    std::vector<float> inp_verify(seq_len * d_model);
    ggml_backend_tensor_get(inp, inp_verify.data(), 0, inp_verify.size() * sizeof(float));
    float v_min = inp_verify[0], v_max = inp_verify[0];
    for (auto v : inp_verify) { if (v < v_min) v_min = v; if (v > v_max) v_max = v; }
    printf("  DEBUG: inp data after set: range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
           v_min, v_max, inp_verify[0], inp_verify[seq_len], inp_verify[2*seq_len], inp_verify[3*seq_len]);

    // Debug: verify input was set
    float inp_min = inp_transposed[0], inp_max = inp_transposed[0];
    for (auto v : inp_transposed) { if (v < inp_min) inp_min = v; if (v > inp_max) inp_max = v; }
    printf("  DEBUG SEANet input: range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
           inp_min, inp_max, inp_transposed[0], inp_transposed[seq_len], inp_transposed[2*seq_len], inp_transposed[3*seq_len]);

    // Set init_conv weights (F32, graph will cast to F16)
    if (model.init_conv_w) {
        ggml_backend_tensor_set(init_w, model.init_conv_w->data, 0, ggml_nbytes(init_w));
        // Debug: verify init_conv weights
        const float * w_ptr = (const float *)model.init_conv_w->data;
        size_t w_size = ggml_nelements(model.init_conv_w);
        float w_min = w_ptr[0], w_max = w_ptr[0];
        for (size_t i = 0; i < w_size; i++) { if (w_ptr[i] < w_min) w_min = w_ptr[i]; if (w_ptr[i] > w_max) w_max = w_ptr[i]; }
        printf("  DEBUG init_conv_w (model): ne=[%lld, %lld, %lld], range=[%.4f, %.4f]\n",
               model.init_conv_w->ne[0], model.init_conv_w->ne[1], model.init_conv_w->ne[2], w_min, w_max);
        printf("  DEBUG init_conv_w first 4: %.6f %.6f %.6f %.6f\n", w_ptr[0], w_ptr[1], w_ptr[2], w_ptr[3]);

        // Read back the F32 tensor to verify it was set
        std::vector<float> w_readback(init_k * init_ic * init_oc);
        ggml_backend_tensor_get(init_w, w_readback.data(), 0, w_readback.size() * sizeof(float));
        printf("  DEBUG init_w (readback): first 4: %.6f %.6f %.6f %.6f\n",
               w_readback[0], w_readback[1], w_readback[2], w_readback[3]);
    }
    if (model.init_conv_b) {
        ggml_backend_tensor_set(init_b, model.init_conv_b->data, 0, ggml_nbytes(model.init_conv_b));
    }

    // Set block weights (F32, graph will cast conv weights to F16)
    for (int i = 0; i < 4; i++) {
        auto & block = model.seanet_blocks[i];

        // ConvTranspose weights (F32)
        if (block.upsample_w) {
            ggml_backend_tensor_set(block_weights[i].up_w, block.upsample_w->data, 0, ggml_nbytes(block.upsample_w));
        }
        if (block.upsample_b) {
            ggml_backend_tensor_set(block_weights[i].up_b, block.upsample_b->data, 0, ggml_nbytes(block.upsample_b));
        }

        // Residual conv1 weights (F32)
        if (block.res_conv1_w) {
            ggml_backend_tensor_set(block_weights[i].res1_w, block.res_conv1_w->data, 0, ggml_nbytes(block.res_conv1_w));
        }
        if (block.res_conv1_b) {
            ggml_backend_tensor_set(block_weights[i].res1_b, block.res_conv1_b->data, 0, ggml_nbytes(block.res_conv1_b));
        }

        // Residual conv2 weights (F32)
        if (block.res_conv2_w) {
            ggml_backend_tensor_set(block_weights[i].res2_w, block.res_conv2_w->data, 0, ggml_nbytes(block.res_conv2_w));
        }
        if (block.res_conv2_b) {
            ggml_backend_tensor_set(block_weights[i].res2_b, block.res_conv2_b->data, 0, ggml_nbytes(block.res_conv2_b));
        }
    }

    // Set final_conv weights (F32, graph will cast to F16)
    if (model.final_conv_w) {
        ggml_backend_tensor_set(final_w, model.final_conv_w->data, 0, ggml_nbytes(model.final_conv_w));
    }
    if (model.final_conv_b) {
        ggml_backend_tensor_set(final_b, model.final_conv_b->data, 0, ggml_nbytes(model.final_conv_b));
    }

    // Compute
    printf("  Computing SEANet graph...\n");
    fflush(stdout);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Error: SEANet compute failed, status=%d\n", (int)status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return {};
    }

    // Debug: check raw conv output (before bias)
    struct ggml_tensor * debug_conv_raw = ggml_get_tensor(ctx, "debug_conv_out_raw");
    if (debug_conv_raw) {
        std::vector<float> conv_raw_data(ggml_nelements(debug_conv_raw));
        ggml_backend_tensor_get(debug_conv_raw, conv_raw_data.data(), 0, ggml_nbytes(debug_conv_raw));
        float cmin = conv_raw_data[0], cmax = conv_raw_data[0];
        for (auto v : conv_raw_data) { if (v < cmin) cmin = v; if (v > cmax) cmax = v; }
        printf("  DEBUG conv_out_raw (before bias): range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
               cmin, cmax, conv_raw_data[0], conv_raw_data[seq_len], conv_raw_data[2*seq_len], conv_raw_data[3*seq_len]);
    }

    // Debug: check bias tensor
    struct ggml_tensor * debug_bias = ggml_get_tensor(ctx, "debug_bias_4d");
    if (debug_bias) {
        std::vector<float> bias_data(ggml_nelements(debug_bias));
        ggml_backend_tensor_get(debug_bias, bias_data.data(), 0, ggml_nbytes(debug_bias));
        float bmin = bias_data[0], bmax = bias_data[0];
        for (auto v : bias_data) { if (v < bmin) bmin = v; if (v > bmax) bmax = v; }
        printf("  DEBUG bias_4d: range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
               bmin, bmax, bias_data[0], bias_data[1], bias_data[2], bias_data[3]);
    }

    // Debug: check with_bias tensor
    struct ggml_tensor * debug_with_bias = ggml_get_tensor(ctx, "debug_with_bias");
    if (debug_with_bias) {
        std::vector<float> wb_data(ggml_nelements(debug_with_bias));
        ggml_backend_tensor_get(debug_with_bias, wb_data.data(), 0, ggml_nbytes(debug_with_bias));
        float wbmin = wb_data[0], wbmax = wb_data[0];
        for (auto v : wb_data) { if (v < wbmin) wbmin = v; if (v > wbmax) wbmax = v; }
        printf("  DEBUG with_bias: range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
               wbmin, wbmax, wb_data[0], wb_data[seq_len], wb_data[2*seq_len], wb_data[3*seq_len]);
    }

    // Debug: check init_w_f16 cast output
    struct ggml_tensor * debug_w_f16 = ggml_get_tensor(ctx, "debug_init_w_f16");
    if (debug_w_f16) {
        size_t w_bytes = ggml_nbytes(debug_w_f16);
        std::vector<uint16_t> w_f16_data(ggml_nelements(debug_w_f16));
        ggml_backend_tensor_get(debug_w_f16, w_f16_data.data(), 0, w_bytes);
        // Convert first few F16 values to F32 for printing
        float w0 = ggml_fp16_to_fp32(w_f16_data[0]);
        float w1 = ggml_fp16_to_fp32(w_f16_data[1]);
        float w2 = ggml_fp16_to_fp32(w_f16_data[2]);
        float w3 = ggml_fp16_to_fp32(w_f16_data[3]);
        printf("  DEBUG init_w_f16 (after cast): first 4 values = %.6f %.6f %.6f %.6f\n",
               w0, w1, w2, w3);
    }

    // Debug: check if input tensor was preserved after computation
    struct ggml_tensor * debug_inp = ggml_get_tensor(ctx, "seanet_input");
    if (debug_inp) {
        std::vector<float> inp_after(seq_len * d_model);
        ggml_backend_tensor_get(debug_inp, inp_after.data(), 0, inp_after.size() * sizeof(float));
        float inp_min_after = inp_after[0], inp_max_after = inp_after[0];
        for (auto v : inp_after) { if (v < inp_min_after) inp_min_after = v; if (v > inp_max_after) inp_max_after = v; }
        printf("  DEBUG inp (after compute): range=[%.4f, %.4f], first 4: %.4f %.4f %.4f %.4f\n",
               inp_min_after, inp_max_after, inp_after[0], inp_after[seq_len], inp_after[2*seq_len], inp_after[3*seq_len]);
    }

    // Debug: print final output
    struct ggml_tensor * debug_output = ggml_get_tensor(ctx, "seanet_output");
    if (debug_output) {
        int64_t n0 = debug_output->ne[0];  // samples
        int64_t n1 = debug_output->ne[1];  // channels (should be 1)
        std::vector<float> out_data(n0 * n1);
        ggml_backend_tensor_get(debug_output, out_data.data(), 0, out_data.size() * sizeof(float));
        float omin = out_data[0], omax = out_data[0];
        for (auto v : out_data) { if (v < omin) omin = v; if (v > omax) omax = v; }
        printf("  DEBUG seanet_output: shape=[%lld, %lld], range=[%.4f, %.4f]\n", n0, n1, omin, omax);
        printf("    first 10 samples: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               out_data[0], out_data[1], out_data[2], out_data[3], out_data[4],
               out_data[5], out_data[6], out_data[7], out_data[8], out_data[9]);
    }

    // Get output: cur is [final_len, 1]
    int64_t out_len = cur->ne[0];
    std::vector<float> audio(out_len);
    ggml_backend_tensor_get(cur, audio.data(), 0, out_len * sizeof(float));

    printf("  SEANet done, output %lld samples\n", (long long)out_len);

    // Debug: print range
    float amin = audio[0], amax = audio[0];
    for (auto v : audio) {
        if (v < amin) amin = v;
        if (v > amax) amax = v;
    }
    printf("  Audio range: [%.4f, %.4f]\n", amin, amax);

    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return audio;
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

        // upsample_w GGUF shape: ne[] = [IC=512, K=4, OC=1]
        // PyTorch weight was [IC=512, OC=1, K=4] for depthwise groups
        // GGML stores in column-major order, so to access channel c, kernel k:
        // w[c + k * n_channels]
        const float * w = (const float *)model.upsample_w->data;
        const int kernel = 4;
        const int stride = 2;
        const int n_channels = hp.d_model;

        std::vector<float> x_up(hp.d_model * T_up, 0.0f);

        // Depthwise transposed conv: each channel processed independently
        // out[c, t_out] = sum_k(w[c, k] * in[c, t_in]) where t_in = (t_out - k) / stride
        for (int c = 0; c < hp.d_model; c++) {
            for (int t_in = 0; t_in < T; t_in++) {
                float val = x[c * T + t_in];
                for (int k = 0; k < kernel; k++) {
                    int t_out = t_in * stride + k;
                    if (t_out >= 0 && t_out < T_up) {
                        // GGML ne[] = [IC, K, OC], access w[c][k][0] = c + k * n_channels
                        x_up[c * T_up + t_out] += val * w[c + k * n_channels];
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

    // Debug: print transformer input values (compare with Python)
    printf("  Transformer input @ t=0 (first 8): ");
    for (int i = 0; i < 8 && i < n_embd; i++) {
        printf("%.6f ", x_transposed[i]);
    }
    printf("\n");
    float inp_min = x_transposed[0], inp_max = x_transposed[0];
    for (size_t i = 0; i < x_transposed.size(); i++) {
        if (x_transposed[i] < inp_min) inp_min = x_transposed[i];
        if (x_transposed[i] > inp_max) inp_max = x_transposed[i];
    }
    printf("  Transformer input range: [%.4f, %.4f]\n", inp_min, inp_max);
    fflush(stdout);

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
        ggml_set_name(norm1, "debug_norm1");

        // QKV projection
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, w_in_proj, norm1);
        ggml_set_name(qkv, "debug_qkv");

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
        ggml_set_name(Qcur, "debug_q_rope");
        Kcur = ggml_rope_ext(ctx, Kcur, inp_pos, nullptr,
                n_embd_head, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);
        ggml_set_name(Kcur, "debug_k_rope");

        // Permute for attention: [head_dim, n_head, n_tokens] -> [head_dim, n_tokens, n_head]
        struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        struct ggml_tensor * v = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        // Compute attention: K @ Q^T
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        // Create causal mask - KQ additive mask for causal attention
        // ggml_soft_max_ext expects the mask to be added to logits before softmax
        // For causal attention, future positions should be -inf
        struct ggml_tensor * KQ_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
        ggml_set_name(KQ_mask, "KQ_mask");
        ggml_set_input(KQ_mask);

        // Scale and softmax with causal mask
        float kq_scale = 1.0f / sqrtf((float)n_embd_head);
        kq = ggml_soft_max_ext(ctx, kq, KQ_mask, kq_scale, 0.0f);

        // Attention output: softmax @ V (transpose V for correct dimensions)
        struct ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
        struct ggml_tensor * attn = ggml_mul_mat(ctx, v_t, kq);

        // Permute back and reshape
        attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
        struct ggml_tensor * t10 = ggml_reshape_2d(ctx, attn, n_embd, n_tokens);
        // attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
        // attn = ggml_cont(ctx, attn);  // Make contiguous BEFORE reshape
        // struct ggml_tensor * t10 = ggml_reshape_2d(ctx, attn, n_embd, n_tokens);
        // attn = ggml_cont_2d(ctx, attn, n_embd, n_tokens);

        // Output projection
        struct ggml_tensor * attn_out = ggml_mul_mat(ctx, w_out_proj, attn);
        ggml_set_name(attn_out, "debug_attn_out");

        // Layer scale
        if (w_scale1) {
            attn_out = ggml_mul(ctx, attn_out, w_scale1);
            ggml_set_name(attn_out, "debug_attn_scaled");
        }

        // Residual
        cur = ggml_add(ctx, attn_out, inpSA);
        ggml_set_name(cur, "debug_after_res1");

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

        // Fill causal mask: mask[i, j] = 0 if i <= j, -inf if i > j
        // This allows query j to attend to keys 0..j (causal attention)
        struct ggml_tensor * kq_mask_tensor = ggml_get_tensor(ctx, "KQ_mask");
        if (kq_mask_tensor) {
            std::vector<float> mask_data(n_tokens * n_tokens);
            for (int64_t i = 0; i < n_tokens; i++) {      // key position
                for (int64_t j = 0; j < n_tokens; j++) {  // query position
                    // mask[i, j] with i = key, j = query
                    // Query j can attend to key i if i <= j
                    mask_data[i + j * n_tokens] = (i <= j) ? 0.0f : -INFINITY;
                }
            }
            ggml_backend_tensor_set(kq_mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(float));
        }

        // Compute
        enum ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "Error: backend compute failed for layer %d, status=%d\n", il, (int)status);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return {};
        }

        // Debug: print intermediate values for layer 0
        if (il == 0) {
            // Get norm1 output
            struct ggml_tensor * dbg_norm1 = ggml_get_tensor(ctx, "debug_norm1");
            struct ggml_tensor * dbg_qkv = ggml_get_tensor(ctx, "debug_qkv");

            if (dbg_norm1) {
                std::vector<float> norm1_data(n_embd * n_tokens);
                ggml_backend_tensor_get(dbg_norm1, norm1_data.data(), 0, norm1_data.size() * sizeof(float));
                printf("    DEBUG norm1 first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       norm1_data[0], norm1_data[1], norm1_data[2], norm1_data[3]);
                fflush(stdout);
            }

            if (dbg_qkv) {
                std::vector<float> qkv_data(3 * n_embd * n_tokens);
                ggml_backend_tensor_get(dbg_qkv, qkv_data.data(), 0, qkv_data.size() * sizeof(float));
                printf("    DEBUG qkv Q first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       qkv_data[0], qkv_data[1], qkv_data[2], qkv_data[3]);
                printf("    DEBUG qkv K first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       qkv_data[512], qkv_data[513], qkv_data[514], qkv_data[515]);
                fflush(stdout);
            }

            // Get Q/K after RoPE
            struct ggml_tensor * dbg_q_rope = ggml_get_tensor(ctx, "debug_q_rope");
            struct ggml_tensor * dbg_k_rope = ggml_get_tensor(ctx, "debug_k_rope");

            if (dbg_q_rope) {
                // Shape: [head_dim, n_head, n_tokens] = [64, 8, 50]
                std::vector<float> q_rope_data(n_embd_head * n_head * n_tokens);
                ggml_backend_tensor_get(dbg_q_rope, q_rope_data.data(), 0, q_rope_data.size() * sizeof(float));
                // First head, first token, first 4 elements
                printf("    DEBUG Q after RoPE head0 t=0 first 4: %.6f %.6f %.6f %.6f\n",
                       q_rope_data[0], q_rope_data[1], q_rope_data[2], q_rope_data[3]);
                // First head, second token (offset by head_dim * n_head = 64*8 = 512)
                printf("    DEBUG Q after RoPE head0 t=1 first 4: %.6f %.6f %.6f %.6f\n",
                       q_rope_data[512], q_rope_data[513], q_rope_data[514], q_rope_data[515]);
                fflush(stdout);
            }

            // Debug attention outputs
            struct ggml_tensor * dbg_attn_out = ggml_get_tensor(ctx, "debug_attn_out");
            struct ggml_tensor * dbg_attn_scaled = ggml_get_tensor(ctx, "debug_attn_scaled");
            struct ggml_tensor * dbg_after_res1 = ggml_get_tensor(ctx, "debug_after_res1");

            if (dbg_attn_out) {
                std::vector<float> data(n_embd * n_tokens);
                ggml_backend_tensor_get(dbg_attn_out, data.data(), 0, data.size() * sizeof(float));
                printf("    DEBUG attn_out first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       data[0], data[1], data[2], data[3]);
                fflush(stdout);
            }
            if (dbg_attn_scaled) {
                std::vector<float> data(n_embd * n_tokens);
                ggml_backend_tensor_get(dbg_attn_scaled, data.data(), 0, data.size() * sizeof(float));
                printf("    DEBUG attn_scaled first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       data[0], data[1], data[2], data[3]);
                fflush(stdout);
            }
            if (dbg_after_res1) {
                std::vector<float> data(n_embd * n_tokens);
                ggml_backend_tensor_get(dbg_after_res1, data.data(), 0, data.size() * sizeof(float));
                printf("    DEBUG after_res1 first 4 at t=0: %.6f %.6f %.6f %.6f\n",
                       data[0], data[1], data[2], data[3]);
                fflush(stdout);
            }
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

    // Step 4: SEANet decoder using GGML (or pure C++ if USE_CPP_SEANET is set)
    std::vector<float> audio;
    int T_final = n_tokens_final;  // Token count after transformer (T_up)

    // Option to load SEANet input from external binary file (for isolated testing)
    // Format: int d_model, int seq_len, float data[d_model * seq_len]
    const char * seanet_input_file = getenv("SEANET_INPUT_BIN");
    if (seanet_input_file) {
        printf("  Loading SEANet input from %s...\n", seanet_input_file);
        FILE * f = fopen(seanet_input_file, "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open %s\n", seanet_input_file);
            return {};
        }
        int d_model_file, seq_len_file;
        fread(&d_model_file, sizeof(int), 1, f);
        fread(&seq_len_file, sizeof(int), 1, f);
        x.resize(d_model_file * seq_len_file);
        fread(x.data(), sizeof(float), x.size(), f);
        fclose(f);
        T_final = seq_len_file;
        printf("  Loaded SEANet input: D=%d, T=%d\n", d_model_file, seq_len_file);
        printf("  SEANet input range: [%.4f, %.4f], first 4 ch@t=0: %.4f %.4f %.4f %.4f\n",
               *std::min_element(x.begin(), x.end()), *std::max_element(x.begin(), x.end()),
               x[0], x[seq_len_file], x[2*seq_len_file], x[3*seq_len_file]);
    }

    // GGML-based SEANet (default)
    audio = seanet_decode_ggml(model, x, T_final);
    if (audio.empty()) {
        fprintf(stderr, "Error: SEANet GGML decode failed\n");
        return {};
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
