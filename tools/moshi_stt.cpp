// Moshi Speech-to-Text CLI Tool
// Transcribes audio tokens to text using the STT model
//
// Usage: moshi_stt <audio_tokens.json> [-m model.gguf] [-v]
//
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#include "moshi_stt.h"

#include <cstdio>
#include <cstring>

static void print_usage(const char * program) {
    fprintf(stderr, "Moshi Speech-to-Text\n\n");
    fprintf(stderr, "Usage: %s <audio_tokens.json> [options]\n\n", program);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>    Path to GGUF model (default: /tmp/moshi-stt-f32.gguf)\n");
    fprintf(stderr, "  -v, --verbose         Verbose output\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Input format:\n");
    fprintf(stderr, "  JSON file with {\"audio_tokens\": [[codebook0], [codebook1], ...]}\n\n");
    fprintf(stderr, "Output:\n");
    fprintf(stderr, "  Text token IDs (decode with SentencePiece tokenizer)\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string json_path;
    std::string model_path = "/tmp/moshi-stt-f32.gguf";
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (argv[i][0] != '-') {
            json_path = argv[i];
        }
    }

    if (json_path.empty()) {
        fprintf(stderr, "Error: no input file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    moshi::STTModel model;
    if (verbose) {
        fprintf(stderr, "Loading model from %s...\n", model_path.c_str());
    }
    if (!model.load(model_path)) {
        return 1;
    }
    if (verbose) {
        const auto & hp = model.get_hparams();
        fprintf(stderr, "Model: dim=%d, layers=%d, heads=%d, codebooks=%d\n",
                hp.dim, hp.n_layers, hp.n_heads, hp.n_codebooks);
    }

    // Load audio tokens
    auto frames = moshi::load_audio_tokens(json_path, model.get_hparams().n_codebooks);
    if (frames.empty()) {
        fprintf(stderr, "Error: failed to load audio tokens from %s\n", json_path.c_str());
        return 1;
    }
    if (verbose) {
        fprintf(stderr, "Loaded %zu frames\n", frames.size());
    }

    // Transcribe
    auto result = model.transcribe(frames, verbose);

    // Output results
    if (verbose) {
        fprintf(stderr, "\n=== Results ===\n");
        fprintf(stderr, "Frames: %d\n", result.n_frames);
        fprintf(stderr, "Text tokens: %zu\n", result.text_tokens.size());
    }

    // Print valid text tokens (machine-readable output)
    for (size_t i = 0; i < result.text_tokens.size(); i++) {
        if (i > 0) printf(" ");
        printf("%d", result.text_tokens[i]);
    }
    printf("\n");

    return 0;
}
