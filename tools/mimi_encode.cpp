// Mimi Neural Audio Codec Encoder
// Standalone tool to encode WAV audio to discrete tokens
// Uses ggml for tensor operations

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "mimi.h"

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <input.wav> [-o output.json] [-m mimi-encoder.gguf] [-v]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -o <path>   Output JSON file for tokens (default: input_tokens.json)\n");
    fprintf(stderr, "  -m <path>   Path to Mimi encoder GGUF model\n");
    fprintf(stderr, "  -v          Verbose output\n");
}

static bool save_tokens_json(const std::string & path,
                              const std::vector<std::vector<int32_t>> & tokens,
                              float duration) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open %s for writing\n", path.c_str());
        return false;
    }

    f << "{\n";
    f << "  \"duration\": " << duration << ",\n";
    f << "  \"n_frames\": " << tokens.size() << ",\n";
    f << "  \"n_codebooks\": " << (tokens.empty() ? 0 : tokens[0].size()) << ",\n";
    f << "  \"audio_tokens\": [\n";

    for (size_t t = 0; t < tokens.size(); t++) {
        f << "    [";
        for (size_t q = 0; q < tokens[t].size(); q++) {
            f << tokens[t][q];
            if (q < tokens[t].size() - 1) f << ", ";
        }
        f << "]";
        if (t < tokens.size() - 1) f << ",";
        f << "\n";
    }

    f << "  ]\n";
    f << "}\n";

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = "";
    std::string model_path = "/tmp/mimi-encoder.gguf";
    bool verbose = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Default output path
    if (output_path.empty()) {
        output_path = input_path;
        size_t pos = output_path.rfind('.');
        if (pos != std::string::npos) {
            output_path = output_path.substr(0, pos);
        }
        output_path += "_tokens.json";
    }

    // Load audio
    printf("Loading audio from %s...\n", input_path.c_str());
    auto audio = mimi::load_wav(input_path);
    if (audio.empty()) {
        fprintf(stderr, "Error: failed to load audio\n");
        return 1;
    }
    printf("Loaded %zu samples (%.2f seconds at 24kHz)\n", audio.size(), audio.size() / 24000.0f);

    // Load model
    printf("Loading encoder model from %s...\n", model_path.c_str());
    mimi::MimiEncoder encoder;
    if (!encoder.load(model_path)) {
        fprintf(stderr, "Error: failed to load encoder model\n");
        return 1;
    }

    const auto & hp = encoder.get_hparams();
    printf("Model params:\n");
    printf("  sample_rate: %d\n", hp.sample_rate);
    printf("  frame_rate: %.1f\n", hp.frame_rate);
    printf("  n_codebooks: %d\n", hp.n_codebooks);
    printf("  codebook_size: %d\n", hp.codebook_size);
    printf("  encoder_dim: %d\n", hp.encoder_dim);

    // Encode
    printf("Encoding...\n");
    auto result = encoder.encode(audio, verbose);

    printf("Encoded %d frames (%.2f seconds)\n", result.n_frames, result.duration);

    // Print sample tokens
    if (verbose && !result.tokens.empty()) {
        printf("First frame tokens: [");
        for (size_t q = 0; q < result.tokens[0].size(); q++) {
            printf("%d", result.tokens[0][q]);
            if (q < result.tokens[0].size() - 1) printf(", ");
        }
        printf("]\n");
    }

    // Save tokens
    if (!save_tokens_json(output_path, result.tokens, result.duration)) {
        return 1;
    }

    printf("Saved tokens to %s\n", output_path.c_str());
    return 0;
}
