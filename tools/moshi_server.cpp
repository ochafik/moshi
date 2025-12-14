// Moshi Audio Server - HTTP API for Speech-to-Text
// OpenAI-compatible /v1/audio/transcriptions endpoint
//
// Endpoints:
//   POST /v1/audio/transcriptions - Transcribe audio file to text
//   POST /v1/stt                  - Transcribe audio tokens to text tokens
//   GET  /health                  - Health check
//
// Usage: moshi_server [options]
//
// Copyright (c) 2024 Anthropic. All rights reserved.
// SPDX-License-Identifier: MIT

#include "moshi_stt.h"
#include "mimi.h"

// Disable SSL support in httplib (we don't need HTTPS for local server)
#include "httplib.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <sstream>
#include <unistd.h>

// Global models
static std::unique_ptr<mimi::MimiEncoder> g_mimi_encoder;
static std::unique_ptr<moshi::STTModel> g_stt_model;
static bool g_models_loaded = false;

// Simple JSON builder
static std::string json_escape(const std::string & s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c;
        }
    }
    return result;
}

static std::string json_error(const std::string & message) {
    return "{\"error\": \"" + json_escape(message) + "\"}";
}

static std::string json_tokens(const std::vector<int32_t> & tokens) {
    std::string result = "{\"tokens\": [";
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) result += ", ";
        result += std::to_string(tokens[i]);
    }
    result += "]}";
    return result;
}

static void print_usage(const char * program) {
    fprintf(stderr, "Moshi Audio Server - OpenAI-compatible STT API\n\n");
    fprintf(stderr, "Usage: %s [options]\n\n", program);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --host <addr>         Host address (default: 127.0.0.1)\n");
    fprintf(stderr, "  --port <port>         Port number (default: 8080)\n");
    fprintf(stderr, "  --mimi <path>         Mimi encoder GGUF (default: /tmp/mimi-encoder.gguf)\n");
    fprintf(stderr, "  --stt <path>          STT model GGUF (default: /tmp/moshi-stt-f32.gguf)\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Endpoints:\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions   Transcribe audio file (OpenAI-compatible)\n");
    fprintf(stderr, "  POST /v1/stt                    Transcribe audio tokens\n");
    fprintf(stderr, "  GET  /health                    Health check\n");
}

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string mimi_path = "/tmp/mimi-encoder.gguf";
    std::string stt_path = "/tmp/moshi-stt-f32.gguf";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--mimi") == 0 && i + 1 < argc) {
            mimi_path = argv[++i];
        } else if (strcmp(argv[i], "--stt") == 0 && i + 1 < argc) {
            stt_path = argv[++i];
        }
    }

    // Load Mimi encoder
    fprintf(stderr, "Loading Mimi encoder from %s...\n", mimi_path.c_str());
    g_mimi_encoder = std::make_unique<mimi::MimiEncoder>();
    if (!g_mimi_encoder->load(mimi_path)) {
        fprintf(stderr, "Error: failed to load Mimi encoder\n");
        return 1;
    }
    const auto & mimi_hp = g_mimi_encoder->get_hparams();
    fprintf(stderr, "Mimi encoder loaded: sample_rate=%d, frame_rate=%.1f, codebooks=%d\n",
            mimi_hp.sample_rate, mimi_hp.frame_rate, mimi_hp.n_codebooks);

    // Load STT model
    fprintf(stderr, "Loading STT model from %s...\n", stt_path.c_str());
    g_stt_model = std::make_unique<moshi::STTModel>();
    if (!g_stt_model->load(stt_path)) {
        fprintf(stderr, "Error: failed to load STT model\n");
        return 1;
    }
    const auto & stt_hp = g_stt_model->get_hparams();
    fprintf(stderr, "STT model loaded: dim=%d, layers=%d, heads=%d, codebooks=%d\n",
            stt_hp.dim, stt_hp.n_layers, stt_hp.n_heads, stt_hp.n_codebooks);

    g_models_loaded = true;

    // Create HTTP server
    httplib::Server svr;

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        std::string json = "{\"status\": \"ok\", \"models_loaded\": " +
                          std::string(g_models_loaded ? "true" : "false") + "}";
        res.set_content(json, "application/json");
    });

    // OpenAI-compatible transcription endpoint
    svr.Post("/v1/audio/transcriptions", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        if (!g_models_loaded) {
            res.status = 503;
            res.set_content(json_error("Models not loaded"), "application/json");
            return;
        }

        // Check for file in multipart form
        if (!req.form.has_file("file")) {
            res.status = 400;
            res.set_content(json_error("No audio file provided"), "application/json");
            return;
        }

        auto file = req.form.get_file("file");
        if (file.content.empty()) {
            res.status = 400;
            res.set_content(json_error("Empty audio file"), "application/json");
            return;
        }

        // Save to temp file
        char temp_path[] = "/tmp/moshi_audio_XXXXXX.wav";
        int fd = mkstemps(temp_path, 4);
        if (fd < 0) {
            res.status = 500;
            res.set_content(json_error("Failed to create temp file"), "application/json");
            return;
        }
        write(fd, file.content.data(), file.content.size());
        close(fd);

        // Load audio
        auto audio = mimi::load_wav(temp_path);
        unlink(temp_path);  // Cleanup

        if (audio.empty()) {
            res.status = 400;
            res.set_content(json_error("Failed to decode audio file"), "application/json");
            return;
        }

        fprintf(stderr, "[transcriptions] Processing %zu samples (%.2f sec)\n",
                audio.size(), audio.size() / 24000.0f);

        // Step 1: Mimi encode audio -> tokens
        auto encode_result = g_mimi_encoder->encode(audio, false);
        if (encode_result.tokens.empty()) {
            res.status = 500;
            res.set_content(json_error("Failed to encode audio"), "application/json");
            return;
        }

        fprintf(stderr, "[transcriptions] Encoded to %d frames\n", encode_result.n_frames);

        // Step 2: STT transcribe tokens -> text tokens
        auto stt_result = g_stt_model->transcribe(encode_result.tokens, false);

        fprintf(stderr, "[transcriptions] Transcribed to %zu text tokens\n",
                stt_result.text_tokens.size());

        // Get response format (default: json)
        std::string response_format = "json";
        if (req.has_param("response_format")) {
            response_format = req.get_param_value("response_format");
        }

        // Build response - for now return text tokens as comma-separated list
        // TODO: Add SentencePiece tokenizer for proper text decoding
        std::ostringstream text_stream;
        text_stream << "[";
        for (size_t i = 0; i < stt_result.text_tokens.size(); i++) {
            if (i > 0) text_stream << ",";
            text_stream << stt_result.text_tokens[i];
        }
        text_stream << "]";
        std::string text = text_stream.str();

        if (response_format == "text") {
            res.set_content(text, "text/plain");
        } else if (response_format == "verbose_json") {
            std::ostringstream json;
            json << "{";
            json << "\"task\":\"transcribe\",";
            json << "\"language\":\"en\",";
            json << "\"duration\":" << encode_result.duration << ",";
            json << "\"text\":\"" << json_escape(text) << "\",";
            json << "\"tokens\":[";
            for (size_t i = 0; i < stt_result.text_tokens.size(); i++) {
                if (i > 0) json << ",";
                json << stt_result.text_tokens[i];
            }
            json << "]}";
            res.set_content(json.str(), "application/json");
        } else {
            // json (default)
            res.set_content("{\"text\":\"" + json_escape(text) + "\"}", "application/json");
        }
    });

    // Legacy STT endpoint (takes audio tokens directly)
    svr.Post("/v1/stt", [&stt_hp](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        if (!g_models_loaded) {
            res.status = 503;
            res.set_content(json_error("Models not loaded"), "application/json");
            return;
        }

        // Parse audio tokens from request body
        auto frames = moshi::parse_audio_tokens_json(req.body, stt_hp.n_codebooks);
        if (frames.empty()) {
            res.status = 400;
            res.set_content(json_error("Invalid audio_tokens format"), "application/json");
            return;
        }

        // Transcribe
        auto result = g_stt_model->transcribe(frames, false);

        // Return tokens
        res.set_content(json_tokens(result.text_tokens), "application/json");
    });

    // CORS preflight handlers
    svr.Options("/v1/audio/transcriptions", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    svr.Options("/v1/stt", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    // Start server
    fprintf(stderr, "\nMoshi Audio Server listening on http://%s:%d\n", host.c_str(), port);
    fprintf(stderr, "Endpoints:\n");
    fprintf(stderr, "  POST /v1/audio/transcriptions - Transcribe audio file (OpenAI-compatible)\n");
    fprintf(stderr, "  POST /v1/stt                  - Transcribe audio tokens\n");
    fprintf(stderr, "  GET  /health                  - Health check\n\n");

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "Error: failed to start server\n");
        return 1;
    }

    return 0;
}
