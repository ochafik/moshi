#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Simple JSON parsing for audio tokens
std::vector<std::vector<int32_t>> parse_audio_tokens(const std::string & json_path) {
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
void save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate) {
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
