#ifndef MIMI_DECODE_UTILS_H
#define MIMI_DECODE_UTILS_H

#include <vector>
#include <string>

// Simple JSON parsing for audio tokens
std::vector<std::vector<int32_t>> parse_audio_tokens(const std::string & json_path);

// Save WAV file
void save_wav(const std::string & path, const std::vector<float> & audio, int sample_rate);

#endif
