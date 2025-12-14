#!/bin/bash
# Moshi STT Test Script
# Usage: test_stt.sh <moshi_stt_binary> [model_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOSHI_STT="${1:-./build/moshi_stt}"
MODEL_PATH="${2:-/tmp/moshi-stt-f32.gguf}"
TEST_JSON="${SCRIPT_DIR}/test_audio_tokens.json"

# Expected tokens for "quick brown fox jumps over the lazy dog. How are you doing today?"
EXPECTED_TOKENS="2457 7289 700 2903 3213 263 580 267 272 5343 2734 264 1268 373 276 701 1066 339"

echo "=== Moshi STT Test ==="
echo "Binary: ${MOSHI_STT}"
echo "Model:  ${MODEL_PATH}"
echo "Input:  ${TEST_JSON}"
echo ""

# Check prerequisites
if [ ! -f "${MOSHI_STT}" ]; then
    echo "ERROR: moshi_stt binary not found: ${MOSHI_STT}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found: ${MODEL_PATH}"
    echo "Generate it with: python scripts/convert_moshi_to_gguf.py --stt"
    exit 1
fi

if [ ! -f "${TEST_JSON}" ]; then
    echo "ERROR: Test data not found: ${TEST_JSON}"
    exit 1
fi

# Set library path for ggml (LLAMA_BUILD_DIR should be set by user)
if [ -n "${LLAMA_BUILD_DIR}" ]; then
    export DYLD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/bin:${DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/bin:${LD_LIBRARY_PATH}"
fi

# Run inference
echo "Running STT inference..."
RESULT=$("${MOSHI_STT}" "${TEST_JSON}" -m "${MODEL_PATH}")

echo "Output tokens: ${RESULT}"
echo "Expected:      ${EXPECTED_TOKENS}"

# Compare
if [ "${RESULT}" = "${EXPECTED_TOKENS}" ]; then
    echo ""
    echo "PASS: Output matches expected tokens"
    exit 0
else
    echo ""
    echo "FAIL: Output does not match expected"
    exit 1
fi
