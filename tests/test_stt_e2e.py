#!/usr/bin/env python3
"""
End-to-end tests for Moshi STT /v1/audio/transcriptions endpoint.

Generates ground truth audio using macOS TTS (or other engines),
then verifies the transcription endpoint returns correct text.

Usage:
    # Start server first:
    python scripts/moshi_server.py --port 8080

    # Run tests:
    pytest tests/test_stt_e2e.py -v

    # Or run standalone:
    python tests/test_stt_e2e.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import requests

# Configuration
SERVER_URL = os.environ.get("MOSHI_SERVER_URL", "http://127.0.0.1:8080")
TEST_DATA_DIR = Path(__file__).parent / "audio_ground_truth"


def generate_audio_macos(text: str, output_path: str, voice: str = "Samantha", rate: int = 180) -> bool:
    """Generate audio using macOS say command."""
    try:
        # Generate AIFF first (say outputs AIFF natively)
        aiff_path = output_path.replace(".wav", ".aiff")
        cmd = ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text]
        subprocess.run(cmd, check=True, capture_output=True)

        # Convert to WAV using afconvert (macOS built-in)
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEF32@24000", aiff_path, output_path]
        subprocess.run(cmd, check=True, capture_output=True)

        # Cleanup AIFF
        os.unlink(aiff_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating audio: {e}")
        return False
    except FileNotFoundError:
        print("macOS 'say' command not available")
        return False


def generate_audio_espeak(text: str, output_path: str) -> bool:
    """Generate audio using espeak (Linux/cross-platform)."""
    try:
        cmd = ["espeak", "-w", output_path, "-s", "150", text]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def generate_audio(text: str, output_path: str) -> bool:
    """Generate audio using available TTS engine."""
    # Try macOS first
    if generate_audio_macos(text, output_path):
        return True
    # Fall back to espeak
    if generate_audio_espeak(text, output_path):
        return True
    return False


def transcribe_audio(audio_path: str, response_format: str = "json") -> dict:
    """Call the transcription endpoint."""
    url = f"{SERVER_URL}/v1/audio/transcriptions"

    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {"response_format": response_format}
        response = requests.post(url, files=files, data=data, timeout=120)

    response.raise_for_status()

    if response_format == "text":
        return {"text": response.text}
    return response.json()


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    # Levenshtein distance at word level
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / max(len(ref_words), 1)


# Test cases: (ground_truth_text, max_wer)
TEST_CASES = [
    ("Hello world", 0.5),
    ("The quick brown fox jumps over the lazy dog", 0.3),
    ("How are you doing today", 0.3),
    ("This is a test of the speech recognition system", 0.3),
    ("One two three four five", 0.3),
]


class TestSTTEndpoint(unittest.TestCase):
    """Test suite for STT transcription endpoint."""

    @classmethod
    def setUpClass(cls):
        """Check server is running and create test data directory."""
        cls.test_dir = TEST_DATA_DIR
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Check server health
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if not health.get("models_loaded"):
                raise RuntimeError("Server models not loaded")
            print(f"Server ready: {health}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to server at {SERVER_URL}. "
                "Start it with: python scripts/moshi_server.py"
            )

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{SERVER_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["models_loaded"])

    def test_transcription_missing_file(self):
        """Test error handling for missing file."""
        response = requests.post(f"{SERVER_URL}/v1/audio/transcriptions")
        self.assertEqual(response.status_code, 400)

    def test_transcription_response_formats(self):
        """Test different response formats."""
        # Generate test audio
        test_text = "Hello world"
        audio_path = self.test_dir / "test_format.wav"

        if not audio_path.exists():
            self.assertTrue(generate_audio(test_text, str(audio_path)),
                           "Failed to generate test audio")

        # Test JSON format (default)
        result = transcribe_audio(str(audio_path), "json")
        self.assertIn("text", result)

        # Test verbose_json format
        result = transcribe_audio(str(audio_path), "verbose_json")
        self.assertIn("text", result)
        self.assertIn("duration", result)

        # Test text format
        result = transcribe_audio(str(audio_path), "text")
        self.assertIn("text", result)


def run_ground_truth_tests():
    """Run ground truth transcription tests."""
    print("=" * 60)
    print("Moshi STT End-to-End Tests")
    print("=" * 60)

    # Check server
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"Server: {SERVER_URL} - OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {SERVER_URL}")
        print("Start it with: python scripts/moshi_server.py")
        return False

    # Create test directory
    test_dir = TEST_DATA_DIR
    test_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, (ground_truth, max_wer) in enumerate(TEST_CASES):
        print(f"\nTest {i + 1}/{len(TEST_CASES)}: {ground_truth[:40]}...")

        # Generate audio file path
        safe_name = "".join(c if c.isalnum() else "_" for c in ground_truth[:30])
        audio_path = test_dir / f"gt_{i:02d}_{safe_name}.wav"

        # Generate audio if needed
        if not audio_path.exists():
            print(f"  Generating audio with TTS...")
            if not generate_audio(ground_truth, str(audio_path)):
                print(f"  SKIP: Could not generate audio")
                results.append(("SKIP", ground_truth, None, None))
                continue

        # Transcribe
        print(f"  Transcribing...")
        try:
            start = time.time()
            result = transcribe_audio(str(audio_path), "verbose_json")
            elapsed = time.time() - start
            transcription = result["text"]
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append(("FAIL", ground_truth, str(e), None))
            continue

        # Calculate WER
        wer = word_error_rate(ground_truth, transcription)

        print(f"  Ground truth: {ground_truth}")
        print(f"  Transcribed:  {transcription}")
        print(f"  WER: {wer:.2%} (max: {max_wer:.0%})")
        print(f"  Time: {elapsed:.2f}s")

        if wer <= max_wer:
            print(f"  PASS")
            results.append(("PASS", ground_truth, transcription, wer))
        else:
            print(f"  FAIL: WER too high")
            results.append(("FAIL", ground_truth, transcription, wer))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    skipped = sum(1 for r in results if r[0] == "SKIP")

    print(f"Passed:  {passed}/{len(TEST_CASES)}")
    print(f"Failed:  {failed}/{len(TEST_CASES)}")
    print(f"Skipped: {skipped}/{len(TEST_CASES)}")

    # Save results
    results_file = test_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": SERVER_URL,
            "results": [
                {
                    "status": r[0],
                    "ground_truth": r[1],
                    "transcription": r[2],
                    "wer": r[3],
                }
                for r in results
            ]
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return failed == 0


if __name__ == "__main__":
    # Check if running as pytest or standalone
    if "pytest" in sys.modules:
        # pytest will handle it
        pass
    else:
        # Run standalone tests
        success = run_ground_truth_tests()
        sys.exit(0 if success else 1)
