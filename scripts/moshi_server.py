#!/usr/bin/env python3
"""
Moshi Audio Server - OpenAI-compatible /v1/audio/transcriptions endpoint.

Usage:
    python moshi_server.py [--host 0.0.0.0] [--port 8080] [--device cpu]

Endpoints:
    POST /v1/audio/transcriptions - Transcribe audio to text
    GET  /health                  - Health check
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
torch._dynamo.config.suppress_errors = True

# Import Flask for REST API
try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Error: Flask required. Install with: pip install flask")
    sys.exit(1)


app = Flask(__name__)

# Global model instances (loaded on startup)
_models = {
    "mimi": None,
    "lm": None,
    "tokenizer": None,
    "lm_gen_config": None,
    "stt_config": None,
    "device": "cpu",
}


def load_models(device: str = "cpu"):
    """Load Moshi models on startup."""
    from moshi.models import loaders, LMGen

    print(f"Loading Moshi STT models on {device}...")
    start = time.time()

    stt_repo = "kyutai/stt-1b-en_fr"
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(stt_repo)

    _models["mimi"] = checkpoint_info.get_mimi(device=device)
    _models["lm"] = checkpoint_info.get_moshi(device=device, dtype=torch.float32)
    _models["tokenizer"] = checkpoint_info.get_text_tokenizer()
    _models["lm_gen_config"] = checkpoint_info.lm_gen_config
    _models["stt_config"] = checkpoint_info.stt_config
    _models["device"] = device

    elapsed = time.time() - start
    print(f"Models loaded in {elapsed:.1f}s")


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio file to text."""
    import sphn
    from moshi.models import LMGen

    mimi = _models["mimi"]
    lm = _models["lm"]
    tokenizer = _models["tokenizer"]
    device = _models["device"]
    lm_gen_config = _models["lm_gen_config"]
    stt_config = _models["stt_config"]

    # Load audio
    wav, sr = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    wav = torch.from_numpy(wav).to(device=device)

    # Ensure [B, C, T] shape
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)

    # Pad audio per STT config
    pad_left = int(stt_config.get('audio_silence_prefix_seconds', 0.0) * 24000)
    pad_right = int((stt_config.get('audio_delay_seconds', 0.0) + 1.0) * 24000)
    wav = torch.nn.functional.pad(wav, (pad_left, pad_right), mode='constant')

    # Calculate duration
    duration = wav.shape[-1] / mimi.sample_rate

    # Run streaming inference
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(lm, **lm_gen_config)

    text_tokens = []
    start_time = time.time()

    with lm_gen.streaming(1), mimi.streaming(1):
        chunks = [chunk for chunk in wav.split(frame_size, dim=2) if chunk.shape[-1] == frame_size]

        for chunk in chunks:
            codes = mimi.encode(chunk)
            tokens = lm_gen.step(codes)

            if tokens is not None:
                text_token = tokens[0, 0].item()
                text_tokens.append(text_token)

    # Decode text
    valid_tokens = [t for t in text_tokens if t >= 4 and t < tokenizer.GetPieceSize()]
    text = tokenizer.Decode(valid_tokens)

    elapsed = time.time() - start_time

    return {
        "text": text.strip(),
        "tokens": valid_tokens,
        "duration": duration,
        "processing_time": elapsed,
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models_loaded": _models["mimi"] is not None,
        "device": _models["device"],
    })


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcriptions():
    """
    OpenAI-compatible audio transcription endpoint.

    Request: multipart/form-data with 'file' field containing audio
    Response: {"text": "transcription..."}
    """
    # Check if file was provided
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]

    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file
    suffix = Path(audio_file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)

        # Check response_format parameter
        response_format = request.form.get("response_format", "json")

        if response_format == "text":
            return result["text"], 200, {"Content-Type": "text/plain"}
        elif response_format == "verbose_json":
            return jsonify({
                "task": "transcribe",
                "language": "en",
                "duration": result["duration"],
                "text": result["text"],
                "tokens": result["tokens"],
            })
        else:  # json (default)
            return jsonify({"text": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.route("/v1/audio/speech", methods=["POST"])
def speech():
    """
    Text-to-speech endpoint (placeholder for future TTS implementation).
    """
    return jsonify({"error": "TTS not yet implemented"}), 501


def main():
    parser = argparse.ArgumentParser(description="Moshi Audio Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    # Load models
    load_models(args.device)

    print(f"\nMoshi Audio Server starting on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /v1/audio/transcriptions - Transcribe audio")
    print("  GET  /health                  - Health check")
    print("")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
