# Moshi TTS - Mac App

A standalone macOS application for text-to-speech using the Moshi TTS model on Apple Silicon with MLX.

## Features

- Native Apple Silicon support via MLX
- Simple GUI for text-to-speech
- Multiple voice selection
- Save audio to WAV files
- 4-bit quantization for faster inference

## Requirements

- macOS 12.0 (Monterey) or later
- Apple Silicon Mac (M1/M2/M3)
- Python 3.10 or later
- ~5GB disk space for models (downloaded on first run)

## Quick Start

### Run from Source

```bash
# Clone the repository (if not already done)
cd moshi/tts_mac_app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install moshi_mlx from parent directory
pip install -e ../moshi_mlx

# Run the app
python tts_app.py
```

### Build Standalone App

```bash
# Activate virtual environment
source venv/bin/activate

# Build the app (this may take a few minutes)
python setup.py py2app

# The app will be in dist/Moshi TTS.app
open "dist/Moshi TTS.app"
```

### Development Mode

For faster iteration during development, use alias mode:

```bash
python setup.py py2app -A
```

This creates a lightweight app that references your source files.

## Usage

1. **Launch the app** - The models will be downloaded automatically on first run (~5GB)
2. **Select a voice** - Choose from the available voices in the dropdown
3. **Enter text** - Type or paste the text you want to synthesize
4. **Click "Speak"** - The audio will be generated and played automatically
5. **Save audio** - Click "Save Audio..." to export as a WAV file

## Model Configuration

The app uses the following default settings:

| Setting | Value | Description |
|---------|-------|-------------|
| Model | `kyutai/tts-1.6b-en_fr` | HuggingFace model repository |
| Quantization | 4-bit | Reduces memory and speeds up inference |
| Temperature | 0.6 | Controls randomness in generation |
| CFG Coefficient | 2.0 | Classifier-free guidance strength |

## Troubleshooting

### "Models not loading"

- Ensure you have internet access for the first run
- Check you have ~5GB free disk space
- Models are cached in `~/.cache/huggingface/`

### "No voices found"

- The voice files should download automatically
- Check `~/.cache/huggingface/` for the voice repository

### "Audio not playing"

- The app uses `afplay` (built into macOS) to play audio
- Try saving the audio and playing it with another app

### "App crashes on launch"

- Ensure you're on Apple Silicon (M1/M2/M3)
- This app does not support Intel Macs
- Try rebuilding with: `rm -rf build dist && python setup.py py2app`

## Technical Details

This app is based on the MLX TTS implementation from `rust/moshi-server/tts_mlx.py`. It provides:

- **TTSEngine**: Wraps the MLX model loading and inference
- **TTSApp**: tkinter GUI for user interaction
- **py2app**: Packages everything into a `.app` bundle

### Architecture

```
tts_mac_app/
├── tts_app.py      # Main application (GUI + TTS engine)
├── setup.py        # py2app build configuration
├── requirements.txt # Python dependencies
└── README.md       # This file
```

### Dependencies

The app depends on:

- `mlx` - Apple's ML framework for Apple Silicon
- `moshi_mlx` - MLX-based Moshi model implementation
- `huggingface_hub` - Model downloading
- `sentencepiece` - Text tokenization
- `sphn` - Audio I/O
- `tkinter` - GUI (included with Python)

## License

Copyright (c) Kyutai, all rights reserved.
