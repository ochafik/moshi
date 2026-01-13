# Copyright (c) Kyutai, all rights reserved.
# Standalone MLX TTS Mac App - Apple Silicon native text-to-speech
"""
A standalone Mac application for MLX-based text-to-speech on Apple Silicon.

This app provides a simple GUI to:
- Enter text to synthesize
- Select from available voices
- Generate and play/save audio

Based on moshi_mlx TTS implementation.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

import numpy as np


# =============================================================================
# MLX TTS Engine - Extracted from tts_mlx.py
# =============================================================================
class TTSEngine:
    """MLX-based TTS engine for Apple Silicon."""

    def __init__(self):
        self.model = None
        self.mimi = None
        self.text_tokenizer = None
        self.tts_model = None
        self.all_attributes = {}
        self.default_voice = None
        self.loaded = False
        self.loading = False
        self.load_error = None

    def load_models(
        self,
        hf_repo: str = "kyutai/tts-1.6b-en_fr",
        voice_folder: str = "hf-snapshot://kyutai/tts-voices/unmute-prod-website/*.safetensors",
        default_voice: str = "unmute-prod-website/default_voice.wav",
        quantize: Optional[int] = 4,
        progress_callback=None,
    ):
        """Load the TTS models from HuggingFace."""
        import huggingface_hub
        import mlx.core as mx
        import mlx.nn as nn
        import sentencepiece

        # Add moshi_mlx to path if needed
        moshi_mlx_path = Path(__file__).parent.parent / "moshi_mlx"
        if moshi_mlx_path.exists():
            sys.path.insert(0, str(moshi_mlx_path))
        else:
            # Try home directory
            home_moshi = Path.home() / "github/moshi/moshi_mlx"
            if home_moshi.exists():
                sys.path.insert(0, str(home_moshi))

        from moshi_mlx import models
        from moshi_mlx.models.tts import (
            DEFAULT_DSM_TTS_REPO,
            DEFAULT_DSM_TTS_VOICE_REPO,
            TTSModel,
        )
        from moshi_mlx.utils.loaders import hf_get

        if progress_callback:
            progress_callback("Setting random seed...")
        mx.random.seed(299792458)

        if progress_callback:
            progress_callback("Downloading config...")
        raw_config_path = hf_get("config.json", hf_repo)

        with open(raw_config_path, "r") as f:
            raw_config = json.load(f)

        if progress_callback:
            progress_callback("Downloading Mimi codec...")
        mimi_weights = hf_get(raw_config["mimi_name"], hf_repo)
        mimi_weights = hf_get(mimi_weights)

        if progress_callback:
            progress_callback("Downloading LM model...")
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, hf_repo)
        moshi_weights = hf_get(moshi_weights)

        if progress_callback:
            progress_callback("Downloading tokenizer...")
        tokenizer_path = hf_get(raw_config["tokenizer_name"], hf_repo)
        tokenizer_path = hf_get(tokenizer_path)

        if progress_callback:
            progress_callback("Building LM model...")
        lm_config = models.LmConfig.from_config_dict(raw_config)
        self.model = models.Lm(lm_config)
        self.model.set_dtype(mx.bfloat16)

        if progress_callback:
            progress_callback("Loading LM weights...")
        self.model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

        if quantize is not None:
            if progress_callback:
                progress_callback(f"Quantizing to {quantize} bits...")
            nn.quantize(self.model.depformer, bits=quantize)
            for layer in self.model.transformer.layers:
                nn.quantize(layer.self_attn, bits=quantize)
                nn.quantize(layer.gating, bits=quantize)

        if progress_callback:
            progress_callback("Loading tokenizer...")
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        if progress_callback:
            progress_callback("Loading Mimi codec...")
        generated_codebooks = lm_config.generated_codebooks
        self.mimi = models.mimi.Mimi(models.mimi.mimi_202407(generated_codebooks))
        self.mimi.load_pytorch_weights(str(mimi_weights), strict=True)

        if progress_callback:
            progress_callback("Creating TTS model...")
        cfg_condition = None
        self.tts_model = TTSModel(
            self.model,
            self.mimi,
            self.text_tokenizer,
            voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
            n_q=24,
            temp=0.6,
            cfg_coef=2.0,
            max_padding=8,
            initial_padding=2,
            final_padding=4,
            padding_bonus=0.0,
            raw_config=raw_config,
        )

        if self.tts_model.valid_cfg_conditionings:
            cfg_condition = self.tts_model.cfg_coef
            self.tts_model.cfg_coef = 1.0

        # Load voices
        if progress_callback:
            progress_callback("Loading voices...")

        voice_suffix = self.tts_model.voice_suffix

        if voice_folder.startswith("hf-snapshot://"):
            voice_folder_str = voice_folder.removeprefix("hf-snapshot://")
            if voice_folder_str.count("/") > 1:
                parts = voice_folder_str.split("/", 2)
                repo = "/".join(parts[0:2])
                pattern = parts[2] if len(parts) > 2 else None
            else:
                repo = voice_folder_str
                pattern = None
            voice_folder = huggingface_hub.snapshot_download(
                repo, allow_patterns=pattern
            )

        voice_path = Path(voice_folder)

        if self.tts_model.multi_speaker:
            for file in voice_path.glob(f"**/*{voice_suffix}"):
                relative = file.relative_to(voice_path)
                name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
                try:
                    voices = [file, file]
                    attributes = self.tts_model.make_condition_attributes(
                        voices, cfg_coef=cfg_condition
                    )
                    self.all_attributes[name] = attributes
                except Exception as e:
                    print(f"Warning: failed to load voice {name}: {e}")

        if not self.all_attributes:
            if progress_callback:
                progress_callback("Warning: No voices found, using unconditioned generation")
        else:
            if default_voice not in self.all_attributes:
                self.default_voice = next(iter(self.all_attributes.keys()))
            else:
                self.default_voice = default_voice

        if progress_callback:
            progress_callback("Ready!")

        self.loaded = True

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        progress_callback=None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize text to audio.

        Returns:
            tuple of (audio_array, sample_rate)
        """
        import mlx.core as mx

        if not self.loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        voice = voice or self.default_voice

        # Reset caches
        for c in self.model.transformer_cache:
            c.reset()
        for c in self.model.depformer_cache:
            c.reset()
        self.mimi.reset_all()

        if progress_callback:
            progress_callback("Preparing text...")

        # Prepare entries from text
        entries = self.tts_model.prepare_script([text], padding_between=1)

        # Get attributes for voice
        if voice and voice in self.all_attributes:
            attributes = [self.all_attributes[voice]]
        else:
            from moshi_mlx.modules.conditioner import ConditionAttributes

            attributes = [ConditionAttributes(text={"control": "ok", "cfg": None}, tensor={})]

        if progress_callback:
            progress_callback("Generating audio...")

        # Generate
        result = self.tts_model.generate(
            [entries],
            attributes,
            prefixes=None,
            cfg_is_no_prefix=True,
            cfg_is_no_text=True,
        )

        if progress_callback:
            progress_callback("Decoding audio...")

        # Decode frames to audio
        wav_frames = []
        frames_to_decode = result.frames[self.tts_model.delay_steps :]

        for frame in frames_to_decode:
            pcm = self.mimi.decode_step(frame)
            wav_frames.append(pcm)

        # Remove first 2 frames to avoid initial artifacts
        if len(wav_frames) > 2:
            wav_frames = wav_frames[2:]

        if not wav_frames:
            raise RuntimeError("No audio generated")

        wavs = mx.concat(wav_frames, axis=-1)

        # Trim to actual content
        end_step = result.end_steps[0]
        if end_step is not None:
            wav_length = int(
                self.mimi.sample_rate
                * (end_step + self.tts_model.final_padding)
                / self.mimi.frame_rate
            )
            wav = wavs[0, :, :wav_length]
        else:
            wav = wavs[0]

        # Clip and convert to numpy
        wav = mx.clip(wav, -1, 1)
        wav_np = np.array(wav).flatten().astype(np.float32)

        if progress_callback:
            progress_callback("Done!")

        return wav_np, self.mimi.sample_rate


# =============================================================================
# GUI Application
# =============================================================================
class TTSApp:
    """Main TTS Application with tkinter GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Moshi TTS - Apple Silicon")
        self.root.geometry("700x550")

        self.engine = TTSEngine()
        self.current_audio = None
        self.current_sample_rate = None

        self._setup_ui()
        self._start_model_loading()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Status bar at top
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(
            status_frame, text="Loading models...", font=("Helvetica", 11)
        )
        self.status_label.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=200)
        self.progress.pack(side=tk.RIGHT)
        self.progress.start()

        # Voice selection
        voice_frame = ttk.LabelFrame(main_frame, text="Voice", padding="5")
        voice_frame.pack(fill=tk.X, pady=(0, 10))

        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(
            voice_frame, textvariable=self.voice_var, state="disabled", width=50
        )
        self.voice_combo.pack(fill=tk.X)

        # Text input
        text_frame = ttk.LabelFrame(main_frame, text="Text to Speak", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.text_input = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, height=10, font=("Helvetica", 12)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.insert(
            tk.END,
            "Hello! This is the Moshi text to speech system running on Apple Silicon with MLX.",
        )

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        self.speak_btn = ttk.Button(
            button_frame, text="Speak", command=self._on_speak, state="disabled"
        )
        self.speak_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.save_btn = ttk.Button(
            button_frame, text="Save Audio...", command=self._on_save, state="disabled"
        )
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.play_btn = ttk.Button(
            button_frame, text="Play", command=self._on_play, state="disabled"
        )
        self.play_btn.pack(side=tk.LEFT)

        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(10, 0))

        # Quantization
        quant_frame = ttk.Frame(settings_frame)
        quant_frame.pack(fill=tk.X)

        ttk.Label(quant_frame, text="Quantization:").pack(side=tk.LEFT)
        self.quant_var = tk.StringVar(value="4-bit")
        quant_options = ["None (bfloat16)", "4-bit", "8-bit"]
        self.quant_combo = ttk.Combobox(
            quant_frame, textvariable=self.quant_var, values=quant_options, width=15
        )
        self.quant_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.quant_combo.configure(state="disabled")

    def _start_model_loading(self):
        """Start loading models in background thread."""

        def load():
            try:
                self.engine.load_models(
                    quantize=4,
                    progress_callback=self._update_status,
                )
                self.root.after(0, self._on_models_loaded)
            except Exception as e:
                self.engine.load_error = str(e)
                self.root.after(0, lambda: self._on_load_error(str(e)))

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _update_status(self, message: str):
        """Update status label from any thread."""
        self.root.after(0, lambda: self.status_label.configure(text=message))

    def _on_models_loaded(self):
        """Called when models finish loading."""
        self.progress.stop()
        self.progress.pack_forget()

        # Update voice combo
        voices = list(self.engine.all_attributes.keys())
        if voices:
            self.voice_combo["values"] = voices
            self.voice_combo.set(self.engine.default_voice or voices[0])
            self.voice_combo.configure(state="readonly")

        # Enable buttons
        self.speak_btn.configure(state="normal")
        self.status_label.configure(text="Ready")

    def _on_load_error(self, error: str):
        """Called when model loading fails."""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.configure(text=f"Error: {error}")
        messagebox.showerror("Load Error", f"Failed to load models:\n\n{error}")

    def _on_speak(self):
        """Handle Speak button click."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to speak.")
            return

        voice = self.voice_var.get() if self.voice_var.get() else None

        self.speak_btn.configure(state="disabled")
        self.status_label.configure(text="Generating...")
        self.progress.pack(side=tk.RIGHT)
        self.progress.start()

        def generate():
            try:
                audio, sr = self.engine.synthesize(
                    text, voice=voice, progress_callback=self._update_status
                )
                self.current_audio = audio
                self.current_sample_rate = sr
                self.root.after(0, self._on_synthesis_complete)
            except Exception as e:
                self.root.after(0, lambda: self._on_synthesis_error(str(e)))

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def _on_synthesis_complete(self):
        """Called when synthesis completes."""
        self.progress.stop()
        self.progress.pack_forget()
        self.speak_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self.play_btn.configure(state="normal")
        self.status_label.configure(text="Audio ready!")

        # Auto-play
        self._on_play()

    def _on_synthesis_error(self, error: str):
        """Called when synthesis fails."""
        self.progress.stop()
        self.progress.pack_forget()
        self.speak_btn.configure(state="normal")
        self.status_label.configure(text="Error")
        messagebox.showerror("Synthesis Error", f"Failed to generate audio:\n\n{error}")

    def _on_save(self):
        """Handle Save button click."""
        if self.current_audio is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Audio",
        )

        if file_path:
            try:
                import sphn

                sphn.write_wav(
                    file_path, self.current_audio, self.current_sample_rate
                )
                self.status_label.configure(text=f"Saved: {Path(file_path).name}")
            except ImportError:
                # Fallback to scipy if sphn not available
                try:
                    from scipy.io import wavfile

                    wavfile.write(
                        file_path,
                        self.current_sample_rate,
                        (self.current_audio * 32767).astype(np.int16),
                    )
                    self.status_label.configure(text=f"Saved: {Path(file_path).name}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save audio:\n\n{e}")

    def _on_play(self):
        """Handle Play button click."""
        if self.current_audio is None:
            return

        def play_audio():
            try:
                # Save to temp file and play with afplay (macOS)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                try:
                    import sphn

                    sphn.write_wav(temp_path, self.current_audio, self.current_sample_rate)
                except ImportError:
                    from scipy.io import wavfile

                    wavfile.write(
                        temp_path,
                        self.current_sample_rate,
                        (self.current_audio * 32767).astype(np.int16),
                    )

                # Use afplay on macOS
                subprocess.run(["afplay", temp_path], check=True)
                os.unlink(temp_path)
            except Exception as e:
                self.root.after(
                    0, lambda: messagebox.showerror("Play Error", f"Failed to play audio:\n\n{e}")
                )

        thread = threading.Thread(target=play_audio, daemon=True)
        thread.start()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = TTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
