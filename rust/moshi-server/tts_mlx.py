# Copyright (c) Kyutai, all rights reserved.
# MLX-based TTS service for Apple Silicon - drop-in replacement for PyTorch tts.py

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece

# Import MLX moshi modules
import sys
sys.path.insert(0, str(Path.home() / "github/moshi/moshi_mlx"))
from moshi_mlx import models
from moshi_mlx.models.generate import LmGen
from moshi_mlx.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    Entry,
    State,
    StateMachine,
    TokenIds,
    TTSModel,
)
from moshi_mlx.modules.conditioner import ConditionAttributes, TensorCondition
from moshi_mlx.utils.loaders import hf_get
from moshi_mlx.utils.sampling import Sampler
from pydantic import BaseModel


class MaskFlags(Enum):
    HAS_PCM = 1
    IS_EOS = 2
    WORD_FINISHED = 4
    AR_STEP = 8
    MISSING_WORDS = 16


class Config(BaseModel):
    log_folder: Path = Path.home() / 'tmp/tts-service'
    hf_repo: str = DEFAULT_DSM_TTS_REPO
    mimi_weight: Path | None = None
    moshi_weight: Path | None = None
    config_path: Path | None = None
    tokenizer: Path | None = None

    n_q: int = 24
    voice_folder: str = str(Path.home() / 'models/tts-voices')
    default_voice: str = "barack_demo.wav"

    temp: float = 0.6
    cfg_coef: float = 2.

    max_padding: int = 8
    initial_padding: int = 2
    final_padding: int = 4
    padding_between: int = 1
    padding_bonus: float = 0.

    quantize: int | None = None
    debug: bool = False


@dataclass
class ClientState:
    is_complete: bool = False
    state: State | None = None
    offset: int = 0
    lm_gen: LmGen | None = None
    cross_attention_src: mx.array | None = None
    ct: mx.array | None = None

    def reset(self, state_machine: StateMachine) -> None:
        self.is_complete = False
        self.offset = 0
        self.state = state_machine.new_state([])
        self.lm_gen = None
        self.cross_attention_src = None
        self.ct = None

    def is_active(self, lookahead: int) -> bool:
        state = self.state
        if state is None:
            return False
        if self.is_complete:
            return True
        if not state.entries:
            return False
        if lookahead == 0:
            return True
        if not state.entries[0].tokens:
            return True
        remaining = lookahead + 1
        for entry in state.entries:
            if entry.tokens:
                remaining -= 1
            if remaining <= 0:
                return True
        return False


@dataclass
class TTSService:
    batch_size: int
    default_attribute_name: str
    all_attributes: dict[str, ConditionAttributes]

    tts_model: TTSModel
    lm: models.Lm
    mimi: models.mimi.Mimi

    cfg_is_no_text: bool = True
    cfg_condition: float | None = None
    padding_between: int = 1
    padding_bonus: float = 0.0
    n_q: int = 32
    debug: bool = False
    final_padding: int = 4

    flags_out: np.ndarray | None = None
    clients: list[ClientState] = field(default_factory=list)
    cross_attention_cache: dict[str, mx.array] = field(default_factory=dict)

    def __post_init__(self):
        machine = self.tts_model.machine

        for _ in range(self.batch_size):
            client = ClientState()
            self.clients.append(client)

        if self.tts_model.multi_speaker:
            print("Filling cross attention cache.")
            for name, attributes in self.all_attributes.items():
                self.cross_attention_cache[name] = self._get_cross_attention_source(attributes)

        # Warmup
        print("warming up.")
        for _ in range(3):
            pcm = mx.zeros((1, 1, 1920))
            mx.eval(self.mimi.encode(pcm))
            codes = mx.zeros((1, self.n_q, 1), dtype=mx.int32)
            mx.eval(self.mimi.decode(codes))
        self.mimi.reset_all()
        print("ready to roll.")

    def _get_cross_attention_source(self, attr: ConditionAttributes) -> mx.array:
        """Get cross attention source from voice attributes."""
        assert self.lm.condition_provider is not None
        for _key, _value in attr.tensor.items():
            _conditioner = self.lm.condition_provider.conditioners[_key]
            return _conditioner.condition(_value)
        return mx.zeros((1, 1, self.lm.cfg.transformer.d_model))

    def _create_lm_gen(self, client: ClientState) -> LmGen:
        """Create a new LmGen for a client."""
        def on_text_hook(text_tokens):
            if client.state is None:
                return
            tokens = text_tokens.tolist()
            out_tokens = []
            for b, token in enumerate(tokens):
                out_token, consumed_new_word = self.tts_model.machine.process(
                    client.offset, client.state, token[0]
                )
                if self.flags_out is not None and consumed_new_word:
                    # Find client index
                    for idx, c in enumerate(self.clients):
                        if c is client:
                            self.flags_out[idx] |= MaskFlags.WORD_FINISHED.value
                            break
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)[:, None]

        return LmGen(
            self.lm,
            max_steps=30000,
            text_sampler=Sampler(temp=self.tts_model.temp),
            audio_sampler=Sampler(temp=self.tts_model.temp),
            batch_size=1,
            cfg_coef=self.tts_model.cfg_coef,
            on_text_hook=on_text_hook,
        )

    def _print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def step(
        self,
        updates: list[tuple[int, list[int], np.ndarray | str | None]],
        pcm_out: np.ndarray,
        flags_out: np.ndarray,
        code_out: np.ndarray,
    ) -> None:
        machine = self.tts_model.machine
        delay_steps = self.tts_model.delay_steps
        lookahead = machine.second_stream_ahead
        pad = machine.token_ids.pad

        self.flags_out = flags_out
        flags_out[:] = 0

        # Process updates
        for b, new_entry, voice in updates:
            client = self.clients[b]
            if new_entry[0] == -1:
                # Reset client
                client.reset(machine)
                self.mimi.reset_all()
                for c in self.lm.transformer_cache:
                    c.reset()
                for c in self.lm.depformer_cache:
                    c.reset()
                client.lm_gen = self._create_lm_gen(client)

                # Set up voice conditioning
                if self.tts_model.multi_speaker:
                    if isinstance(voice, np.ndarray):
                        # Dynamic voice from numpy array
                        voice_tensor = mx.array(voice)
                        attr = make_condition_attributes_mlx([voice_tensor], cfg_condition=self.cfg_condition)
                        client.cross_attention_src = self._get_cross_attention_source(attr)
                    else:
                        # Cached voice
                        voice_name = voice or self.default_attribute_name
                        client.cross_attention_src = self.cross_attention_cache.get(
                            voice_name, self.cross_attention_cache.get(self.default_attribute_name)
                        )

                    # Set up condition tensor
                    attr = self.all_attributes.get(voice or '', self.all_attributes.get(self.default_attribute_name))
                    if attr:
                        ct_tensor = None
                        for _key, _value in attr.text.items():
                            _ct = self.lm.condition_provider.condition_tensor(_key, _value)
                            tensor = _ct.tensor.squeeze(0)
                            ct_tensor = tensor if ct_tensor is None else ct_tensor + tensor
                        if ct_tensor is not None:
                            from moshi_mlx.modules.conditioner import ConditionTensor
                            client.ct = ConditionTensor(ct_tensor[None])

                new_entry = new_entry[1:]
                self._print(f"[{b}] Reset, voice is {voice}.")

            if client.state is None:
                self._print(f"[{b}] Trying to push {new_entry}, but not assigned.")
            elif not new_entry:
                pass
            elif new_entry == [-2]:
                self._print(f"[{b}] Done.")
                client.is_complete = True
            elif new_entry[0] == pad:
                self._print(f"[{b}] Pushing pause {new_entry}.")
                padding = len(new_entry)
                client.state.entries.append(Entry([], '', padding=padding))
            else:
                self._print(f"[{b}] Pushing {new_entry}.")
                padding = 0
                if self.padding_between > 0:
                    padding = max(0, self.padding_between + len(new_entry) - 1)
                client.state.entries.append(Entry(new_entry, '', padding=padding))

        # Run inference for each active client
        for b, client in enumerate(self.clients):
            active = client.is_active(lookahead)
            if not active:
                if client.state is not None:
                    flags_out[b] |= MaskFlags.MISSING_WORDS.value
                continue

            if client.lm_gen is None:
                continue

            flags_out[b] |= MaskFlags.AR_STEP.value

            # Run one step of generation
            missing = self.lm.n_q - self.lm.dep_q
            input_tokens = mx.full((1, missing), machine.token_ids.zero, dtype=mx.int64)

            client.lm_gen.step(
                input_tokens,
                ct=client.ct,
                cross_attention_src=client.cross_attention_src,
            )

            # Get audio tokens if available
            audio_tokens = client.lm_gen.last_audio_tokens()
            if audio_tokens is not None and not (audio_tokens == machine.token_ids.zero).all():
                # Decode to PCM
                frame = audio_tokens[:, :, None]  # [1, nq, 1]
                pcm = self.mimi.decode_step(frame)
                mx.eval(pcm)

                # Copy to output
                pcm_np = np.array(pcm[0, 0])
                if pcm_np.shape[0] == 1920:
                    pcm_out[b, :] = pcm_np
                    flags_out[b] |= MaskFlags.HAS_PCM.value

                # Store tokens
                code_out[b, :audio_tokens.shape[1]] = np.array(audio_tokens[0])
                code_out[b, audio_tokens.shape[1]:] = 0

            client.offset += 1
            self._print(f"[{b}] Offset {client.offset: 3d}, pendings={len(client.state.entries): 3d}.")

            # Check if generation is complete
            if client.is_complete and client.state.end_step is not None:
                real_end = client.state.end_step + delay_steps + self.final_padding
                if client.offset >= real_end:
                    self._print(f"[{b}] Done.")
                    client.reset(machine)
                    flags_out[b] |= MaskFlags.IS_EOS.value

        self.flags_out = None


def make_condition_attributes_mlx(
    voices: list[mx.array | Path],
    max_speakers: int = 5,
    cfg_condition: float | None = None
) -> ConditionAttributes:
    """Create condition attributes from voice embeddings."""
    voice_tensor = None
    mask = None
    for idx in range(max_speakers):
        if idx < len(voices):
            voice = voices[idx]
            if isinstance(voice, Path):
                emb = mx.load(str(voice))['speaker_wavs']
            else:
                emb = voice
            assert emb.ndim == 3
            if voice_tensor is None:
                voice_tensor = mx.zeros((1, max_speakers, emb.shape[2], emb.shape[1]))
            if mask is None:
                mask = mx.zeros((1, max_speakers, emb.shape[2]), dtype=mx.uint8)
            voice_tensor = voice_tensor.at[:, idx, :, :].set(emb.swapaxes(1, 2))
            mask = mask.at[:, idx, :].set(1)

    if voice_tensor is None:
        return ConditionAttributes(text={'control': 'ok', 'cfg': None}, tensor={})

    voice_tensor = voice_tensor.reshape(1, -1, voice_tensor.shape[-1])
    mask = mask.reshape(1, -1)
    tensors = {'speaker_wavs': TensorCondition(voice_tensor, mask)}

    text: dict[str, str | None] = {'control': 'ok'}
    if cfg_condition is None:
        text['cfg'] = None
    else:
        text['cfg'] = format(cfg_condition, '.1f')

    return ConditionAttributes(text=text, tensor=tensors)


def init(batch_size: int, config_override: dict) -> TTSService:
    """Initialize the MLX TTS service."""
    config = Config(**config_override)
    config.log_folder.mkdir(parents=True, exist_ok=True)

    mx.random.seed(299792458)

    print("retrieving checkpoint")

    # Load config
    raw_config_path = config.config_path
    if raw_config_path is None:
        raw_config_path = hf_get("config.json", config.hf_repo)

    print(f"loading config from {raw_config_path}")
    with open(hf_get(raw_config_path), "r") as fobj:
        raw_config = json.load(fobj)

    # Load weights
    mimi_weights = config.mimi_weight
    if mimi_weights is None:
        mimi_weights = hf_get(raw_config["mimi_name"], config.hf_repo)
    mimi_weights = hf_get(mimi_weights)

    moshi_weights = config.moshi_weight
    if moshi_weights is None:
        moshi_name = raw_config.get("moshi_name", "model.safetensors")
        moshi_weights = hf_get(moshi_name, config.hf_repo)
    moshi_weights = hf_get(moshi_weights)

    tokenizer_path = config.tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_get(raw_config["tokenizer_name"], config.hf_repo)
    tokenizer_path = hf_get(tokenizer_path)

    # Build LM model
    lm_config = models.LmConfig.from_config_dict(raw_config)
    lm = models.Lm(lm_config)
    lm.set_dtype(mx.bfloat16)

    print(f"loading model weights from {moshi_weights}")
    lm.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    # Quantize if requested
    if config.quantize is not None:
        print(f"quantizing model to {config.quantize} bits")
        nn.quantize(lm.depformer, bits=config.quantize)
        for layer in lm.transformer.layers:
            nn.quantize(layer.self_attn, bits=config.quantize)
            nn.quantize(layer.gating, bits=config.quantize)

    # Load tokenizer
    print(f"loading the text tokenizer from {tokenizer_path}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    # Load Mimi
    print(f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    mimi = models.mimi.Mimi(models.mimi.mimi_202407(generated_codebooks))
    mimi.load_pytorch_weights(str(mimi_weights), strict=True)

    # Create TTS model
    cfg_condition = None
    tts_model = TTSModel(
        lm,
        mimi,
        text_tokenizer,
        voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
        n_q=config.n_q,
        temp=config.temp,
        cfg_coef=config.cfg_coef,
        max_padding=config.max_padding,
        initial_padding=config.initial_padding,
        final_padding=config.final_padding,
        padding_bonus=config.padding_bonus,
        raw_config=raw_config,
    )

    if tts_model.valid_cfg_conditionings:
        cfg_condition = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0
        cfg_is_no_text = False
    else:
        cfg_is_no_text = True

    # Load voices
    voice_suffix = tts_model.voice_suffix
    print(f"loading voices from {config.voice_folder}, with suffix {voice_suffix}.")
    all_attributes = {}
    voice_folder = config.voice_folder

    if voice_folder.startswith("hf-snapshot://"):
        voice_folder = voice_folder.removeprefix("hf-snapshot://")
        if voice_folder.count("/") > 1:
            voice_folder, pattern = voice_folder.split("/", 2)[0:2], voice_folder.split("/", 2)[2]
            voice_folder = "/".join(voice_folder)
        else:
            pattern = None
        print(f"retrieving voices from {voice_folder}")
        voice_folder = huggingface_hub.snapshot_download(voice_folder, allow_patterns=pattern)

    voice_folder = Path(voice_folder)

    if tts_model.multi_speaker:
        for file in voice_folder.glob(f'**/*{voice_suffix}'):
            relative = file.relative_to(voice_folder)
            name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
            try:
                voices = [file, file]
                attributes = tts_model.make_condition_attributes(voices, cfg_coef=cfg_condition)
            except Exception:
                print(f"[WARNING] failed to load voice {name}")
            else:
                all_attributes[name] = attributes

        if not all_attributes:
            raise RuntimeError(
                f"No voices found. Searched for files matching {voice_folder}/**/*{voice_suffix}"
            )

        if config.default_voice not in all_attributes:
            print(f"[WARNING] Default voice {config.default_voice} not found, using first available")
            config.default_voice = next(iter(all_attributes.keys()))

    service = TTSService(
        batch_size=batch_size,
        default_attribute_name=config.default_voice,
        all_attributes=all_attributes,
        tts_model=tts_model,
        lm=lm,
        mimi=mimi,
        cfg_condition=cfg_condition,
        cfg_is_no_text=cfg_is_no_text,
        padding_between=config.padding_between,
        padding_bonus=config.padding_bonus,
        debug=config.debug,
        final_padding=config.final_padding,
        n_q=config.n_q,
    )

    return service


if __name__ == '__main__':
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-c', '--cfg_coef', default=2., type=float)
    parser.add_argument('-r', '--hf-repo', default=DEFAULT_DSM_TTS_REPO)
    parser.add_argument('-q', '--quantize', type=int, help='Quantize to N bits')
    args = parser.parse_args()

    bs = args.batch_size
    config_override = {
        'hf_repo': args.hf_repo,
        'cfg_coef': args.cfg_coef,
        'voice_folder': 'hf-snapshot://kyutai/tts-voices/unmute-prod-website/*.safetensors',
        'default_voice': 'unmute-prod-website/default_voice.wav',
        'quantize': args.quantize,
    }
    service = init(batch_size=bs, config_override=config_override)
    print("Service initialized")

    pcm_out = np.zeros((bs, 1920), dtype=np.float32)
    flags_out = np.zeros(bs, dtype=np.int32)
    code_out = np.zeros((bs, 33), dtype=np.int32)

    # Test run
    rng = random.Random(1234)
    service.step([(0, [-1, 32, 21], '')], pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)

    for i in range(100):
        inp = []
        if rng.random() < 0.1:
            word = [13, 34]
            inp.append((0, word, None))
        be = time.time()
        service.step(inp, pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)
        el = time.time() - be
        print(f"FR {el * 1000:.1f}ms flags={flags_out[0]}")
