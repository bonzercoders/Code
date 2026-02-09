import os
import logging
import torch
import numpy as np
from typing import Optional, AsyncGenerator

from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.database_director import db

logger = logging.getLogger(__name__)


def revert_delay_pattern(data: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Undo Higgs delay pattern so decoded frames line up."""
    if data.ndim != 2:
        raise ValueError('Expected 2D tensor from audio tokenizer')
    if data.shape[1] - data.shape[0] < start_idx:
        raise ValueError('Invalid start_idx for delay pattern reversion')

    out = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out.append(data[i:(i + 1), i + start_idx:(data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out, dim=0)


class TTS:
    """Audio synthesis engine using Higgs Audio.

    Provides generate_audio_for_sentence() for PCM audio generation.
    Task management (process_sentences) lives in ConversationPipeline.
    """

    def __init__(self):
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.sample_rate = 2400
        self._chunk_size = 14
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_dir = "/workspace/tts/Code/backend/voices"

    async def initialize(self):
        self.engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=self._device,
        )
        logger.info("Higgs Audio TTS initialized")

    # ──────────────────────────────────────────────
    #  Audio generation
    # ──────────────────────────────────────────────

    def load_voice_reference(self, voice: str):
        """Load reference audio and text for voice cloning."""
        audio_path = os.path.join(self.voice_dir, f"{voice}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice}.txt")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path)),
        ]

        return messages

    async def generate_audio_for_sentence(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming. Yields PCM16 chunks."""
        messages = self.load_voice_reference(voice)
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop_strings=['<|end_of_text|>', '<|eot_id|>'],
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                force_audio_gen=True,
            ):
                if delta.audio_tokens is None:
                    continue

                # Check for end token (1025)
                if torch.all(delta.audio_tokens == 1025):
                    break

                # Accumulate tokens
                audio_tokens.append(delta.audio_tokens[:, None])

                # Count non-padding tokens (1024 is padding)
                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1

                # Decode when chunk size reached
                if seq_len > 0 and seq_len % self._chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    try:
                        vq_code = (
                            revert_delay_pattern(audio_tensor, start_idx=seq_len - self._chunk_size + 1)
                            .clip(0, 1023)
                            .to(self._device)
                        )
                        waveform = self.engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                        if isinstance(waveform, torch.Tensor):
                            waveform_np = waveform.detach().cpu().numpy()
                        else:
                            waveform_np = np.asarray(waveform, dtype=np.float32)

                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()

                    except Exception as e:
                        logger.warning(f"Error decoding chunk: {e}")
                        continue

        # Flush remaining tokens
        if seq_len > 0 and seq_len % self._chunk_size != 0 and audio_tokens:
            audio_tensor = torch.cat(audio_tokens, dim=-1)
            remaining = seq_len % self._chunk_size

            try:
                vq_code = (
                    revert_delay_pattern(audio_tensor, start_idx=seq_len - remaining + 1)
                    .clip(0, 1023)
                    .to(self._device)
                )
                waveform = self.engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.detach().cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform, dtype=np.float32)

                pcm = np.clip(waveform_np, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

            except Exception as e:
                logger.warning(f"Error flushing remaining audio: {e}")

    # ──────────────────────────────────────────────
    #  Voice listing
    # ──────────────────────────────────────────────

    async def get_available_voices(self):
        """Get list of available voices from database."""
        voices = await db.get_all_voices()

        available = []
        for voice in voices:
            audio_path = voice.audio_path or os.path.join(self.voice_dir, f"{voice.voice}.wav")
            text_path = voice.text_path or os.path.join(self.voice_dir, f"{voice.voice}.txt")

            available.append({
                "id": voice.voice,
                "name": voice.voice,
                "method": voice.method,
                "audio_path": audio_path,
                "text_path": text_path,
                "speaker_desc": voice.speaker_desc,
                "scene_prompt": voice.scene_prompt,
            })

        available.sort(key=lambda v: v['name'])
        return available
