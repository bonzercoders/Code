import asyncio
import logging
from typing import Dict, Optional
from threading import Thread

from backend.RealtimeSTT import AudioToTextRecorder
from backend.services.pipeline_types import Callback

logger = logging.getLogger(__name__)

class STT:
    """Realtime transcription of user's audio prompt"""

    def __init__(self,
        on_transcription_update: Optional[Callback] = None,
        on_transcription_stabilized: Optional[Callback] = None,
        on_transcription_finished: Optional[Callback] = None,
        on_vad_detect_start: Optional[Callback] = None,
        on_vad_detect_stop: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,
    ):
        # Store callbacks with consistent key names
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_transcription_update': on_transcription_update,
            'on_transcription_stabilized': on_transcription_stabilized,
            'on_transcription_finished': on_transcription_finished,
            'on_vad_detect_start': on_vad_detect_start,
            'on_vad_detect_stop': on_vad_detect_stop,
            'on_vad_start': on_vad_start,
            'on_vad_stop': on_vad_stop,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
        }

        self.is_listening = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[Thread] = None

        self.recorder = AudioToTextRecorder(
            model="small.en",
            language="en",
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            realtime_model_type="small.en",
            on_realtime_transcription_update=self._on_transcription_update,
            on_realtime_transcription_stabilized=self._on_transcription_stabilized,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_vad_detect_start=self._on_vad_detect_start,
            on_vad_detect_stop=self._on_vad_detect_stop,
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.5,
            spinner=False,
            level=logging.WARNING,
            use_microphone=False
        )

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for callback execution"""
        self.loop = loop

    def transcriber(self):
        """Transcribes in real-time from browser audio feed"""

        while self.is_listening:
            try:
                user_message = self.recorder.text()

                if user_message and user_message.strip():
                    callback = self.callbacks.get('on_transcription_finished')
                    if callback:
                        self.run_callback(callback, user_message)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")

    def run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run a user callback from a RealtimeSTT background thread."""

        if callback is None or self.loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)

        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    def start_listening(self):
        """Start listening for audio input"""
        if self.is_listening:
            return

        self.is_listening = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self.transcriber, daemon=True)
            self._thread.start()
        logger.info("Started listening for audio")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False
        logger.info("Stopped listening for audio")

    def _on_transcription_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self.run_callback(self.callbacks.get('on_transcription_update'), text)

    def _on_transcription_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_stabilized'), text)

    def _on_transcription_finished(self, user_message: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_finished'), user_message)

    def _on_vad_detect_start(self) -> None:
        """RealtimeSTT callback: started listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_start'))

    def _on_vad_detect_stop(self) -> None:
        """RealtimeSTT callback: stopped listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_stop'))

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        self.run_callback(self.callbacks.get('on_vad_start'))

    def _on_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self.run_callback(self.callbacks.get('on_vad_stop'))

    def _on_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self.run_callback(self.callbacks.get('on_recording_start'))

    def _on_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self.run_callback(self.callbacks.get('on_recording_stop'))
