"""
STT (Speech-to-Text) Worker for processing audio files.
"""

import pathlib
from typing import Optional

from ..worker import BaseWorker, get_rate_limit_config
from ..STT import SpeechToText


class STTWorker(BaseWorker):
    """
    Worker for Speech-to-Text processing.
    """

    def __init__(self,
                 dump_dir: pathlib.Path = pathlib.Path('.'),
                 dump_stt_response: bool = True):
        """
        Initialize the STT worker.

        Args:
            dump_dir: Directory to dump STT responses
            dump_stt_response: Whether to dump STT responses to JSON
        """
        rpm, tpm = get_rate_limit_config('ASR')
        super().__init__(name='STTWorker', rpm=rpm, tpm=tpm)

        self._dump_dir = dump_dir
        self._dump_stt_response = dump_stt_response
        self._stt = SpeechToText(
            dump_dir=dump_dir,
            dump_stt_response=dump_stt_response
        )

        print(f"[STTWorker] Initialized")
        print(f"[STTWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register STT methods."""
        self._methods = {
            'stt': self._stt.stt,
        }

    def _estimate_tokens(self, task) -> int:
        """
        Estimate tokens for STT requests.
        STT typically returns a lot of text, so we estimate conservatively.
        """
        # Estimate based on typical audio length
        # A 1-minute audio might produce ~100-200 tokens of text
        return 500

    def process_audio(self, audio_path: pathlib.Path, timeout: Optional[float] = None) -> str:
        """
        Process a single audio file with STT.

        Args:
            audio_path: Path to the audio file
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Transcribed text
        """
        future = self.submit('stt', audio_path)
        return future.get(timeout=timeout)

    def process_audios(self, audio_paths: list[pathlib.Path], timeout_per_audio: Optional[float] = None) -> list[str]:
        """
        Process multiple audio files with STT.

        Args:
            audio_paths: List of paths to audio files
            timeout_per_audio: Maximum time to wait per audio file (seconds)

        Returns:
            List of transcribed texts
        """
        results = []
        for audio_path in audio_paths:
            result = self.process_audio(audio_path, timeout=timeout_per_audio)
            results.append(result)
        return results
