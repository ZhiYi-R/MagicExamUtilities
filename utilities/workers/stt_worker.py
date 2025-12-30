"""
STT (Speech-to-Text) Worker for processing audio files.
"""

import os
import json
import pathlib
from typing import Optional
import requests

from ..worker import BaseWorker, get_rate_limit_config


class STTWorker(BaseWorker):
    """
    Worker for Speech-to-Text processing.
    """

    def __init__(self,
                 dump_dir: pathlib.Path = pathlib.Path('.'),
                 dump_stt_response: bool = True,
                 task_timeout: Optional[float] = None):
        """
        Initialize the STT worker.

        Args:
            dump_dir: Directory to dump STT responses
            dump_stt_response: Whether to dump STT responses to JSON
            task_timeout: Timeout for individual STT tasks in seconds (None = no timeout)
        """
        rpm, tpm = get_rate_limit_config('ASR')
        super().__init__(name='STTWorker', rpm=rpm, tpm=tpm, task_timeout=task_timeout)

        self._dump_dir = dump_dir
        self._dump_stt_response = dump_stt_response
        self._api_url = os.environ['ASR_API_URL']
        self._api_key = os.environ['ASR_API_KEY']
        self._model = os.environ['ASR_MODEL']

        # Register methods after STT is initialized
        self._register_methods()

        print(f"[STTWorker] Initialized")
        print(f"[STTWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register STT methods."""
        self._methods = {
            'stt': self._stt,
        }

    def get_model_name(self) -> str:
        """Get the model name used by this worker."""
        return self._model

    def _stt(self, audio_path: pathlib.Path) -> str:
        """Process an audio file with STT."""
        if not audio_path.exists():
            raise FileNotFoundError(f'Audio {audio_path} does not exist')

        print(f'STTing audio: {audio_path}')

        with open(audio_path, 'rb') as f:
            response = requests.post(
                url=self._api_url,
                headers={
                    'Authorization': f'Bearer {self._api_key}',
                },
                files={
                    'file': (audio_path.name, f.read(), 'audio/mpeg'),
                    'model': (None, self._model)
                }
            )

        if response.status_code != 200:
            raise RuntimeError(f'STT failed for audio {audio_path}, status code: {response.status_code}, response: {response.text}')

        if self._dump_stt_response:
            dump_file_path = self._dump_dir.joinpath(f'{audio_path.stem}.json')
            with open(dump_file_path, 'w') as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=4)
            print(f'Dumped STT response to {dump_file_path}')

        if not response.json()['text']:
            raise RuntimeError(f'No text returned from ASR API for audio {audio_path}')

        return response.json()['text']

    def _estimate_tokens(self, task) -> int:
        """Estimate tokens for STT requests."""
        return 500  # Conservative estimate: 500 tokens per audio

    def process_audio(self, audio_path: pathlib.Path, timeout: Optional[float] = None) -> str:
        """
        Process a single audio file with STT.

        Args:
            audio_path: Path to the audio file
            timeout: Maximum time to wait for result (seconds) - enforced at worker level

        Returns:
            Transcribed text
        """
        future = self.submit('stt', audio_path, _task_timeout=timeout)
        return future.get()

    def process_audios(self, audio_paths: list[pathlib.Path], timeout_per_audio: Optional[float] = None) -> list[str]:
        """
        Process multiple audio files with STT.

        Args:
            audio_paths: List of paths to audio files
            timeout_per_audio: Maximum time to wait per audio file (seconds) - enforced at worker level

        Returns:
            List of transcribed texts
        """
        results = []
        for audio_path in audio_paths:
            result = self.process_audio(audio_path, timeout=timeout_per_audio)
            results.append(result)
        return results
