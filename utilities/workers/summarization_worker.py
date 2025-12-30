"""
Summarization Worker for text summarization.
"""

import pathlib
from typing import Optional

from ..worker import BaseWorker, get_rate_limit_config
from ..Summarization import Summarization, TextSource


class SummarizationWorker(BaseWorker):
    """
    Worker for text summarization.
    """

    def __init__(self,
                 text_source: TextSource,
                 dump_dir: pathlib.Path = pathlib.Path('.'),
                 dump_summarization_response: bool = True):
        """
        Initialize the Summarization worker.

        Args:
            text_source: Source of the text (OCR or STT)
            dump_dir: Directory to dump summarization responses
            dump_summarization_response: Whether to dump summarization responses to JSON
        """
        rpm, tpm = get_rate_limit_config('SUMMARIZATION')
        super().__init__(name='SummarizationWorker', rpm=rpm, tpm=tpm)

        self._dump_dir = dump_dir
        self._dump_summarization_response = dump_summarization_response
        self._text_source = text_source
        self._summarization = Summarization(
            text_source=text_source,
            dump_dir=dump_dir,
            dump_summarization_response=dump_summarization_response
        )

        print(f"[SummarizationWorker] Initialized with text_source={text_source}")
        print(f"[SummarizationWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register Summarization methods."""
        self._methods = {
            'summarize': self._summarization.summarize,
        }

    def _estimate_tokens(self, task) -> int:
        """
        Estimate tokens for summarization requests.
        Summarization typically consumes input tokens + output tokens.
        """
        # Get the text length from the task args
        if task.args and len(task.args) > 0:
            text = task.args[0]
            # Estimate: text length + output tokens (conservative estimate)
            return len(text) + 2000
        return 5000  # Conservative default

    def summarize(self, text: str, timeout: Optional[float] = None) -> str:
        """
        Summarize the given text.

        Args:
            text: Text to summarize
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Summarized text in Markdown format
        """
        future = self.submit('summarize', text)
        return future.get(timeout=timeout)
