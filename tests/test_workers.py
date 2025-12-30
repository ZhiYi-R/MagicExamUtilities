"""
Unit tests for workers module.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from utilities.workers import OCRWorker, STTWorker, SummarizationWorker
from utilities.Summarization import TextSource


@pytest.mark.unit
class TestOCRWorker:
    """Test cases for OCRWorker class."""

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_initialization_deepseek(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test OCRWorker initialization with DeepSeek OCR."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)

        assert worker.name == "OCRWorker"
        assert worker._ocr_type == "deepseek"
        assert worker._dump_dir == temp_dir

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_initialization_generic(self, mock_rate_config, temp_dir, mock_openai_client):
        """Test OCRWorker initialization with Generic VL."""
        mock_rate_config.return_value = (60, 100000)
        os.environ['OCR_USE_DEEPSEEK_OCR'] = '0'

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)

        assert worker.name == "OCRWorker"
        assert worker._ocr_type == "generic"

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_registers_methods(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that OCRWorker registers OCR method."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        assert 'ocr' in worker._methods

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_estimate_tokens(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test token estimation for OCR tasks."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)

        # Create a mock task
        mock_task = Mock()
        mock_task.args = (Path("test.jpg"),)

        tokens = worker._estimate_tokens(mock_task)
        assert tokens == 1000  # OCR estimate

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_process_image(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test processing a single image."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        # Create a temporary image file
        test_image = temp_dir / "test_image.jpg"
        test_image.write_bytes(b"fake image data")

        result = worker.process_image(test_image, timeout=5.0)
        assert result == "Test response"

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_process_images(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test processing multiple images."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        # Create temporary image files
        images = []
        for i in range(3):
            img = temp_dir / f"test_image_{i}.jpg"
            img.write_bytes(b"fake image data")
            images.append(img)

        results = worker.process_images(images, timeout_per_image=5.0)
        assert len(results) == 3
        assert all(r == "Test response" for r in results)

        worker.shutdown(wait=True, timeout=2.0)


@pytest.mark.unit
class TestSTTWorker:
    """Test cases for STTWorker class."""

    @patch('utilities.workers.stt_worker.get_rate_limit_config')
    def test_stt_worker_initialization(self, mock_rate_config, mock_env_vars, temp_dir, mock_requests):
        """Test STTWorker initialization."""
        mock_rate_config.return_value = (60, 100000)

        worker = STTWorker(dump_dir=temp_dir, dump_stt_response=True)

        assert worker.name == "STTWorker"
        assert worker._dump_dir == temp_dir

    @patch('utilities.workers.stt_worker.get_rate_limit_config')
    def test_stt_worker_registers_methods(self, mock_rate_config, mock_env_vars, temp_dir, mock_requests):
        """Test that STTWorker registers STT method."""
        mock_rate_config.return_value = (60, 100000)

        worker = STTWorker(dump_dir=temp_dir, dump_stt_response=True)
        worker.start()

        assert 'stt' in worker._methods

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.stt_worker.get_rate_limit_config')
    def test_stt_worker_estimate_tokens(self, mock_rate_config, mock_env_vars, temp_dir, mock_requests):
        """Test token estimation for STT tasks."""
        mock_rate_config.return_value = (60, 100000)

        worker = STTWorker(dump_dir=temp_dir, dump_stt_response=True)

        # Create a mock task
        mock_task = Mock()
        mock_task.args = (Path("test.mp3"),)

        tokens = worker._estimate_tokens(mock_task)
        assert tokens == 500  # STT estimate

    @patch('utilities.workers.stt_worker.get_rate_limit_config')
    def test_stt_worker_process_audio(self, mock_rate_config, mock_env_vars, temp_dir, mock_requests):
        """Test processing a single audio file."""
        mock_rate_config.return_value = (60, 100000)

        worker = STTWorker(dump_dir=temp_dir, dump_stt_response=True)
        worker.start()

        # Create a temporary audio file
        test_audio = temp_dir / "test_audio.mp3"
        test_audio.write_bytes(b"fake audio data")

        result = worker.process_audio(test_audio, timeout=5.0)
        assert result == "Test transcription"

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.stt_worker.get_rate_limit_config')
    def test_stt_worker_process_audios(self, mock_rate_config, mock_env_vars, temp_dir, mock_requests):
        """Test processing multiple audio files."""
        mock_rate_config.return_value = (60, 100000)

        worker = STTWorker(dump_dir=temp_dir, dump_stt_response=True)
        worker.start()

        # Create temporary audio files
        audios = []
        for i in range(2):
            audio = temp_dir / f"test_audio_{i}.mp3"
            audio.write_bytes(b"fake audio data")
            audios.append(audio)

        results = worker.process_audios(audios, timeout_per_audio=5.0)
        assert len(results) == 2
        assert all(r == "Test transcription" for r in results)

        worker.shutdown(wait=True, timeout=2.0)


@pytest.mark.unit
class TestSummarizationWorker:
    """Test cases for SummarizationWorker class."""

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_initialization_ocr(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test SummarizationWorker initialization for OCR source."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        assert worker.name == "SummarizationWorker"
        assert worker._text_source == TextSource.OCR

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_initialization_stt(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test SummarizationWorker initialization for STT source."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.STT,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        assert worker.name == "SummarizationWorker"
        assert worker._text_source == TextSource.STT

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_registers_methods(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that SummarizationWorker registers summarize method."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        assert 'summarize' in worker._methods

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_estimate_tokens(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test token estimation for summarization tasks."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        # Create a mock task with text argument
        mock_task = Mock()
        mock_task.args = ("This is a test text that is fifty characters long!!",)

        tokens = worker._estimate_tokens(mock_task)
        # Text length (51) + 2000 output estimate
        assert tokens == 2051

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_estimate_tokens_no_args(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test token estimation when no text argument provided."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        # Create a mock task without text argument
        mock_task = Mock()
        mock_task.args = ()

        tokens = worker._estimate_tokens(mock_task)
        # Should use conservative default
        assert tokens == 5000

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_summarize(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test summarizing text."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        test_text = "This is a test text for summarization."
        result = worker.summarize(test_text, timeout=5.0)
        assert result == "Test response"

        worker.shutdown(wait=True, timeout=2.0)
