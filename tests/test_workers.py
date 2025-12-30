"""
Unit tests for workers module.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from utilities.workers import OCRWorker, STTWorker, SummarizationWorker, TextSource


@pytest.mark.unit
class TestOCRWorker:
    """Test cases for OCRWorker class."""

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_initialization_deepseek(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test OCRWorker initialization with DeepSeek OCR."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)

        assert worker.name == "OCRWorker"
        assert worker._use_deepseek is True
        assert worker._dump_dir == temp_dir

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_initialization_generic(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test OCRWorker initialization with Generic VL."""
        mock_rate_config.return_value = (60, 100000)
        # Temporarily set OCR_USE_DEEPSEEK_OCR to 0
        original = os.environ.get('OCR_USE_DEEPSEEK_OCR')
        os.environ['OCR_USE_DEEPSEEK_OCR'] = '0'

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)

        # Restore original value
        if original is not None:
            os.environ['OCR_USE_DEEPSEEK_OCR'] = original
        else:
            del os.environ['OCR_USE_DEEPSEEK_OCR']

        assert worker.name == "OCRWorker"
        assert worker._use_deepseek is False

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
        # OCR mock returns longer content to pass validation
        assert result == "# Test Document\n\nThis is a sample text with multiple words and diverse characters including symbols like !@#$% and numbers 12345."

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
        # OCR mock returns longer content to pass validation
        expected = "# Test Document\n\nThis is a sample text with multiple words and diverse characters including symbols like !@#$% and numbers 12345."
        assert all(r == expected for r in results)

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_process_image_structured(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test processing a single image with structured result."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        # Create a temporary image file with page number in filename
        test_image = temp_dir / "test_pdf_0.jpg"
        test_image.write_bytes(b"fake image data")

        result = worker.process_image_structured(test_image, timeout=5.0)

        # Check structured result
        assert result.raw_text == "# Test Document\n\nThis is a sample text with multiple words and diverse characters including symbols like !@#$% and numbers 12345."
        assert result.file_path == str(test_image)
        assert result.page_number == 0
        assert result.char_count > 0
        assert result.token_estimate > 0

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_process_images_structured(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test processing multiple images with structured results."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        # Create temporary image files
        images = []
        for i in range(2):
            img = temp_dir / f"test_pdf_{i}.jpg"
            img.write_bytes(b"fake image data")
            images.append(img)

        results = worker.process_images_structured(images, timeout_per_image=5.0)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.page_number == i
            assert result.char_count > 0

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_stores_dumped_response(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that OCR responses are dumped to JSON."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        # Create a temporary image file
        test_image = temp_dir / "test_image.jpg"
        test_image.write_bytes(b"fake image data")

        worker.process_image(test_image, timeout=5.0)

        # Check that dump file was created
        dump_files = list(temp_dir.glob("test_image.json"))
        assert len(dump_files) >= 1

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.ocr_worker.get_rate_limit_config')
    def test_ocr_worker_nonexistent_file(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test handling of non-existent image file."""
        mock_rate_config.return_value = (60, 100000)

        worker = OCRWorker(dump_dir=temp_dir, dump_ocr_response=True)
        worker.start()

        nonexistent = temp_dir / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError):
            worker.process_image(nonexistent, timeout=5.0)

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

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_split_text_into_chunks(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test text splitting into chunks."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        # Create a long text
        long_text = "This is paragraph one.\n\n" + "This is paragraph two.\n\n" * 100

        chunks = worker._split_text_into_chunks(long_text, max_chars=1000)

        assert len(chunks) > 1
        # All chunks should be under max_chars
        for chunk in chunks:
            assert len(chunk) <= 1000 + 100  # Allow some tolerance for paragraph boundaries

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_split_text_preserves_content(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that text splitting preserves all content."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        # Text with known paragraph count
        text = "Para1\n\nPara2\n\nPara3\n\nPara4\n\nPara5"
        chunks = worker._split_text_into_chunks(text, max_chars=20)

        # When combined, should have all content
        combined = "\n\n".join(chunks)
        assert "Para1" in combined
        assert "Para5" in combined

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_with_long_text_chunking(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test summarization with chunking enabled for long text."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        # Create a long text that would trigger chunking
        long_text = "Paragraph content. " * 200  # ~2500 chars

        # With chunking enabled
        result = worker.summarize(long_text, timeout=10.0, use_chunking=True, max_chars=1000)
        assert result == "Test response"  # Mock response

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_stores_dumped_response(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that summarization responses are dumped to JSON."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        test_text = "Test text for dumping."
        result = worker.summarize(test_text, timeout=5.0)

        # Check that dump file was created
        dump_files = list(temp_dir.glob("Summarization_*.json"))
        assert len(dump_files) >= 1

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_generate_title(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test generating a title from content."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        # Content with headings
        content = """# 数据库系统概论

## 第一章 关系数据库

### 1.1 关系模型

关系数据库是基于关系模型的数据库系统。

### 1.2 SQL 语言

SQL 是结构化查询语言。

## 第二章 数据库设计

数据库设计是创建有效数据库结构的过程。
"""
        result = worker.generate_title(content, timeout=5.0)
        assert result == "Test response"  # Mock response

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_generate_title_no_headings(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test generating a title from content without headings."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.STT,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )
        worker.start()

        # Content without headings
        content = """今天我们来讲一下计算机网络的基本概念。计算机网络是指将地理位置不同的计算机，
通过通信线路连接起来，实现资源共享和信息传递的系统。主要分为局域网、城域网和广域网。
"""
        result = worker.generate_title(content, timeout=5.0)
        assert result  # Should return a non-empty string

        worker.shutdown(wait=True, timeout=2.0)

    @patch('utilities.workers.summarization_worker.get_rate_limit_config')
    def test_summarization_worker_generate_title_registers_method(self, mock_rate_config, mock_env_vars, temp_dir, mock_openai_client):
        """Test that generate_title method is registered."""
        mock_rate_config.return_value = (60, 100000)

        worker = SummarizationWorker(
            text_source=TextSource.OCR,
            dump_dir=temp_dir,
            dump_summarization_response=True
        )

        assert 'generate_title' in worker._methods
