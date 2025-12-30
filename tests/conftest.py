"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for testing."""
    env_vars = {
        'OCR_API_URL': 'https://api.test.com/v1',
        'OCR_API_KEY': 'test-ocr-key',
        'OCR_MODEL': 'test-ocr-model',
        'OCR_USE_DEEPSEEK_OCR': '1',
        'OCR_RPM': '60',
        'OCR_TPM': '100000',
        'ASR_API_URL': 'https://asr.test.com/v1',
        'ASR_API_KEY': 'test-asr-key',
        'ASR_MODEL': 'test-asr-model',
        'ASR_RPM': '60',
        'ASR_TPM': '100000',
        'SUMMARIZATION_API_URL': 'https://summarization.test.com/v1',
        'SUMMARIZATION_API_KEY': 'test-summarization-key',
        'SUMMARIZATION_MODEL': 'test-summarization-model',
        'SUMMARIZATION_RPM': '60',
        'SUMMARIZATION_TPM': '100000',
        'OPENAI_LIKE_API_URL': 'https://fallback.test.com/v1',
        'OPENAI_LIKE_API_KEY': 'test-fallback-key',
    }

    original_env = os.environ.copy()
    os.environ.update(env_vars)

    yield env_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('utilities.workers.ocr_worker.OpenAI') as mock_ocr, \
         patch('utilities.workers.summarization_worker.OpenAI') as mock_sum:

        # OCR mock response - needs to be long enough to pass validation
        ocr_response = Mock()
        ocr_choice = Mock()
        ocr_message = Mock()
        # Use a longer response with diverse characters to pass OCR validation
        ocr_message.content = "# Test Document\n\nThis is a sample text with multiple words and diverse characters including symbols like !@#$% and numbers 12345."
        ocr_choice.message = ocr_message
        ocr_response.choices = [ocr_choice]
        ocr_response.usage = Mock()
        ocr_response.usage.total_tokens = 200
        ocr_response.usage.prompt_tokens = 100
        ocr_response.usage.completion_tokens = 100
        ocr_response.model_dump_json.return_value = '{"test": "ocr"}'
        mock_ocr.return_value.chat.completions.create.return_value = ocr_response

        # Summarization mock response
        sum_response = Mock()
        sum_choice = Mock()
        sum_message = Mock()
        sum_message.content = "Test response"
        sum_choice.message = sum_message
        sum_response.choices = [sum_choice]
        sum_response.usage = Mock()
        sum_response.usage.total_tokens = 100
        sum_response.usage.prompt_tokens = 50
        sum_response.usage.completion_tokens = 50
        sum_response.model_dump_json.return_value = '{"test": "summarization"}'
        mock_sum.return_value.chat.completions.create.return_value = sum_response

        yield {
            'ocr': mock_ocr,
            'summarization': mock_sum,
        }


@pytest.fixture
def mock_requests():
    """Mock requests module for STT testing."""
    with patch('utilities.workers.stt_worker.requests') as mock:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Test transcription'}
        mock.post.return_value = mock_response
        yield mock
