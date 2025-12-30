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

        # Configure mock responses
        for mock_client in [mock_ocr.return_value, mock_sum.return_value]:
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = 100
            mock_response.model_dump_json.return_value = '{"test": "data"}'
            mock_client.chat.completions.create.return_value = mock_response

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
