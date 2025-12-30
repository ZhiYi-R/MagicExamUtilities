"""
Summarization Worker for text summarization.
"""

import os
import time
import json
import pathlib
from enum import Enum
from typing import Optional
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config


class TextSource(Enum):
    """Source of text for summarization."""
    OCR = 'OCR'
    STT = 'STT'


class SummarizationWorker(BaseWorker):
    """
    Worker for text summarization.
    """

    # OCR system prompt
    _OCR_PROMPT = '''
    你是一个专业的AI助手，专门为OCR识别出的文本进行纠错和总结，你的行为守则如下：
    1. 输出结果为Markdown文本。
    2. 输出结果中应当包含原始文本中的所有关键信息，不要缺少任何重要的信息。
    3. 确保你的输出只包含简体中文，如果原文是别的语言，确保你已经进行了翻译。
    4. 请确保你的公式输出正确，并LaTeX格式嵌入到文本中。
    5. 请确保你的表格输出正确，请使用Markdown格式。
    6. 如果内容是复习课，确保你输出的文本包含复习的所有内容，不要缺少任何重要的信息，如果其讲的不够详细，请补充其详细内容。
    7. 你的输出应当仅包含Markdown源码本身。
    '''

    # STT system prompt
    _STT_PROMPT = '''
    你是一个专业的AI助手，专门为语音转录出的文本进行纠错和总结，你的行为守则如下：
    1. 确保你的输出是Markdown格式的，并只包含原始录音的内容。
    2. 确保你的输出是经过格式化、缩进和换行处理的，并符合Markdown语法。
    3. 确保你的输出是经过润色处理的，并符合原始录音的内容。
    4. 确保你的输出只包含简体中文，如果原文是别的语言，确保你已经进行了翻译。
    5. 确保你输出的文本包含原始录音的所有关键信息，不要缺少任何重要的信息。
    6. 如果录音内容是复习课，确保你输出的文本包含复习的所有内容，不要缺少任何重要的信息，如果其讲的不够详细，请补充其详细内容。
    7. 你的输出除了报告的Markdown文本外，请不要输出任何多余的文本（直接输出Markdown源码，不要包裹任何代码块之类的东西)。
    '''

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

        # Use Summarization-specific config if available, otherwise fall back to legacy config
        api_url = os.environ.get('SUMMARIZATION_API_URL', os.environ.get('OPENAI_LIKE_API_URL'))
        api_key = os.environ.get('SUMMARIZATION_API_KEY', os.environ.get('OPENAI_LIKE_API_KEY'))
        self._client = OpenAI(base_url=api_url, api_key=api_key)
        self._model = os.environ['SUMMARIZATION_MODEL']

        # Set system prompt based on text source
        if text_source == TextSource.OCR:
            self._system_prompt = self._OCR_PROMPT
        elif text_source == TextSource.STT:
            self._system_prompt = self._STT_PROMPT
        else:
            raise ValueError(f'Invalid text source: {text_source}')

        # Register methods after Summarization is initialized
        self._register_methods()

        print(f"[SummarizationWorker] Initialized with text_source={text_source}")
        print(f"[SummarizationWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register Summarization methods."""
        self._methods = {
            'summarize': self._summarize,
        }

    def _summarize(self, text: str) -> str:
        """Summarize the given text."""
        print(f'Summarizing text of length {len(text)}')

        # For Qwen3 models, use \no_think prefix to disable chain-of-thought
        if 'qwen3' in self._model.lower():
            user_prompt = f'\\no_think 请对以下文本进行总结，请使用Markdown格式输出：\n{text}'
        else:
            user_prompt = f'请对以下文本进行总结，请使用Markdown格式输出：\n{text}'

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': self._system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.1,
            top_p=1.0,
        )

        if not response.choices:
            raise RuntimeError(f'No choices returned from API for text')
        if not response.choices[0].message.content:
            raise RuntimeError(f'No content returned from API for text')

        content = response.choices[0].message.content

        if not response.usage:
            print(f'Summarization Done for text, length: {len(content)}')
        else:
            print(f'Summarization Done for text, length: {len(content)}, usage: {response.usage.total_tokens}')

        if self._dump_summarization_response:
            dump_file_path = self._dump_dir.joinpath(f'Summarization_{time.strftime("%Y_%m_%d-%H_%M_%S")}.json')
            with open(dump_file_path, 'w') as f:
                f.write(response.model_dump_json(indent=4, ensure_ascii=False))
            print(f'Dumped summarization response to {dump_file_path}')

        return content

    def _estimate_tokens(self, task) -> int:
        """Estimate tokens for summarization requests."""
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
