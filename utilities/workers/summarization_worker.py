"""
Summarization Worker for text summarization.

Supports both direct summarization and retrieval-augmented summarization
for handling documents that exceed context window limits.
"""

import os
import time
import json
import pathlib
import re
from enum import Enum
from typing import Optional, List
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config, get_retry_config


def _strip_outer_markdown_block(content: str) -> str:
    """
    Strip the outer markdown code block wrapper if present.

    Some models (like Qwen) wrap the entire output in:
    ```markdown
    ...content...
    ```

    This function detects and removes that wrapper while preserving
    any internal code blocks.

    Args:
        content: The raw content from the model

    Returns:
        Content with outer markdown block removed
    """
    content = content.strip()
    lines = content.split('\n')

    # Check if content starts with ```markdown or ``` and ends with ```
    if len(lines) >= 3 and lines[0].startswith('```'):
        # First line is the opening marker (e.g., ```markdown or ```)
        # Last line should be closing ```
        if lines[-1].strip() == '```':
            # Extract inner content (between first and last lines)
            inner_lines = lines[1:-1]
            return '\n'.join(inner_lines)

    return content


def _clean_inline_math(content: str) -> str:
    """
    Clean up inline math formulas to ensure proper rendering.

    Some markdown renderers have issues with spaces around $ symbols.
    This function normalizes inline math formulas.

    Examples:
    - "$ foo $" -> "$foo$"
    - "$ x + y $" -> "$x + y$"
    - "$$" (display math) is left unchanged

    Args:
        content: The markdown content

    Returns:
        Content with normalized inline math formulas
    """
    # Fix inline math with spaces: $ foo $ -> $foo$
    # But preserve display math: $$...$$
    result = []

    i = 0
    while i < len(content):
        # Check for display math $$...$$
        if i + 1 < len(content) and content[i:i+2] == '$$':
            # Find the closing $$
            j = content.find('$$', i + 2)
            if j != -1:
                result.append(content[i:j+2])
                i = j + 2
                continue

        # Check for inline math $...$
        if content[i] == '$':
            # Find the closing $
            j = content.find('$', i + 1)
            if j != -1:
                # Extract the formula and trim spaces
                formula = content[i+1:j].strip()
                result.append(f'${formula}$')
                i = j + 1
                continue

        result.append(content[i])
        i += 1

    return ''.join(result)


class TextSource(Enum):
    """Source of text for summarization."""
    OCR = 'OCR'
    STT = 'STT'


class SummarizationWorker(BaseWorker):
    """
    Worker for text summarization.
    """

    # OCR system prompt - for summarizing PDF/OCR text
    _OCR_PROMPT = r'''你是一个专业的学习助手，负责将 OCR 识别的课件内容整理成结构化的复习笔记。

## 任务目标

将原始的 OCR 文本转换为易于理解的 Markdown 格式复习笔记。

## 输出要求

### 结构化组织
- 使用标题层级组织内容（`#` 主标题、`##` 章节标题、`###` 小节）
- 相关内容归类到同一章节
- 保持逻辑顺序，便于学习和复习

### 内容完整性
- 保留所有关键概念、定义和公式
- 重要的图表和表格需保留或描述
- 如果原文内容过于简略，需要适当补充说明

### 格式规范
- **公式**：使用 LaTeX 格式，行内用 `$公式$`，独立公式用 `$$公式$$`
- **表格**：使用 Markdown 表格格式
- **代码**：使用 ```语言 ... ``` 代码块
- **重点**：用 `**加粗**` 或 `*斜体*` 标注重点内容
- **列表**：使用 `-` 无序列表或 `1.` 有序列表

### 语言要求
- 全部使用简体中文
- 如果原文包含英文，保留英文术语并在括号中补充中文解释
- 专业术语首次出现时加粗标注

### 输出格式
- 仅输出 Markdown 内容，不包含任何解释性文字
- 不使用代码块包裹整个输出
- 确保可以直接保存为 .md 文件使用'''

    # STT system prompt - for summarizing speech-to-text transcripts
    _STT_PROMPT = r'''你是一个专业的学习助手，负责将语音转录的复习课内容整理成结构化的复习笔记。

## 任务目标

将口语化的语音转录文本转换为易于理解的 Markdown 格式复习笔记。

## 输出要求

### 文本处理
- **去除口语冗余**：删除"嗯"、"啊"、"那个"等语气词
- **修正语法错误**：整理破碎的句子，使其通顺
- **提炼关键信息**：去除重复和无关的内容

### 结构化组织
- 使用标题层级组织内容（`#` 课程主题、`##` 知识点）
- 按主题分类，相关内容归类到同一章节
- 保持授课逻辑顺序

### 内容完整性
- 保留所有重要概念、定义和公式
- 老师强调的重点需特别标注
- 如果讲解不够详细，适当补充解释

### 格式规范
- **公式**：使用 LaTeX 格式，行内用 `$公式$`，独立公式用 `$$公式$$`
- **表格**：使用 Markdown 表格格式
- **代码**：使用 ```语言 ... ``` 代码块
- **重点**：用 `**加粗**` 标注老师强调的内容
- **列表**：使用 `-` 无序列表或 `1.` 有序列表

### 语言要求
- 全部使用简体中文
- 保留专业术语，必要时在括号中补充解释
- 使用学术化、书面化的表达方式

### 输出格式
- 仅输出 Markdown 内容，不包含任何解释性文字
- 不使用代码块包裹整个输出
- 确保可以直接保存为 .md 文件使用'''

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
        max_retries, retry_delay = get_retry_config('SUMMARIZATION')

        super().__init__(name='SummarizationWorker', rpm=rpm, tpm=tpm, pricing_prefix='SUMMARIZATION',
                        max_retries=max_retries, retry_delay=retry_delay)

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
            'summarize_long': self._summarize_long_text,
        }

    def _summarize(self, text: str) -> str:
        """Summarize the given text."""
        print(f'Summarizing text of length {len(text)}')

        # Build user prompt based on text source
        if self._text_source == TextSource.OCR:
            user_prompt = f'''请将以下 OCR 识别的课件内容整理成结构化的复习笔记：

---
{text}
---

请按照上述要求整理，确保输出格式规范、内容完整、便于学习复习。'''
        else:  # STT
            user_prompt = f'''请将以下语音转录的复习课内容整理成结构化的复习笔记：

---
{text}
---

请按照上述要求整理，去除口语冗余，修正语法错误，提炼关键信息，输出格式规范、内容完整的复习笔记。'''

        # For Qwen3 models, use \no_think prefix to disable chain-of-thought
        if 'qwen3' in self._model.lower():
            user_prompt = f'\\no_think {user_prompt}'

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
        # Strip outer markdown code block wrapper (Qwen models add this)
        content = _strip_outer_markdown_block(content)
        # Clean inline math formulas to fix rendering issues
        content = _clean_inline_math(content)

        # Track usage and cost
        if not response.usage:
            print(f'Summarization Done for text, length: {len(content)}')
        else:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._track_usage(input_tokens, output_tokens)
            cost_str = self._cost_tracker.format_cost(cost) if cost > 0 else "N/A"
            print(f'Summarization Done for text, length: {len(content)}, '
                  f'usage: {input_tokens} in + {output_tokens} out = {response.usage.total_tokens} total, cost: {cost_str}')

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

    def _split_text_into_chunks(self, text: str, max_chars: int = 8000) -> List[str]:
        """
        Split text into chunks for processing.

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If single paragraph is too long, split by sentences
            if len(paragraph) > max_chars:
                sentences = paragraph.split('。')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sentence += '。'
                    if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += ' ' + sentence if current_chunk else sentence
            else:
                if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += '\n\n' + paragraph if current_chunk else paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _summarize_long_text(
        self,
        text: str,
        max_chars_per_chunk: int = 8000,
        progress_callback=None
    ) -> str:
        """
        Summarize long text by splitting into chunks.

        Args:
            text: Long text to summarize
            max_chars_per_chunk: Maximum characters per chunk
            progress_callback: Optional callback for progress updates

        Returns:
            Combined summary
        """
        # Split into chunks
        chunks = self._split_text_into_chunks(text, max_chars_per_chunk)
        print(f"[SummarizationWorker] Split text into {len(chunks)} chunks")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, len(chunks), f"Summarizing chunk {i+1}/{len(chunks)}")

            print(f"[SummarizationWorker] Summarizing chunk {i+1}/{len(chunks)} (length: {len(chunk)})")
            summary = self._summarize(chunk)
            chunk_summaries.append(summary)

        # Combine summaries
        combined_text = '\n\n'.join(chunk_summaries)
        print(f"[SummarizationWorker] Combined summaries length: {len(combined_text)}")

        # If combined text is still too long, summarize again
        if len(combined_text) > max_chars_per_chunk:
            if progress_callback:
                progress_callback(len(chunks), len(chunks), "Creating final summary...")
            print("[SummarizationWorker] Combined summary still long, creating final summary...")
            return self._summarize(combined_text)

        return combined_text

    def summarize(
        self,
        text: str,
        timeout: Optional[float] = None,
        use_chunking: bool = False,
        max_chars: int = 8000,
        progress_callback=None
    ) -> str:
        """
        Summarize the given text.

        Args:
            text: Text to summarize
            timeout: Maximum time to wait for result (seconds)
            use_chunking: Whether to use chunking for long texts
            max_chars: Maximum characters before chunking is triggered (if use_chunking=True)
            progress_callback: Optional callback(chunk_index, total_chunks, message)

        Returns:
            Summarized text in Markdown format
        """
        # Decide whether to use chunking
        if use_chunking and len(text) > max_chars:
            # For long text, use chunking
            future = self.submit('summarize_long', text, max_chars, progress_callback)
            return future.get(timeout=timeout)
        else:
            # For short text, use direct summarization
            future = self.submit('summarize', text)
            return future.get(timeout=timeout)
