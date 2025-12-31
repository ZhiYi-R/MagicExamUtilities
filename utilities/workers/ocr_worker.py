"""
OCR Worker for processing images with OCR models.
Supports both DeepSeek OCR and generic vision-language models.
"""

import os
import base64
import hashlib
import pathlib
import re
import time
import logging
from typing import Optional, Tuple
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config, get_retry_config
from ..models import OCRResult, compute_file_hash, estimate_tokens

logger = logging.getLogger(__name__)


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


def _detect_and_fix_duplicates(content: str) -> Tuple[str, bool]:
    """
    Detect and fix duplicate content in OCR output.

    Common duplicate patterns:
    1. Consecutive repeated paragraphs
    2. Entire content repeated at the end
    3. Repeated lines or sections

    Args:
        content: The OCR output content

    Returns:
        Tuple of (cleaned_content, had_duplicates)
    """
    had_duplicates = False
    original = content

    # Split into paragraphs
    paragraphs = content.split('\n\n')
    cleaned_paragraphs = []
    i = 0

    while i < len(paragraphs):
        para = paragraphs[i].strip()
        if not para:
            i += 1
            continue

        # Check if current paragraph is a duplicate of the next one
        if i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if para == next_para:
                had_duplicates = True
                print(f"[OCR] Found duplicate paragraph, skipping: {para[:50]}...")
                i += 2  # Skip both duplicates
                continue

        # Check for multiple consecutive repeated paragraphs (3+ times)
        repeat_count = 1
        if i + 1 < len(paragraphs):
            next_para = paragraphs[i + 1].strip()
            if para == next_para:
                # Count how many times it repeats
                j = i + 1
                while j < len(paragraphs) and paragraphs[j].strip() == para:
                    repeat_count += 1
                    j += 1

                if repeat_count >= 3:
                    had_duplicates = True
                    print(f"[OCR] Found {repeat_count}x repeated paragraph, keeping one")
                    i = j  # Skip all but first
                    cleaned_paragraphs.append(para)
                    continue

        cleaned_paragraphs.append(para)
        i += 1

    content = '\n\n'.join(cleaned_paragraphs)

    # Check for entire content being repeated at the end
    lines = content.split('\n')
    # Try different suffix lengths to find repeats
    for suffix_len in [min(20, len(lines) // 3), min(10, len(lines) // 4), min(5, len(lines) // 5)]:
        if suffix_len < 2:
            continue
        if len(lines) < suffix_len * 2:
            continue

        first_part = '\n'.join(lines[:len(lines) - suffix_len])
        suffix = '\n'.join(lines[-suffix_len:])

        # Check if the suffix appears somewhere in the first part
        if suffix in first_part:
            had_duplicates = True
            print(f"[OCR] Found {suffix_len}-line suffix repeated in content, removing suffix")
            content = first_part
            break

    return content, had_duplicates


def _validate_ocr_output(content: str) -> Tuple[bool, str]:
    """
    Validate OCR output for common issues.

    Args:
        content: The OCR output content

    Returns:
        Tuple of (is_valid, issue_description)
    """
    if not content or content.strip() == '':
        return False, "Empty output"

    if len(content.strip()) < 10:
        return False, f"Output too short ({len(content.strip())} chars)"

    # Check for excessive repetition (same character repeated > 50 times)
    for char in ['\n', ' ', '.', '-']:
        if char * 50 in content:
            return False, f"Excessive repetition of '{char}'"

    # Check for very low content diversity (less than 10 unique characters)
    unique_chars = set(content)
    if len(unique_chars) < 10:
        return False, f"Low character diversity ({len(unique_chars)} unique chars)"

    return True, ""


class OCRWorker(BaseWorker):
    """
    Worker for OCR processing.

    Supports both DeepSeek OCR and generic vision-language models
    based on the OCR_USE_DEEPSEEK_OCR environment variable.
    """

    def __init__(self,
                 dump_dir: pathlib.Path = pathlib.Path('.'),
                 dump_ocr_response: bool = True,
                 max_retries: Optional[int] = None,
                 task_timeout: Optional[float] = None):
        """
        Initialize the OCR worker.

        Args:
            dump_dir: Directory to dump OCR responses
            dump_ocr_response: Whether to dump OCR responses to JSON
            max_retries: Maximum number of retries for failed OCR attempts (None = use env var)
            task_timeout: Timeout for individual OCR tasks in seconds (None = no timeout)
        """
        rpm, tpm = get_rate_limit_config('OCR')
        retry_max, retry_delay = get_retry_config('OCR')

        # Use parameter if provided, otherwise use env var, default to 2 for OCR
        if max_retries is None:
            max_retries = retry_max if retry_max > 0 else 2

        super().__init__(name='OCRWorker', rpm=rpm, tpm=tpm, pricing_prefix='OCR',
                        max_retries=max_retries, retry_delay=retry_delay, task_timeout=task_timeout)

        self._dump_dir = dump_dir
        self._dump_ocr_response = dump_ocr_response
        self._max_retries = max_retries

        # Use OCR-specific config if available, otherwise fall back to legacy config
        api_url = os.environ.get('OCR_API_URL', os.environ.get('OPENAI_LIKE_API_URL'))
        api_key = os.environ.get('OCR_API_KEY', os.environ.get('OPENAI_LIKE_API_KEY'))
        self._client = OpenAI(base_url=api_url, api_key=api_key)
        self._model = os.environ['OCR_MODEL']

        # Select OCR type based on environment variable
        self._use_deepseek = os.environ.get('OCR_USE_DEEPSEEK_OCR', '1') == '1'

        # Set prompt based on OCR type
        # DeepSeek OCR is extremely sensitive to prompt - DO NOT change
        self._deepseek_prompt = '<image>\nOCR this image with Markdown format.'

        # Generic VL prompt for more flexible OCR
        self._generic_prompt = r'''你是一个专业的 OCR 文本识别助手，负责将图像中的内容准确地转换为 Markdown 格式文本。

## 核心原则

**准确性优先**：只输出图像中实际存在的文本内容，绝不编造任何不存在的内容。

## 输出要求

1. **严格保持原文**：不添加、不删除、不修改任何信息
2. **结构清晰**：保持原文的层次结构和排版逻辑
3. **格式规范**：使用标准 Markdown 语法

## 特殊内容处理

| 内容类型 | 输出格式 | 重要限制 |
|---------|---------|---------|
| **标题** | 使用 `#` 表示层级 | - |
| **表格** | 使用 Markdown 表格格式 `| 列1 \| 列2 \|` | - |
| **公式** | 行内公式用 `$公式$`，独立公式用 `$$公式$$` | - |
| **代码** | 使用 ```语言 代码块 ``` 标注语言类型 | - |
| **列表** | 无序列表用 `-` 或 `*`，有序列表用 `1.` | - |
| **强调** | 加粗用 `**文本**`，斜体用 `*文本*` | - |
| **图片/图表/流程图** | 用文字描述图片内容 | **禁止使用 `![描述](url)` 格式，禁止编造任何 URL** |
| **电路图** | 用文字描述电路结构 | **禁止输出任何假的 URL 或路径** |

## 图片处理规则（非常重要）

当图像中包含图片、图表、流程图等时：

✅ **正确做法**：
- 用文字描述：`[图片：显示某系统架构的示意图]`
- 简化说明：`[流程图：展示了从A到B的处理流程]`
- 或用 Mermaid 描述结构（如果能准确理解）

❌ **禁止做法**：
- 禁止输出：`![图片描述](https://...)` 或 `![图片](./images/...)`
- 禁止编造任何 URL、文件路径或图片链接
- 禁止输出 `![图片](image.png)` 这种占位符

## 注意事项

- 遇到模糊不清的内容，用 `[unclear:...]` 标注
- 保持段落分隔，使用空行分段
- 如果是扫描件的页面，请保持页面内容的完整性和顺序
- 仅输出 Markdown 内容，不要包含任何解释性文字
- **绝对不要编造图片 URL 或文件路径**'''

        # Register methods after initialization
        self._register_methods()

        ocr_type = 'deepseek' if self._use_deepseek else 'generic'
        print(f"[OCRWorker] Initialized with {ocr_type} OCR")
        print(f"[OCRWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register OCR methods."""
        self._methods = {
            'ocr': self._ocr,
            'ocr_structured': self._ocr_structured,
        }

    def get_model_name(self) -> str:
        """Get the model name used by this worker."""
        return self._model

    def _ocr(self, image_path: pathlib.Path, _timeout: Optional[float] = None) -> str:
        """Process an image with OCR (legacy, returns plain text)."""
        result = self._ocr_structured(image_path, _timeout=_timeout)
        return result.raw_text

    def _ocr_structured(self, image_path: pathlib.Path, _timeout: Optional[float] = None) -> OCRResult:
        """Process an image with OCR and return structured result with retry logic."""
        if not image_path.exists():
            raise FileNotFoundError(f'Image {image_path} does not exist')

        print(f'OCRing image: {image_path}')

        # Compute file hash for caching
        file_hash = compute_file_hash(image_path)

        # Extract page number from filename (format: pdfname_0.jpg)
        try:
            page_number = int(image_path.stem.split('_')[-1])
        except (ValueError, IndexError):
            page_number = 0

        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Perform OCR (retry is handled by Worker level)
        if self._use_deepseek:
            response = self._deepseek_ocr(image_base64, image_path, _timeout=_timeout)
        else:
            response = self._generic_ocr(image_base64, image_path, _timeout=_timeout)

        if not response.choices:
            raise RuntimeError(f'No choices returned from API for image {image_path}')
        if not response.choices[0].message.content:
            raise RuntimeError(f'No content returned from API for image {image_path}')

        content = response.choices[0].message.content
        # Strip outer markdown code block wrapper (Qwen models add this)
        content = _strip_outer_markdown_block(content)

        # Validate output (raise exception for retry if validation fails)
        is_valid, issue = _validate_ocr_output(content)
        if not is_valid:
            # Create a retryable error for Worker level retry
            raise RuntimeError(f'OCR output validation failed: {issue}')

        # Detect and fix duplicates
        content, had_duplicates = _detect_and_fix_duplicates(content)
        if had_duplicates:
            print(f'[OCR] Duplicates detected and removed in output')

        token_count = response.usage.total_tokens if response.usage else estimate_tokens(len(content))

        # Track usage and cost
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._track_usage(input_tokens, output_tokens)
            cost_str = self._cost_tracker.format_cost(cost) if cost > 0 else "N/A"
            print(f'OCR Done for image: {image_path}, length: {len(content)}, '
                  f'usage: {input_tokens} in + {output_tokens} out = {token_count} total, cost: {cost_str}')
        else:
            logger.warning(f'OCR response for {image_path} did not include usage information, cost tracking disabled for this request')
            print(f'OCR Done for image: {image_path}, length: {len(content)}')

        # Create structured result
        result = OCRResult(
            file_path=str(image_path),
            file_hash=file_hash,
            page_number=page_number,
            page_hash=file_hash,  # Use image file hash as page hash for page-level caching
            timestamp=os.path.getmtime(image_path),
            raw_text=content,
            sections=[],
            char_count=len(content),
            token_estimate=token_count
        )

        if self._dump_ocr_response:
            # Save dump file in the same directory as the image (PDF cache folder)
            dump_file_path = image_path.parent.joinpath(f'{image_path.stem}.json')
            with open(dump_file_path, 'w') as f:
                f.write(response.model_dump_json(indent=4, ensure_ascii=False))
            print(f'Dumped OCR response to {dump_file_path}')

        return result

    def _deepseek_ocr(self, image_base64: str, image_path: pathlib.Path, _timeout: Optional[float] = None):
        """DeepSeek OCR implementation."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{'role': 'user', 'content': [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{image_base64}',
                        'detail': 'high'
                    }
                },
                {
                    'type': 'text',
                    'text': self._deepseek_prompt
                }
            ]}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            extra_body={
                'repetition_penalty': 1.0,
            },
            timeout=_timeout
        )

    def _generic_ocr(self, image_base64: str, image_path: pathlib.Path, _timeout: Optional[float] = None):
        """Generic Vision-Language OCR implementation."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=[
                {'role': 'system', 'content': self._generic_prompt},
                {'role': 'user', 'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{image_base64}',
                            'detail': 'high'
                        }
                    },
                    {
                        'type': 'text',
                        'text': '请识别这张图像中的所有内容，按原文顺序输出 Markdown 格式。'
                    }
                ]}
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            extra_body={
                'repetition_penalty': 1.0,
            },
            timeout=_timeout
        )

    def _estimate_tokens(self, task) -> int:
        """Estimate tokens for OCR requests."""
        # Use actual token count from result if available
        return 1000  # Conservative estimate: 1000 tokens per image

    def process_image(self, image_path: pathlib.Path, timeout: Optional[float] = None) -> str:
        """
        Process a single image with OCR (legacy, returns plain text).

        Args:
            image_path: Path to the image file
            timeout: Maximum time to wait for result (seconds) - enforced at worker level

        Returns:
            OCR result as text
        """
        future = self.submit('ocr', image_path, _task_timeout=timeout)
        return future.get(timeout=timeout)

    def process_image_structured(self, image_path: pathlib.Path, timeout: Optional[float] = None) -> OCRResult:
        """
        Process a single image with OCR and return structured result.

        Args:
            image_path: Path to the image file
            timeout: Maximum time to wait for result (seconds) - enforced at worker level

        Returns:
            Structured OCR result
        """
        future = self.submit('ocr_structured', image_path, _task_timeout=timeout)
        return future.get(timeout=timeout)

    def process_images(self, image_paths: list[pathlib.Path], timeout_per_image: Optional[float] = None) -> list[str]:
        """
        Process multiple images with OCR.

        Args:
            image_paths: List of paths to image files
            timeout_per_image: Maximum time to wait per image (seconds) - enforced at worker level

        Returns:
            List of OCR results
        """
        results = []
        for image_path in image_paths:
            result = self.process_image(image_path, timeout=timeout_per_image)
            results.append(result)
        return results

    def process_images_structured(
        self,
        image_paths: list[pathlib.Path],
        timeout_per_image: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> list[OCRResult]:
        """
        Process multiple images with OCR and return structured results.

        Args:
            image_paths: List of paths to image files
            timeout_per_image: Maximum time to wait per image (seconds) - enforced at worker level
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            List of structured OCR results
        """
        results = []
        for i, image_path in enumerate(image_paths):
            result = self.process_image_structured(image_path, timeout=timeout_per_image)
            results.append(result)
            # Report progress after each page
            if progress_callback:
                progress_callback(i + 1, len(image_paths), f"OCR 页 {i+1}/{len(image_paths)}")
        return results
