"""
OCR Worker for processing images with OCR models.
Supports both DeepSeek OCR and generic vision-language models.
"""

import os
import base64
import hashlib
import pathlib
from typing import Optional
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config
from ..models import OCRResult, compute_file_hash, estimate_tokens


class OCRWorker(BaseWorker):
    """
    Worker for OCR processing.

    Supports both DeepSeek OCR and generic vision-language models
    based on the OCR_USE_DEEPSEEK_OCR environment variable.
    """

    def __init__(self,
                 dump_dir: pathlib.Path = pathlib.Path('.'),
                 dump_ocr_response: bool = True):
        """
        Initialize the OCR worker.

        Args:
            dump_dir: Directory to dump OCR responses
            dump_ocr_response: Whether to dump OCR responses to JSON
        """
        rpm, tpm = get_rate_limit_config('OCR')
        super().__init__(name='OCRWorker', rpm=rpm, tpm=tpm)

        self._dump_dir = dump_dir
        self._dump_ocr_response = dump_ocr_response

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

## 输出要求

1. **准确性优先**：严格保持原文内容，不添加、不删除、不修改任何信息
2. **结构清晰**：保持原文的层次结构和排版逻辑
3. **格式规范**：使用标准 Markdown 语法

## 特殊内容处理

| 内容类型 | 输出格式 |
|---------|---------|
| **标题** | 使用 `#` 表示层级，如 `# 一级标题`、`## 二级标题` |
| **表格** | 使用 Markdown 表格格式 `| 列1 | 列2 |` |
| **公式** | 行内公式用 `$公式$`，独立公式用 `$$公式$$` |
| **代码** | 使用 ```语言 代码块 ``` 标注语言类型 |
| **列表** | 无序列表用 `-` 或 `*`，有序列表用 `1.` |
| **强调** | 加粗用 `**文本**`，斜体用 `*文本*` |
| **流程图/架构图** | 简化描述为文本，可用 Mermaid 格式 |
| **图片说明** | 用 `![描述](url)` 格式标注图片位置 |

## 注意事项

- 遇到模糊不清的内容，用 `[ unclear:...]` 标注
- 保持段落分隔，使用空行分段
- 如果是扫描件的页面，请保持页面内容的完整性和顺序
- 仅输出 Markdown 内容，不要包含任何解释性文字'''

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

    def _ocr(self, image_path: pathlib.Path) -> str:
        """Process an image with OCR (legacy, returns plain text)."""
        result = self._ocr_structured(image_path)
        return result.raw_text

    def _ocr_structured(self, image_path: pathlib.Path) -> OCRResult:
        """Process an image with OCR and return structured result."""
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

            if self._use_deepseek:
                response = self._deepseek_ocr(image_base64, image_path)
            else:
                response = self._generic_ocr(image_base64, image_path)

        if not response.choices:
            raise RuntimeError(f'No choices returned from API for image {image_path}')
        if not response.choices[0].message.content:
            raise RuntimeError(f'No content returned from API for image {image_path}')

        content = response.choices[0].message.content
        token_count = response.usage.total_tokens if response.usage else estimate_tokens(len(content))

        if not response.usage:
            print(f'OCR Done for image: {image_path}, length: {len(content)}')
        else:
            print(f'OCR Done for image: {image_path}, length: {len(content)}, usage: {token_count}')

        # Create structured result
        result = OCRResult(
            file_path=str(image_path),
            file_hash=file_hash,
            page_number=page_number,
            timestamp=os.path.getmtime(image_path),
            raw_text=content,
            sections=[],  # Could be enhanced with text parsing
            char_count=len(content),
            token_estimate=token_count
        )

        if self._dump_ocr_response:
            dump_file_path = self._dump_dir.joinpath(f'{image_path.stem}.json')
            with open(dump_file_path, 'w') as f:
                f.write(response.model_dump_json(indent=4, ensure_ascii=False))
            print(f'Dumped OCR response to {dump_file_path}')

        return result

    def _deepseek_ocr(self, image_base64: str, image_path: pathlib.Path):
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
            }
        )

    def _generic_ocr(self, image_base64: str, image_path: pathlib.Path):
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
            }
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
            timeout: Maximum time to wait for result (seconds)

        Returns:
            OCR result as text
        """
        future = self.submit('ocr', image_path)
        return future.get(timeout=timeout)

    def process_image_structured(self, image_path: pathlib.Path, timeout: Optional[float] = None) -> OCRResult:
        """
        Process a single image with OCR and return structured result.

        Args:
            image_path: Path to the image file
            timeout: Maximum time to wait for result (seconds)

        Returns:
            Structured OCR result
        """
        future = self.submit('ocr_structured', image_path)
        return future.get(timeout=timeout)

    def process_images(self, image_paths: list[pathlib.Path], timeout_per_image: Optional[float] = None) -> list[str]:
        """
        Process multiple images with OCR.

        Args:
            image_paths: List of paths to image files
            timeout_per_image: Maximum time to wait per image (seconds)

        Returns:
            List of OCR results
        """
        results = []
        for image_path in image_paths:
            result = self.process_image(image_path, timeout=timeout_per_image)
            results.append(result)
        return results

    def process_images_structured(self, image_paths: list[pathlib.Path], timeout_per_image: Optional[float] = None) -> list[OCRResult]:
        """
        Process multiple images with OCR and return structured results.

        Args:
            image_paths: List of paths to image files
            timeout_per_image: Maximum time to wait per image (seconds)

        Returns:
            List of structured OCR results
        """
        results = []
        for image_path in image_paths:
            result = self.process_image_structured(image_path, timeout=timeout_per_image)
            results.append(result)
        return results
