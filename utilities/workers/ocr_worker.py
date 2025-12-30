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
        self._generic_prompt = '''
        你是一个专业的图像抄录员，你需要抄录图像中的文本内容，并输出Markdown格式的文本。你需要注意以下几点：
        1. 抄录的文本内容必须与图像中的内容一致。
        2. 如果图像中包含表格，你需要将表格抄录为Markdown格式的表格。
        3. 如果图像中包含公式，你需要将公式抄录为Markdown格式的公式。
        4. 如果图像中包含流程图或是架构图，你需要将其转换为Mermaid图像嵌入到Markdown中。
        5. 如果图像中包含代码，你需要将代码抄录为Markdown格式的代码块。
        6. 如果图像中包含电路图，你需要将电路图转换为Tikz代码块（使用CircuiTikz宏包）嵌入到Markdown中。
        '''

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
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.2,
            extra_body={
                'repetition_penalty': 1.02,
                'presence_penalty': 0.2,
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
                        'text': '请抄录图像中的文本内容'
                    }
                ]}
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            frequency_penalty=0.0,
            presence_penalty=0.2,
            extra_body={
                'repetition_penalty': 1.02,
                'presence_penalty': 0.2
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
