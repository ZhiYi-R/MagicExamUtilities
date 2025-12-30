"""
OCR Worker for processing images with OCR models.
Supports both DeepSeek OCR and generic vision-language models.
"""

import os
import pathlib
from typing import Optional

from ..worker import BaseWorker, get_rate_limit_config
from ..DeepSeekOCR import DeepSeekOCR
from ..GenericVisionLanguageOCR import GenericVL


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

        # Select OCR implementation based on environment variable
        use_deepseek = os.environ.get('OCR_USE_DEEPSEEK_OCR', '1') == '1'

        if use_deepseek:
            self._ocr = DeepSeekOCR(
                dump_dir=dump_dir,
                dump_ocr_response=dump_ocr_response
            )
            self._ocr_type = 'deepseek'
        else:
            self._ocr = GenericVL(
                dump_dir=dump_dir,
                dump_ocr_response=dump_ocr_response
            )
            self._ocr_type = 'generic'

        # Register methods after OCR is initialized
        self._register_methods()

        print(f"[OCRWorker] Initialized with {self._ocr_type} OCR")
        print(f"[OCRWorker] Rate limits: RPM={rpm}, TPM={tpm}")

    def _register_methods(self) -> None:
        """Register OCR methods."""
        self._methods = {
            'ocr': self._ocr.ocr,
        }

    def _estimate_tokens(self, task) -> int:
        """
        Estimate tokens for OCR requests.
        OCR typically consumes more tokens for image processing.
        """
        # Conservative estimate: 1000 tokens per image
        return 1000

    def process_image(self, image_path: pathlib.Path, timeout: Optional[float] = None) -> str:
        """
        Process a single image with OCR.

        Args:
            image_path: Path to the image file
            timeout: Maximum time to wait for result (seconds)

        Returns:
            OCR result as text
        """
        future = self.submit('ocr', image_path)
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
