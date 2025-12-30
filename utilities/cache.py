"""
Cache management for OCR results.
"""

import os
from pathlib import Path
from typing import Optional

from .models import PDFCache, OCRResult, compute_file_hash


class OCRCache:
    """
    Manager for OCR cache operations.

    Handles loading, saving, and validating cached OCR results.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._ocr_cache_dir = cache_dir.joinpath('ocr')
        self._ocr_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf_cache(self, pdf_path: Path) -> Optional[PDFCache]:
        """
        Get cached OCR results for a PDF if available and valid.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFCache if cache is valid, None otherwise
        """
        file_hash = compute_file_hash(pdf_path)
        cache_path = self._ocr_cache_dir.joinpath(f'{file_hash}.json')

        if not cache_path.exists():
            return None

        # Check if cache is newer than the PDF
        cache_mtime = cache_path.stat().st_mtime
        pdf_mtime = pdf_path.stat().st_mtime

        if cache_mtime < pdf_mtime:
            print(f'[Cache] Cache outdated for {pdf_path.name}')
            return None

        print(f'[Cache] Loading cached OCR results for {pdf_path.name}')
        return PDFCache.load(cache_path)

    def save_pdf_cache(self, pdf_path: Path, page_results: list[OCRResult], full_text: str) -> PDFCache:
        """
        Save OCR results to cache.

        Args:
            pdf_path: Path to the PDF file
            page_results: List of OCR results for each page
            full_text: Concatenated text from all pages

        Returns:
            The saved PDFCache object
        """
        file_hash = compute_file_hash(pdf_path)
        cache_path = self._ocr_cache_dir.joinpath(f'{file_hash}.json')

        cache = PDFCache(
            file_path=str(pdf_path),
            file_hash=file_hash,
            timestamp=os.path.getmtime(pdf_path),
            page_results=page_results,
            full_text=full_text
        )

        cache.save(cache_path)
        print(f'[Cache] Saved OCR cache for {pdf_path.name} to {cache_path}')
        return cache

    def save_page_result(self, image_path: Path, ocr_result: OCRResult) -> None:
        """
        Save a single page's OCR result to cache.

        Args:
            image_path: Path to the image file
            ocr_result: The OCR result to save
        """
        cache_path = self._ocr_cache_dir.joinpath(f'{ocr_result.file_hash}.json')
        ocr_result.save(cache_path)

    def get_page_result(self, image_path: Path, image_hash: str) -> Optional[OCRResult]:
        """
        Get cached OCR result for a single image.

        Args:
            image_path: Path to the image file
            image_hash: MD5 hash of the image

        Returns:
            OCRResult if cache exists, None otherwise
        """
        cache_path = self._ocr_cache_dir.joinpath(f'{image_hash}.json')

        if not cache_path.exists():
            return None

        # Check if cache is newer than the image
        cache_mtime = cache_path.stat().st_mtime
        if image_path.exists():
            image_mtime = image_path.stat().st_mtime
            if cache_mtime < image_mtime:
                return None

        return OCRResult.load(cache_path)

    def clear_cache(self, pdf_path: Optional[Path] = None) -> None:
        """
        Clear cache for a specific PDF or all caches.

        Args:
            pdf_path: If specified, clear cache for this PDF only. Otherwise clear all.
        """
        if pdf_path:
            file_hash = compute_file_hash(pdf_path)
            cache_path = self._ocr_cache_dir.joinpath(f'{file_hash}.json')
            if cache_path.exists():
                cache_path.unlink()
                print(f'[Cache] Cleared cache for {pdf_path.name}')
        else:
            # Clear all OCR caches
            for cache_file in self._ocr_cache_dir.glob('*.json'):
                cache_file.unlink()
            print(f'[Cache] Cleared all OCR caches')
