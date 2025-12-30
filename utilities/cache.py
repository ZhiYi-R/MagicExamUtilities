"""
Cache management for OCR and STT results.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from .models import PDFCache, OCRResult, AudioCache, compute_file_hash


class OCRCache:
    """
    Manager for OCR cache operations.

    Handles loading, saving, and validating cached OCR results.

    Directory structure:
    cache/
      ocr/
        stored/              # Directory for cached PDF data
          {pdf_hash}/        # One directory per PDF
            cache.json       # OCR cache data
            page_0.jpg       # Image files
            page_1.jpg
            ...
            page_0.json      # OCR response dump for page 0
            page_1.json      # OCR response dump for page 1
            ...
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
        self._ocr_stored_dir = self._ocr_cache_dir.joinpath('stored')
        self._ocr_stored_dir.mkdir(parents=True, exist_ok=True)

    def _get_pdf_cache_dir(self, file_hash: str) -> Path:
        """Get the cache directory for a specific PDF (by hash)."""
        return self._ocr_stored_dir.joinpath(file_hash)

    def get_pdf_cache(self, pdf_path: Path) -> Optional[PDFCache]:
        """
        Get cached OCR results for a PDF if available and valid.

        Uses hash-based cache validation - cache is valid if the file hash matches.
        This allows cache to work even if the file is copied/moved to a different location.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFCache if cache is valid, None otherwise
        """
        file_hash = compute_file_hash(pdf_path)
        pdf_cache_dir = self._get_pdf_cache_dir(file_hash)
        cache_path = pdf_cache_dir.joinpath('cache.json')

        if not cache_path.exists():
            return None

        # Load the cache and verify the hash matches
        try:
            cache = PDFCache.load(cache_path)
            # Verify the stored hash matches the current file hash
            if cache.file_hash != file_hash:
                print(f'[Cache] PDF file hash changed for {pdf_path.name} (stored: {cache.file_hash[:8]}..., current: {file_hash[:8]}...)')
                return None

            print(f'[Cache] Loading cached OCR results for {pdf_path.name}')
            return cache
        except Exception as e:
            print(f'[Cache] Failed to load cache for {pdf_path.name}: {e}')
            return None

    def get_page_cache(self, image_path: Path) -> Optional[OCRResult]:
        """
        Get cached OCR result for a single page image.

        Checks for cached OCR result in two formats:
        1. OCRResult format (page_n.ocr.json)
        2. API response dump format (page_n.json from OCRWorker)

        No hash validation - if a cache file exists, use it.

        Args:
            image_path: Path to the page image

        Returns:
            OCRResult if cache exists, None otherwise
        """
        import json

        # Try OCRResult format first
        ocr_cache_path = image_path.parent.joinpath(f'{image_path.stem}.ocr.json')
        if ocr_cache_path.exists():
            try:
                result = OCRResult.load(ocr_cache_path)
                print(f'[Cache] Loading cached OCR result for {image_path.name}')
                return result
            except Exception as e:
                print(f'[Cache] Failed to load OCR cache: {e}')

        # Try API response dump format
        dump_cache_path = image_path.parent.joinpath(f'{image_path.stem}.json')
        if not dump_cache_path.exists():
            return None

        try:
            with open(dump_cache_path, 'r') as f:
                dump_data = json.load(f)

            # Extract OCR text from API response
            if 'choices' in dump_data and len(dump_data['choices']) > 0:
                content = dump_data['choices'][0]['message']['content']
            else:
                return None

            # Apply the same post-processing that OCRWorker does
            from utilities.workers.ocr_worker import _strip_outer_markdown_block, _detect_and_fix_duplicates
            content = _strip_outer_markdown_block(content)
            content, _ = _detect_and_fix_duplicates(content)

            # Compute file hash
            file_hash = compute_file_hash(image_path)

            # Extract page number from filename
            try:
                page_number = int(image_path.stem.split('_')[-1])
            except (ValueError, IndexError):
                page_number = 0

            # Construct OCRResult from dump
            from utilities.models import estimate_tokens
            result = OCRResult(
                file_path=str(image_path),
                file_hash=file_hash,
                page_number=page_number,
                page_hash=file_hash,
                timestamp=0,
                raw_text=content,
                sections=[],
                char_count=len(content),
                token_estimate=estimate_tokens(content)
            )
            print(f'[Cache] Loaded OCR result from API dump for {image_path.name}')
            return result

        except Exception as e:
            print(f'[Cache] Failed to load API dump for {image_path.name}: {e}')
            return None

    def save_page_cache(self, ocr_result: OCRResult) -> None:
        """
        Save a single page's OCR result to cache.

        Cache is saved with .ocr.json extension to avoid overwriting API dump files.

        Args:
            ocr_result: The OCR result to save
        """
        # Save with .ocr.json extension to avoid conflict with API dump files
        image_path = Path(ocr_result.file_path)
        cache_path = image_path.parent.joinpath(f'{image_path.stem}.ocr.json')

        ocr_result.save(cache_path)
        print(f'[Cache] Saved OCR cache for {image_path.name}')

    def clear_page_cache(self, image_path: Path) -> None:
        """
        Clear cache for a specific page.

        Clears both OCRResult cache (.ocr.json) and API dump (.json).

        Args:
            image_path: Path to the page image
        """
        ocr_cache_path = image_path.parent.joinpath(f'{image_path.stem}.ocr.json')
        dump_cache_path = image_path.parent.joinpath(f'{image_path.stem}.json')

        cleared = False
        if ocr_cache_path.exists():
            ocr_cache_path.unlink()
            cleared = True
        if dump_cache_path.exists():
            dump_cache_path.unlink()
            cleared = True

        if cleared:
            print(f'[Cache] Cleared cache for {image_path.name}')

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
        pdf_cache_dir = self._get_pdf_cache_dir(file_hash)
        pdf_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = pdf_cache_dir.joinpath('cache.json')

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

    def get_pdf_cache_dir(self, pdf_path: Path) -> Path:
        """
        Get the cache directory for a PDF (where images should be stored).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Directory path for storing PDF-related cache and images
        """
        file_hash = compute_file_hash(pdf_path)
        return self._get_pdf_cache_dir(file_hash)

    def save_page_result(self, image_path: Path, ocr_result: OCRResult) -> None:
        """
        Save a single page's OCR result to cache.

        Args:
            image_path: Path to the image file
            ocr_result: The OCR result to save
        """
        # Page results are saved in the PDF's cache directory
        # Extract PDF hash from the image path (format: {pdf_hash}/page_{n}.jpg)
        parent_dir = image_path.parent
        if parent_dir.name == compute_file_hash(Path(image_path)):
            # This is a hash-based directory
            cache_path = parent_dir.joinpath('cache.json')
        else:
            # Fallback to old structure - save to ocr directory
            cache_path = self._ocr_cache_dir.joinpath(f'{ocr_result.file_hash}.json')
        ocr_result.save(cache_path)

    def get_page_result(self, image_path: Path, image_hash: str) -> Optional[OCRResult]:
        """
        Get cached OCR result for a single image.

        Uses hash-based cache validation - cache is valid if the image hash matches.

        Args:
            image_path: Path to the image file
            image_hash: MD5 hash of the image

        Returns:
            OCRResult if cache exists and valid, None otherwise
        """
        # Try to find cache in the image's parent directory first
        parent_dir = image_path.parent
        cache_path = parent_dir.joinpath('cache.json')

        if not cache_path.exists():
            # Fallback to old structure
            cache_path = self._ocr_cache_dir.joinpath(f'{image_hash}.json')

        if not cache_path.exists():
            return None

        # Load the cache and verify the hash matches
        try:
            result = OCRResult.load(cache_path)
            # Verify the stored hash matches the provided hash
            if result.file_hash != image_hash:
                print(f'[Cache] Cache hash mismatch for {image_path.name}')
                return None

            return result
        except Exception as e:
            print(f'[Cache] Failed to load page cache for {image_path.name}: {e}')
            return None

    def clear_cache(self, pdf_path: Optional[Path] = None) -> None:
        """
        Clear cache for a specific PDF or all caches.

        Args:
            pdf_path: If specified, clear cache for this PDF only. Otherwise clear all.
        """
        if pdf_path:
            file_hash = compute_file_hash(pdf_path)
            pdf_cache_dir = self._get_pdf_cache_dir(file_hash)
            if pdf_cache_dir.exists():
                shutil.rmtree(pdf_cache_dir)
                print(f'[Cache] Cleared cache for {pdf_path.name}')
        else:
            # Clear all OCR caches in the stored directory
            if self._ocr_stored_dir.exists():
                for cache_dir in self._ocr_stored_dir.iterdir():
                    if cache_dir.is_dir():
                        shutil.rmtree(cache_dir)
            print(f'[Cache] Cleared all OCR caches')


class STTCache:
    """
    Manager for STT cache operations.

    Handles loading, saving, and validating cached STT results.

    Directory structure:
    cache/
      stt/
        {audio_hash}/          # One directory per audio file
          cache.json           # STT cache data
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._stt_cache_dir = cache_dir.joinpath('stt')
        self._stt_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_audio_cache_dir(self, file_hash: str) -> Path:
        """Get the cache directory for a specific audio file (by hash)."""
        return self._stt_cache_dir.joinpath(file_hash)

    def get_audio_cache(self, audio_path: Path) -> Optional[AudioCache]:
        """
        Get cached STT results for an audio file if available and valid.

        Uses hash-based cache validation - cache is valid if the file hash matches.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioCache if cache is valid, None otherwise
        """
        file_hash = compute_file_hash(audio_path)
        audio_cache_dir = self._get_audio_cache_dir(file_hash)
        cache_path = audio_cache_dir.joinpath('cache.json')

        if not cache_path.exists():
            return None

        # Load the cache and verify the hash matches
        try:
            cache = AudioCache.load(cache_path)
            # Verify the stored hash matches the current file hash
            if cache.file_hash != file_hash:
                print(f'[Cache] Cache hash mismatch for {audio_path.name} (stored: {cache.file_hash[:8]}..., current: {file_hash[:8]}...)')
                return None

            print(f'[Cache] Loading cached STT results for {audio_path.name}')
            return cache
        except Exception as e:
            print(f'[Cache] Failed to load cache for {audio_path.name}: {e}')
            return None

    def save_audio_cache(self, audio_path: Path, text: str) -> AudioCache:
        """
        Save STT results to cache.

        Args:
            audio_path: Path to the audio file
            text: Transcribed text from STT

        Returns:
            The saved AudioCache object
        """
        from .models import estimate_tokens

        file_hash = compute_file_hash(audio_path)
        audio_cache_dir = self._get_audio_cache_dir(file_hash)
        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = audio_cache_dir.joinpath('cache.json')

        cache = AudioCache(
            file_path=str(audio_path),
            file_hash=file_hash,
            timestamp=os.path.getmtime(audio_path),
            raw_text=text,
            char_count=len(text),
            token_estimate=estimate_tokens(text)
        )

        cache.save(cache_path)
        print(f'[Cache] Saved STT cache for {audio_path.name} to {cache_path}')
        return cache

    def get_audio_cache_dir(self, audio_path: Path) -> Path:
        """
        Get the cache directory for an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Directory path for storing audio-related cache
        """
        file_hash = compute_file_hash(audio_path)
        return self._get_audio_cache_dir(file_hash)

    def clear_cache(self, audio_path: Optional[Path] = None) -> None:
        """
        Clear cache for a specific audio file or all caches.

        Args:
            audio_path: If specified, clear cache for this audio file only. Otherwise clear all.
        """
        if audio_path:
            file_hash = compute_file_hash(audio_path)
            audio_cache_dir = self._get_audio_cache_dir(file_hash)
            if audio_cache_dir.exists():
                shutil.rmtree(audio_cache_dir)
                print(f'[Cache] Cleared cache for {audio_path.name}')
        else:
            # Clear all STT caches
            for cache_dir in self._stt_cache_dir.iterdir():
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
            print(f'[Cache] Cleared all STT caches')
