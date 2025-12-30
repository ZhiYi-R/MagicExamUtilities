"""
Cache loader for OCR cached data.

Provides functionality to load and parse cached OCR results.
"""

import json
from pathlib import Path
from typing import List, Optional

from ..models import OCRResult, PDFCache, SectionType


class CacheLoader:
    """
    Loader for cached OCR data.

    Loads and parses cached OCR results from the cache directory.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the cache loader.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._ocr_cache_dir = cache_dir.joinpath('ocr')

    def list_cached_pdfs(self) -> List[str]:
        """
        List all cached PDF documents.

        Returns:
            List of PDF file names that have cached OCR results
        """
        if not self._ocr_cache_dir.exists():
            return []

        cached_files = []
        for cache_file in self._ocr_cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if it's a PDF cache (has page_results)
                    if 'page_results' in data:
                        file_path = data.get('file_path', '')
                        if file_path:
                            cached_files.append(Path(file_path).name)
            except Exception:
                continue

        return cached_files

    def load_pdf_cache(self, file_path: Path) -> Optional[PDFCache]:
        """
        Load cached OCR results for a PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFCache if cache exists, None otherwise
        """
        cache_file = self._ocr_cache_dir.joinpath(f'{file_path.stem}.json')

        if not cache_file.exists():
            return None

        try:
            return PDFCache.load(cache_file)
        except Exception:
            return None

    def load_all_caches(self) -> List[PDFCache]:
        """
        Load all cached PDF results.

        Returns:
            List of all cached PDFCache objects
        """
        caches = []
        if not self._ocr_cache_dir.exists():
            return caches

        for cache_file in self._ocr_cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if it's a PDF cache
                    if 'page_results' in data:
                        cache = PDFCache.load(cache_file)
                        caches.append(cache)
            except Exception:
                continue

        return caches

    def get_all_text(self, caches: Optional[List[PDFCache]] = None) -> str:
        """
        Get all text from caches.

        Args:
            caches: List of caches to get text from. If None, loads all caches.

        Returns:
            Concatenated text from all caches
        """
        if caches is None:
            caches = self.load_all_caches()

        texts = []
        for cache in caches:
            texts.append(f"# {Path(cache.file_path).name}\n")
            texts.append(cache.full_text)
            texts.append("\n\n")

        return "\n".join(texts)

    def get_sections_by_type(
        self,
        section_type: SectionType,
        caches: Optional[List[PDFCache]] = None
    ) -> List[str]:
        """
        Get all sections of a specific type.

        Args:
            section_type: Type of section to retrieve
            caches: List of caches to search. If None, loads all caches.

        Returns:
            List of section contents
        """
        if caches is None:
            caches = self.load_all_caches()

        sections = []
        for cache in caches:
            for page_result in cache.page_results:
                for section in page_result.sections:
                    if section.type == section_type:
                        sections.append(section.content)

        return sections

    def search_keyword(
        self,
        keyword: str,
        caches: Optional[List[PDFCache]] = None,
        case_sensitive: bool = False
    ) -> List[str]:
        """
        Search for keyword in cached text.

        Args:
            keyword: Keyword to search for
            caches: List of caches to search. If None, loads all caches.
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching text snippets with context
        """
        if caches is None:
            caches = self.load_all_caches()

        search_keyword = keyword if case_sensitive else keyword.lower()
        results = []

        for cache in caches:
            for page_result in cache.page_results:
                text = page_result.raw_text
                search_text = text if case_sensitive else text.lower()

                if search_keyword in search_text:
                    # Find context around the match
                    lines = text.split('\n')
                    for i, line in enumerate(lines):
                        line_search = line if case_sensitive else line.lower()
                        if search_keyword in line_search:
                            # Get context (2 lines before and after)
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            context = '\n'.join(lines[start:end])
                            source = f"{Path(cache.file_path).name}, Page {page_result.page_number}"
                            results.append(f"[{source}]\n{context}\n")
                            break

        return results
