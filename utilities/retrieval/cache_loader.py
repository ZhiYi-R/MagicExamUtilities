"""
Cache loader for OCR and STT cached data.

Provides functionality to load and parse cached OCR and STT results.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from ..models import OCRResult, PDFCache, AudioCache, SectionType


class CacheLoader:
    """
    Loader for cached OCR and STT data.

    Loads and parses cached OCR and STT results from the cache directory.

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
      stt/
        {audio_hash}/         # One directory per audio file
          cache.json          # STT cache data
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the cache loader.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._ocr_cache_dir = cache_dir.joinpath('ocr')
        self._ocr_stored_dir = self._ocr_cache_dir.joinpath('stored')
        self._stt_cache_dir = cache_dir.joinpath('stt')

    def list_cached_pdfs(self) -> List[str]:
        """
        List all cached PDF documents.

        Returns:
            List of PDF file names that have cached OCR results
        """
        if not self._ocr_stored_dir.exists():
            return []

        cached_files = []
        # New structure: ocr/stored/{pdf_hash}/cache.json
        for cache_dir in self._ocr_stored_dir.iterdir():
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
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

        # Also support legacy structure: ocr/{pdf_stem}.json
        for cache_file in self._ocr_cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if it's a PDF cache (has page_results)
                    if 'page_results' in data:
                        file_path = data.get('file_path', '')
                        if file_path:
                            pdf_name = Path(file_path).name
                            if pdf_name not in cached_files:
                                cached_files.append(pdf_name)
            except Exception:
                continue

        return cached_files

    def load_pdf_cache(self, file_path: Path) -> Optional[PDFCache]:
        """
        Load cached OCR results for a PDF.

        Tries both new structure (hash-based) and legacy structure (stem-based).

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFCache if cache exists, None otherwise
        """
        from ..cache import OCRCache
        from ..models import compute_file_hash

        # Check if file exists first (for hash computation)
        if not file_path.exists():
            # Try legacy structure by stem only
            cache_file_legacy = self._ocr_cache_dir.joinpath(f'{file_path.stem}.json')
            if cache_file_legacy.exists():
                try:
                    return PDFCache.load(cache_file_legacy)
                except Exception:
                    pass
            return None

        # Try new structure first: ocr/stored/{file_hash}/cache.json
        file_hash = compute_file_hash(file_path)
        cache_dir = self._ocr_stored_dir.joinpath(file_hash)
        cache_file = cache_dir.joinpath('cache.json')

        if cache_file.exists():
            try:
                return PDFCache.load(cache_file)
            except Exception:
                pass

        # Fallback to legacy structure: ocr/{pdf_stem}.json
        cache_file_legacy = self._ocr_cache_dir.joinpath(f'{file_path.stem}.json')
        if cache_file_legacy.exists():
            try:
                return PDFCache.load(cache_file_legacy)
            except Exception:
                pass

        return None

    def load_all_caches(self) -> List[PDFCache]:
        """
        Load all cached PDF results.

        Returns:
            List of all cached PDFCache objects
        """
        caches = []
        if not self._ocr_stored_dir.exists():
            return caches

        # New structure: ocr/stored/{pdf_hash}/cache.json
        for cache_dir in self._ocr_stored_dir.iterdir():
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if it's a PDF cache
                        if 'page_results' in data:
                            cache = PDFCache.load(cache_file)
                            caches.append(cache)
                except Exception:
                    continue

        # Also support legacy structure: ocr/{pdf_stem}.json
        for cache_file in self._ocr_cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if it's a PDF cache
                    if 'page_results' in data:
                        cache = PDFCache.load(cache_file)
                        # Avoid duplicates by checking file_hash
                        if not any(c.file_hash == cache.file_hash for c in caches):
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

    # STT Cache Methods

    def list_cached_audios(self) -> List[str]:
        """
        List all cached audio files.

        Returns:
            List of audio file names that have cached STT results
        """
        if not self._stt_cache_dir.exists():
            return []

        cached_files = []
        for cache_dir in self._stt_cache_dir.iterdir():
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if it's an audio cache (has raw_text)
                        if 'raw_text' in data and 'file_path' in data:
                            file_path = data.get('file_path', '')
                            if file_path:
                                cached_files.append(Path(file_path).name)
                except Exception:
                    continue

        return cached_files

    def load_audio_cache(self, file_path: Path) -> Optional[AudioCache]:
        """
        Load cached STT results for an audio file.

        Tries hash-based structure.

        Args:
            file_path: Path to the audio file

        Returns:
            AudioCache if cache exists, None otherwise
        """
        from ..models import compute_file_hash

        # Check if file exists first (for hash computation)
        if not file_path.exists():
            return None

        # Try hash-based structure: stt/{file_hash}/cache.json
        file_hash = compute_file_hash(file_path)
        cache_dir = self._stt_cache_dir.joinpath(file_hash)
        cache_file = cache_dir.joinpath('cache.json')

        if cache_file.exists():
            try:
                return AudioCache.load(cache_file)
            except Exception:
                pass

        return None

    def load_all_audio_caches(self) -> List[AudioCache]:
        """
        Load all cached audio STT results.

        Returns:
            List of all cached AudioCache objects
        """
        caches = []
        if not self._stt_cache_dir.exists():
            return caches

        for cache_dir in self._stt_cache_dir.iterdir():
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Check if it's an audio cache
                        if 'raw_text' in data:
                            cache = AudioCache.load(cache_file)
                            caches.append(cache)
                except Exception:
                    continue

        return caches

    def get_all_text(
        self,
        caches: Optional[Union[List[PDFCache], List[AudioCache]]] = None,
        include_stt: bool = True
    ) -> str:
        """
        Get all text from caches (both OCR and STT).

        Args:
            caches: List of caches to get text from. If None, loads all caches.
            include_stt: Whether to include STT caches (default: True)

        Returns:
            Concatenated text from all caches
        """
        texts = []

        # Handle PDF caches
        if caches is None:
            pdf_caches = self.load_all_caches()
        else:
            pdf_caches = [c for c in caches if isinstance(c, PDFCache)]

        for cache in pdf_caches:
            texts.append(f"# {Path(cache.file_path).name}\n")
            texts.append(cache.full_text)
            texts.append("\n\n")

        # Handle STT caches
        if include_stt:
            if caches is None:
                audio_caches = self.load_all_audio_caches()
            else:
                audio_caches = [c for c in caches if isinstance(c, AudioCache)]

            for cache in audio_caches:
                texts.append(f"# {Path(cache.file_path).name}\n")
                texts.append(cache.raw_text)
                texts.append("\n\n")

        return "\n".join(texts)

    def get_all_stt_text(self, audio_caches: Optional[List[AudioCache]] = None) -> str:
        """
        Get all text from STT caches only.

        Args:
            audio_caches: List of audio caches to get text from. If None, loads all STT caches.

        Returns:
            Concatenated text from all STT caches
        """
        if audio_caches is None:
            audio_caches = self.load_all_audio_caches()

        texts = []
        for cache in audio_caches:
            texts.append(f"# {Path(cache.file_path).name}\n")
            texts.append(cache.raw_text)
            texts.append("\n\n")

        return "\n".join(texts)

    def search_keyword_in_stt(
        self,
        keyword: str,
        audio_caches: Optional[List[AudioCache]] = None,
        case_sensitive: bool = False
    ) -> List[str]:
        """
        Search for keyword in cached STT text.

        Args:
            keyword: Keyword to search for
            audio_caches: List of audio caches to search. If None, loads all STT caches.
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matching text snippets with context
        """
        if audio_caches is None:
            audio_caches = self.load_all_audio_caches()

        search_keyword = keyword if case_sensitive else keyword.lower()
        results = []

        for cache in audio_caches:
            text = cache.raw_text
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
                        source = f"{Path(cache.file_path).name}"
                        results.append(f"[{source}]\n{context}\n")
                        break

        return results
