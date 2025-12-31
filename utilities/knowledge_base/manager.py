"""
Knowledge base manager.

Handles creation, modification, and indexing of knowledge bases.
"""

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .models import KnowledgeBase, KnowledgeBaseStore
from ..cache import OCRCache, STTCache

# Use TYPE_CHECKING for type hints to avoid circular import
if TYPE_CHECKING:
    from ..retrieval.retriever import SemanticSearcher


class KnowledgeBaseManager:
    """
    Manager for knowledge base operations.

    Handles CRUD operations for knowledge bases and manages
    their semantic search indices.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the manager.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._store = KnowledgeBaseStore(cache_dir)
        self._ocr_cache = OCRCache(cache_dir=cache_dir)
        self._stt_cache = STTCache(cache_dir=cache_dir)

    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        """
        List all knowledge bases.

        Returns:
            List of all KnowledgeBase objects
        """
        return self._store.load_all()

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """
        Get a specific knowledge base by ID.

        Args:
            kb_id: Knowledge base ID

        Returns:
            KnowledgeBase if found, None otherwise
        """
        for kb in self.list_knowledge_bases():
            if kb.id == kb_id:
                return kb
        return None

    def create_knowledge_base(
        self,
        kb_id: str,
        name: str,
        description: str = '',
        pdf_files: Optional[List[str]] = None,
        audio_files: Optional[List[str]] = None
    ) -> KnowledgeBase:
        """
        Create a new knowledge base.

        Args:
            kb_id: Unique identifier (alphanumeric, underscore, hyphen)
            name: Display name
            description: Description
            pdf_files: Optional list of PDF file names to include
            audio_files: Optional list of audio file names to include

        Returns:
            Created KnowledgeBase

        Raises:
            ValueError: If kb_id already exists
        """
        # Check for duplicate ID
        if self.get_knowledge_base(kb_id):
            raise ValueError(f"Knowledge base with ID '{kb_id}' already exists")

        # Validate kb_id format
        if not kb_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid kb_id: '{kb_id}'. Use only alphanumeric, underscore, and hyphen")

        kb = KnowledgeBase.create(kb_id, name, description, pdf_files, audio_files)

        # Save to store
        all_kbs = self.list_knowledge_bases()
        all_kbs.append(kb)
        self._store.save_all(all_kbs)

        # Build index if files are specified
        if kb.pdf_files or kb.audio_files:
            self.build_kb_index(kb_id)

        print(f"[KBManager] Created knowledge base: {kb_id}")
        return kb

    def delete_knowledge_base(self, kb_id: str) -> None:
        """
        Delete a knowledge base.

        Args:
            kb_id: Knowledge base ID to delete

        Raises:
            ValueError: If kb_id not found
        """
        all_kbs = self.list_knowledge_bases()
        filtered_kbs = [kb for kb in all_kbs if kb.id != kb_id]

        if len(all_kbs) == len(filtered_kbs):
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        # Delete index
        self._store.delete_kb_index(kb_id)

        # Save updated list
        self._store.save_all(filtered_kbs)

        print(f"[KBManager] Deleted knowledge base: {kb_id}")

    def add_pdf_to_kb(self, kb_id: str, pdf_name: str) -> None:
        """
        Add a PDF to a knowledge base.

        Args:
            kb_id: Knowledge base ID
            pdf_name: PDF file name

        Raises:
            ValueError: If kb_id not found
        """
        kb = self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb.add_pdf(pdf_name)

        # Save updated KB
        all_kbs = self.list_knowledge_bases()
        for i, existing_kb in enumerate(all_kbs):
            if existing_kb.id == kb_id:
                all_kbs[i] = kb
                break

        self._store.save_all(all_kbs)

        # Rebuild index
        self.build_kb_index(kb_id)

        print(f"[KBManager] Added '{pdf_name}' to knowledge base: {kb_id}")

    def remove_pdf_from_kb(self, kb_id: str, pdf_name: str) -> None:
        """
        Remove a PDF from a knowledge base.

        Args:
            kb_id: Knowledge base ID
            pdf_name: PDF file name

        Raises:
            ValueError: If kb_id not found
        """
        kb = self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb.remove_pdf(pdf_name)

        # Save updated KB
        all_kbs = self.list_knowledge_bases()
        for i, existing_kb in enumerate(all_kbs):
            if existing_kb.id == kb_id:
                all_kbs[i] = kb
                break

        self._store.save_all(all_kbs)

        # Rebuild index
        self.build_kb_index(kb_id)

        print(f"[KBManager] Removed '{pdf_name}' from knowledge base: {kb_id}")

    def build_kb_index(self, kb_id: str) -> None:
        """
        Build semantic search index for a knowledge base.

        Args:
            kb_id: Knowledge base ID

        Raises:
            ValueError: If kb_id not found
        """
        kb = self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        if not kb.has_content:
            print(f"[KBManager] KB '{kb_id}' has no content, skipping index build")
            return

        print(f"[KBManager] Building index for knowledge base: {kb_id}")

        # Get cached data from both PDFs and audio
        documents = []
        metadata = []

        # Process PDF files
        for pdf_name in kb.pdf_files:
            # Find the cache file for this PDF (look in stored/ subdirectory)
            cache_path = self._cache_dir.joinpath('ocr', 'stored')
            found = False

            for cache_dir in cache_path.glob('*'):
                if not cache_dir.is_dir():
                    continue
                cache_file = cache_dir.joinpath('cache.json')
                if not cache_file.exists():
                    continue

                try:
                    from ..models import PDFCache
                    pdf_cache = PDFCache.load(cache_file)
                    if Path(pdf_cache.file_path).name == pdf_name:
                        # Found matching cache
                        # Add full text
                        documents.append(pdf_cache.full_text)
                        metadata.append({
                            'source': pdf_name,
                            'type': 'pdf_full_text',
                            'kb_id': kb_id
                        })

                        # Add sections
                        for page_result in pdf_cache.page_results:
                            for section in page_result.sections:
                                documents.append(section.content)
                                metadata.append({
                                    'source': pdf_name,
                                    'page': page_result.page_number,
                                    'type': section.type.value,
                                    'kb_id': kb_id
                                })
                        found = True
                        break
                except Exception:
                    continue

        # Process audio files
        for audio_name in kb.audio_files:
            # Find the cache file for this audio
            cache_path = self._cache_dir.joinpath('stt')
            found = False

            for cache_dir in cache_path.glob('*'):
                if not cache_dir.is_dir():
                    continue
                cache_file = cache_dir.joinpath('cache.json')
                if not cache_file.exists():
                    continue

                try:
                    from ..models import AudioCache
                    audio_cache = AudioCache.load(cache_file)
                    if Path(audio_cache.file_path).name == audio_name:
                        # Found matching cache
                        # Add full transcript
                        documents.append(audio_cache.raw_text)
                        metadata.append({
                            'source': audio_name,
                            'type': 'audio_transcript',
                            'kb_id': kb_id
                        })
                        found = True
                        break
                except Exception:
                    continue

        if not documents:
            print(f"[KBManager] No cached data found for KB '{kb_id}'")
            return

        # Build semantic index
        index_path = self._store.get_kb_index_path(kb_id)
        print(f"[KBManager] Building semantic index at {index_path}...")
        from ..retrieval.retriever import SemanticSearcher
        searcher = SemanticSearcher()
        searcher.build_index(documents, metadata)
        print(f"[KBManager] Index built, saving to disk...")
        searcher.save_index(index_path)

        print(f"[KBManager] Index built for '{kb_id}': {len(documents)} documents")

    def get_available_pdfs(self) -> List[str]:
        """
        Get list of PDF files that have cached OCR data.

        Returns:
            List of PDF file names
        """
        cache_path = self._cache_dir.joinpath('ocr', 'stored')
        pdf_names = []

        for cache_dir in cache_path.glob('*'):
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
                try:
                    from ..models import PDFCache
                    pdf_cache = PDFCache.load(cache_file)
                    pdf_names.append(Path(pdf_cache.file_path).name)
                except Exception:
                    continue

        return pdf_names

    def get_available_audios(self) -> List[str]:
        """
        Get list of audio files that have cached STT data.

        Returns:
            List of audio file names
        """
        cache_path = self._cache_dir.joinpath('stt')
        audio_names = []

        for cache_dir in cache_path.glob('*'):
            if not cache_dir.is_dir():
                continue
            cache_file = cache_dir.joinpath('cache.json')
            if cache_file.exists():
                try:
                    from ..models import AudioCache
                    audio_cache = AudioCache.load(cache_file)
                    audio_names.append(Path(audio_cache.file_path).name)
                except Exception:
                    continue

        return audio_names

    def add_audio_to_kb(self, kb_id: str, audio_name: str) -> None:
        """
        Add an audio file to a knowledge base.

        Args:
            kb_id: Knowledge base ID
            audio_name: Audio file name

        Raises:
            ValueError: If kb_id not found
        """
        kb = self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb.add_audio(audio_name)

        # Save updated KB
        all_kbs = self.list_knowledge_bases()
        for i, existing_kb in enumerate(all_kbs):
            if existing_kb.id == kb_id:
                all_kbs[i] = kb
                break

        self._store.save_all(all_kbs)

        # Rebuild index
        self.build_kb_index(kb_id)

        print(f"[KBManager] Added '{audio_name}' to knowledge base: {kb_id}")

    def remove_audio_from_kb(self, kb_id: str, audio_name: str) -> None:
        """
        Remove an audio file from a knowledge base.

        Args:
            kb_id: Knowledge base ID
            audio_name: Audio file name

        Raises:
            ValueError: If kb_id not found
        """
        kb = self.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb.remove_audio(audio_name)

        # Save updated KB
        all_kbs = self.list_knowledge_bases()
        for i, existing_kb in enumerate(all_kbs):
            if existing_kb.id == kb_id:
                all_kbs[i] = kb
                break

        self._store.save_all(all_kbs)

        # Rebuild index
        self.build_kb_index(kb_id)

        print(f"[KBManager] Removed '{audio_name}' from knowledge base: {kb_id}")

    def get_semantic_searcher(self, kb_id: str) -> Optional["SemanticSearcher"]:
        """
        Get semantic searcher for a specific knowledge base.

        Args:
            kb_id: Knowledge base ID

        Returns:
            SemanticSearcher if index exists, None otherwise
        """
        index_path = self._store.get_kb_index_path(kb_id)
        if not index_path.exists():
            return None

        from ..retrieval.retriever import SemanticSearcher
        searcher = SemanticSearcher()
        searcher.load_index(index_path)

        if searcher.is_ready:
            return searcher

        return None
