"""
Cache retriever with keyword and semantic search.

Provides both keyword-based and semantic search capabilities
for cached OCR content.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import faiss

from .cache_loader import CacheLoader
from ..models import PDFCache, SectionType
from ..knowledge_base.manager import KnowledgeBaseManager


class SemanticSearcher:
    """
    Semantic search using sentence embeddings and FAISS.

    Provides vector-based semantic search for cached content.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the semantic searcher.

        Args:
            model_name: Name of the sentence-transformers model
        """
        self._model_name = model_name
        self._encoder = None
        self._index = None
        self._documents = []
        self._metadata = []

    def _ensure_encoder(self):
        """Lazy load the encoder model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print("[SemanticSearch] Loading embedding model...")
            self._encoder = SentenceTransformer(self._model_name)
            print("[SemanticSearch] Model loaded")

    def build_index(self, documents: List[str], metadata: List[dict] = None):
        """
        Build FAISS index from documents.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        self._ensure_encoder()

        if not documents:
            return

        print(f"[SemanticSearch] Building index for {len(documents)} documents...")

        self._documents = documents
        self._metadata = metadata or [{}] * len(documents)

        # Generate embeddings
        embeddings = self._encoder.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype('float32'))

        print(f"[SemanticSearch] Index built with {self._index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, metadata, score) tuples
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        self._ensure_encoder()

        # Encode query
        query_embedding = self._encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        # Search
        distances, indices = self._index.search(query_embedding, min(k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._documents):
                # Convert L2 distance to similarity score (0-1)
                score = 1 / (1 + dist)
                results.append((
                    self._documents[idx],
                    self._metadata[idx] if idx < len(self._metadata) else {},
                    score
                ))

        return results

    def save_index(self, path: Path):
        """
        Save the FAISS index to disk.

        Args:
            path: Directory to save index files
        """
        if self._index is None:
            return

        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path.joinpath('index.faiss')))

        # Save documents and metadata
        import json
        with open(path.joinpath('documents.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self._documents,
                'metadata': self._metadata
            }, f, ensure_ascii=False, indent=2)

        print(f"[SemanticSearch] Index saved to {path}")

    def load_index(self, path: Path):
        """
        Load the FAISS index from disk.

        Args:
            path: Directory containing index files
        """
        index_file = path.joinpath('index.faiss')
        documents_file = path.joinpath('documents.json')

        if not index_file.exists() or not documents_file.exists():
            return

        # Load FAISS index
        self._index = faiss.read_index(str(index_file))

        # Load documents and metadata
        import json
        with open(documents_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self._documents = data['documents']
            self._metadata = data['metadata']

        print(f"[SemanticSearch] Index loaded from {path}")

    @property
    def is_ready(self) -> bool:
        """Check if the index is built and ready for search."""
        return self._index is not None and self._index.ntotal > 0


class CacheRetriever:
    """
    Main retriever for cached OCR content.

    Combines keyword search and semantic search.
    Supports knowledge base isolation.
    """

    def __init__(self, cache_dir: Path, kb_manager: KnowledgeBaseManager = None):
        """
        Initialize the cache retriever.

        Args:
            cache_dir: Base cache directory
            kb_manager: Optional KnowledgeBaseManager for KB isolation
        """
        self._cache_dir = cache_dir
        self._loader = CacheLoader(cache_dir)
        self._semantic_searcher = SemanticSearcher()
        self._caches = []
        self._index_path = cache_dir.joinpath('semantic_index')
        self._kb_manager = kb_manager

    def load_caches(self):
        """Load all cached PDF results."""
        self._caches = self._loader.load_all_caches()
        print(f"[CacheRetriever] Loaded {len(self._caches)} cached PDFs")

    def build_semantic_index(self):
        """Build semantic search index from loaded caches."""
        if not self._caches:
            self.load_caches()

        documents = []
        metadata = []

        for cache in self._caches:
            # Add full text
            documents.append(cache.full_text)
            metadata.append({
                'source': Path(cache.file_path).name,
                'type': 'full_text'
            })

            # Add individual sections
            for page_result in cache.page_results:
                for section in page_result.sections:
                    documents.append(section.content)
                    metadata.append({
                        'source': Path(cache.file_path).name,
                        'page': page_result.page_number,
                        'type': section.type.value,
                        'section_type': section.type.value
                    })

        self._semantic_searcher.build_index(documents, metadata)

    def load_or_build_index(self):
        """Load existing index or build a new one."""
        if self._index_path.exists():
            try:
                self._semantic_searcher.load_index(self._index_path)
                if self._semantic_searcher.is_ready:
                    return
            except Exception:
                pass

        # Build new index
        self.load_caches()
        self.build_semantic_index()

    def save_index(self):
        """Save the semantic index to disk."""
        self._semantic_searcher.save_index(self._index_path)

    def search(
        self,
        query: str,
        method: str = 'hybrid',
        limit: int = 5
    ) -> List[str]:
        """
        Search cached content.

        Args:
            query: Search query
            method: Search method ('keyword', 'semantic', 'hybrid')
            limit: Maximum number of results

        Returns:
            List of formatted search results
        """
        if not self._caches:
            self.load_caches()

        results = []

        if method in ('keyword', 'hybrid'):
            # Keyword search
            keyword_results = self._loader.search_keyword(query, self._caches)
            for result in keyword_results[:limit]:
                results.append(result)

        if method in ('semantic', 'hybrid'):
            # Semantic search
            if not self._semantic_searcher.is_ready:
                self.load_or_build_index()

            semantic_results = self._semantic_searcher.search(query, k=limit)
            for doc, meta, score in semantic_results:
                source = meta.get('source', 'Unknown')
                page = meta.get('page', '')
                page_info = f", Page {page}" if page else ""
                results.append(f"[{source}{page_info}] (similarity: {score:.2f})\n{doc}\n")

        # Deduplicate and limit results
        seen = set()
        unique_results = []
        for result in results:
            # Simple deduplication by first line
            first_line = result.split('\n')[0]
            if first_line not in seen:
                seen.add(first_line)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break

        return unique_results

    def search_in_kb(
        self,
        query: str,
        kb_id: str,
        method: str = 'semantic',
        limit: int = 5
    ) -> List[str]:
        """
        Search within a specific knowledge base.

        Args:
            query: Search query
            kb_id: Knowledge base ID
            method: Search method ('keyword', 'semantic', 'hybrid')
            limit: Maximum number of results

        Returns:
            List of formatted search results

        Raises:
            ValueError: If kb_id not found
        """
        if not self._kb_manager:
            raise ValueError("KnowledgeBaseManager not initialized")

        kb = self._kb_manager.get_knowledge_base(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        results = []

        if method in ('keyword', 'hybrid'):
            # Keyword search within KB PDFs
            if not self._caches:
                self.load_caches()

            # Filter caches to only KB PDFs
            kb_caches = [c for c in self._caches if Path(c.file_path).name in kb.pdf_files]
            keyword_results = self._loader.search_keyword(query, kb_caches)
            for result in keyword_results[:limit]:
                results.append(result)

        if method in ('semantic', 'hybrid'):
            # Semantic search using KB-specific index
            searcher = self._kb_manager.get_semantic_searcher(kb_id)
            if not searcher or not searcher.is_ready:
                # Build index if not exists
                self._kb_manager.build_kb_index(kb_id)
                searcher = self._kb_manager.get_semantic_searcher(kb_id)

            if searcher and searcher.is_ready:
                semantic_results = searcher.search(query, k=limit)
                for doc, meta, score in semantic_results:
                    source = meta.get('source', 'Unknown')
                    page = meta.get('page', '')
                    page_info = f", Page {page}" if page else ""
                    results.append(f"[{source}{page_info}] (similarity: {score:.2f})\n{doc}\n")

        # Deduplicate and limit results
        seen = set()
        unique_results = []
        for result in results:
            first_line = result.split('\n')[0]
            if first_line not in seen:
                seen.add(first_line)
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break

        return unique_results

    def get_sections_by_type(
        self,
        section_type: str,
        limit: int = 10
    ) -> List[str]:
        """
        Get sections of a specific type.

        Args:
            section_type: Type of section (table, formula, code, heading, etc.)
            limit: Maximum number of results

        Returns:
            List of section contents with source info
        """
        try:
            section_enum = SectionType(section_type)
        except ValueError:
            valid_types = [t.value for t in SectionType]
            raise ValueError(f"Invalid section type. Valid types: {valid_types}")

        if not self._caches:
            self.load_caches()

        sections = self._loader.get_sections_by_type(section_enum, self._caches)

        results = []
        for i, section in enumerate(sections[:limit]):
            results.append(f"[{i+1}] {section}")

        return results

    def list_cached_documents(self) -> List[str]:
        """
        List all cached PDF documents.

        Returns:
            List of PDF file names
        """
        return self._loader.list_cached_pdfs()
