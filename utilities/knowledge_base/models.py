"""
Data models for knowledge base management.
"""

import dataclasses
import json
import os
from pathlib import Path
from typing import List, Optional


@dataclasses.dataclass
class KnowledgeBase:
    """
    Knowledge base configuration.

    A knowledge base groups related PDF documents and audio transcripts together
    for isolated retrieval.

    Attributes:
        id: Unique identifier (e.g., 'database_sys')
        name: Display name (e.g., '数据库系统')
        description: Human-readable description
        pdf_files: List of PDF file names associated with this KB
        audio_files: List of audio file names associated with this KB
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str
    name: str
    description: str
    pdf_files: List[str]
    audio_files: List[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'pdf_files': self.pdf_files,
            'audio_files': self.audio_files,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeBase':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            pdf_files=data.get('pdf_files', []),
            audio_files=data.get('audio_files', []),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
        )

    @classmethod
    def create(cls, id: str, name: str, description: str = '', pdf_files: Optional[List[str]] = None, audio_files: Optional[List[str]] = None) -> 'KnowledgeBase':
        """
        Create a new knowledge base with timestamp.

        Args:
            id: Unique identifier
            name: Display name
            description: Description
            pdf_files: Associated PDF files
            audio_files: Associated audio files

        Returns:
            New KnowledgeBase instance
        """
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(
            id=id,
            name=name,
            description=description,
            pdf_files=pdf_files or [],
            audio_files=audio_files or [],
            created_at=timestamp,
            updated_at=timestamp,
        )

    def add_pdf(self, pdf_name: str) -> None:
        """Add a PDF file to the knowledge base."""
        if pdf_name not in self.pdf_files:
            self.pdf_files.append(pdf_name)
            import time
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def remove_pdf(self, pdf_name: str) -> None:
        """Remove a PDF file from the knowledge base."""
        if pdf_name in self.pdf_files:
            self.pdf_files.remove(pdf_name)
            import time
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def add_audio(self, audio_name: str) -> None:
        """Add an audio file to the knowledge base."""
        if audio_name not in self.audio_files:
            self.audio_files.append(audio_name)
            import time
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def remove_audio(self, audio_name: str) -> None:
        """Remove an audio file from the knowledge base."""
        if audio_name in self.audio_files:
            self.audio_files.remove(audio_name)
            import time
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def has_content(self) -> bool:
        """Check if knowledge base has any content."""
        return bool(self.pdf_files or self.audio_files)


class KnowledgeBaseStore:
    """
    Storage manager for knowledge bases.

    Handles loading and saving knowledge base configurations.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the store.

        Args:
            cache_dir: Base cache directory
        """
        self._cache_dir = cache_dir
        self._config_file = cache_dir.joinpath('knowledge_bases.json')
        self._indices_dir = cache_dir.joinpath('kb_indices')
        self._indices_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> List[KnowledgeBase]:
        """
        Load all knowledge bases.

        Returns:
            List of KnowledgeBase objects
        """
        if not self._config_file.exists():
            return []

        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [KnowledgeBase.from_dict(kb_data) for kb_data in data]
        except Exception:
            return []

    def save_all(self, knowledge_bases: List[KnowledgeBase]) -> None:
        """
        Save all knowledge bases.

        Args:
            knowledge_bases: List of KnowledgeBase objects
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, 'w', encoding='utf-8') as f:
            data = [kb.to_dict() for kb in knowledge_bases]
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_kb_index_path(self, kb_id: str) -> Path:
        """
        Get the index directory for a specific knowledge base.

        Args:
            kb_id: Knowledge base ID

        Returns:
            Path to the index directory
        """
        return self._indices_dir.joinpath(kb_id)

    def delete_kb_index(self, kb_id: str) -> None:
        """
        Delete index directory for a knowledge base.

        Args:
            kb_id: Knowledge base ID
        """
        index_path = self.get_kb_index_path(kb_id)
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
