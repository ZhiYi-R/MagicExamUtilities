"""
Data models for OCR results and caching.
"""

import dataclasses
import hashlib
import json
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Any


class SectionType(Enum):
    """Types of OCR sections."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class OCRSection:
    """
    A section of OCR text with structure information.

    Attributes:
        type: Type of the section
        content: The text content
        level: Heading level (for headings)
        language: Detected language (optional)
    """
    type: SectionType
    content: str
    level: Optional[int] = None
    language: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type.value,
            'content': self.content,
            'level': self.level,
            'language': self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OCRSection':
        """Create from dictionary."""
        return cls(
            type=SectionType(data['type']),
            content=data['content'],
            level=data.get('level'),
            language=data.get('language'),
        )


@dataclasses.dataclass
class OCRResult:
    """
    Structured OCR result for a single image/page.

    Attributes:
        file_path: Path to the source image file
        file_hash: MD5 hash of the image file
        page_number: Page number in the PDF
        timestamp: When the OCR was performed
        raw_text: Raw OCR text output
        sections: Parsed sections (optional)
        char_count: Character count of raw_text
        token_estimate: Estimated token count
    """
    file_path: str
    file_hash: str
    page_number: int
    timestamp: str
    raw_text: str
    sections: List[OCRSection]
    char_count: int
    token_estimate: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'page_number': self.page_number,
            'timestamp': self.timestamp,
            'raw_text': self.raw_text,
            'sections': [s.to_dict() for s in self.sections],
            'char_count': self.char_count,
            'token_estimate': self.token_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OCRResult':
        """Create from dictionary."""
        return cls(
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            page_number=data['page_number'],
            timestamp=data['timestamp'],
            raw_text=data['raw_text'],
            sections=[OCRSection.from_dict(s) for s in data.get('sections', [])],
            char_count=data['char_count'],
            token_estimate=data['token_estimate'],
        )

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'OCRResult':
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclasses.dataclass
class PDFCache:
    """
    Cache data for a processed PDF.

    Attributes:
        file_path: Path to the PDF file
        file_hash: MD5 hash of the PDF file
        timestamp: When the PDF was processed
        page_results: List of OCR results for each page
        full_text: Concatenated OCR text from all pages
    """
    file_path: str
    file_hash: str
    timestamp: str
    page_results: List[OCRResult]
    full_text: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'timestamp': self.timestamp,
            'page_results': [r.to_dict() for r in self.page_results],
            'full_text': self.full_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PDFCache':
        """Create from dictionary."""
        return cls(
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            timestamp=data['timestamp'],
            page_results=[OCRResult.from_dict(r) for r in data['page_results']],
            full_text=data['full_text'],
        )

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'PDFCache':
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal MD5 hash string
    """
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough estimate: 1 token ≈ 3 characters for Chinese).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Rough estimate: Chinese ~1.5 chars/token, English ~4 chars/token
    # Use 3 as a conservative middle ground
    return max(1, len(text) // 3)
