"""
Tests for data models (OCRResult, OCRSection, PDFCache, etc.).
"""

import json
import pytest
from pathlib import Path

from utilities.models import (
    SectionType,
    OCRSection,
    OCRResult,
    PDFCache,
    compute_file_hash,
    estimate_tokens,
)


class TestSectionType:
    """Tests for SectionType enum."""

    def test_section_type_values(self):
        """Test that SectionType has all expected values."""
        assert SectionType.HEADING.value == "heading"
        assert SectionType.PARAGRAPH.value == "paragraph"
        assert SectionType.TABLE.value == "table"
        assert SectionType.FORMULA.value == "formula"
        assert SectionType.CODE.value == "code"
        assert SectionType.LIST.value == "list"
        assert SectionType.UNKNOWN.value == "unknown"


class TestOCRSection:
    """Tests for OCRSection dataclass."""

    def test_section_creation(self):
        """Test creating an OCRSection."""
        section = OCRSection(
            type=SectionType.HEADING,
            content="Test Heading",
            level=1,
            language="zh"
        )
        assert section.type == SectionType.HEADING
        assert section.content == "Test Heading"
        assert section.level == 1
        assert section.language == "zh"

    def test_section_with_optional_fields(self):
        """Test creating an OCRSection without optional fields."""
        section = OCRSection(
            type=SectionType.PARAGRAPH,
            content="Test paragraph"
        )
        assert section.type == SectionType.PARAGRAPH
        assert section.content == "Test paragraph"
        assert section.level is None
        assert section.language is None

    def test_section_to_dict(self):
        """Test converting OCRSection to dictionary."""
        section = OCRSection(
            type=SectionType.CODE,
            content="print('hello')",
            level=2,
            language="python"
        )
        data = section.to_dict()
        assert data == {
            'type': 'code',
            'content': "print('hello')",
            'level': 2,
            'language': 'python'
        }

    def test_section_to_dict_without_optional(self):
        """Test converting OCRSection without optional fields."""
        section = OCRSection(
            type=SectionType.PARAGRAPH,
            content="Test"
        )
        data = section.to_dict()
        assert data == {
            'type': 'paragraph',
            'content': 'Test',
            'level': None,
            'language': None
        }

    def test_section_from_dict(self):
        """Test creating OCRSection from dictionary."""
        data = {
            'type': 'table',
            'content': '| A | B |',
            'level': None,
            'language': None
        }
        section = OCRSection.from_dict(data)
        assert section.type == SectionType.TABLE
        assert section.content == '| A | B |'

    def test_section_roundtrip(self):
        """Test roundtrip conversion (to_dict and from_dict)."""
        original = OCRSection(
            type=SectionType.FORMULA,
            content="E = mc^2",
            level=None,
            language="latex"
        )
        data = original.to_dict()
        restored = OCRSection.from_dict(data)
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.level == original.level
        assert restored.language == original.language


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """Test creating an OCRResult."""
        sections = [
            OCRSection(type=SectionType.HEADING, content="Title", level=1),
            OCRSection(type=SectionType.PARAGRAPH, content="Content"),
        ]
        result = OCRResult(
            file_path="/test/image.jpg",
            file_hash="abc123",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Title\n\nContent",
            sections=sections,
            char_count=14,
            token_estimate=5
        )
        assert result.file_path == "/test/image.jpg"
        assert result.file_hash == "abc123"
        assert result.page_number == 0
        assert result.raw_text == "Title\n\nContent"
        assert len(result.sections) == 2
        assert result.char_count == 14
        assert result.token_estimate == 5

    def test_ocr_result_to_dict(self):
        """Test converting OCRResult to dictionary."""
        sections = [
            OCRSection(type=SectionType.HEADING, content="Title", level=1),
        ]
        result = OCRResult(
            file_path="/test/image.jpg",
            file_hash="abc123",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Title",
            sections=sections,
            char_count=5,
            token_estimate=2
        )
        data = result.to_dict()
        assert data['file_path'] == "/test/image.jpg"
        assert data['file_hash'] == "abc123"
        assert data['page_number'] == 0
        assert data['raw_text'] == "Title"
        assert len(data['sections']) == 1
        assert data['char_count'] == 5
        assert data['token_estimate'] == 2

    def test_ocr_result_from_dict(self):
        """Test creating OCRResult from dictionary."""
        data = {
            'file_path': '/test/image.jpg',
            'file_hash': 'abc123',
            'page_number': 0,
            'timestamp': '2024-12-30T00:00:00',
            'raw_text': 'Title',
            'sections': [
                {'type': 'heading', 'content': 'Title', 'level': 1, 'language': None}
            ],
            'char_count': 5,
            'token_estimate': 2
        }
        result = OCRResult.from_dict(data)
        assert result.file_path == '/test/image.jpg'
        assert result.file_hash == 'abc123'
        assert result.page_number == 0
        assert result.raw_text == 'Title'
        assert len(result.sections) == 1
        assert result.sections[0].type == SectionType.HEADING

    def test_ocr_result_roundtrip(self):
        """Test roundtrip conversion (to_dict and from_dict)."""
        sections = [
            OCRSection(type=SectionType.HEADING, content="Title", level=1),
            OCRSection(type=SectionType.PARAGRAPH, content="Content"),
        ]
        original = OCRResult(
            file_path="/test/image.jpg",
            file_hash="abc123",
            page_number=1,
            timestamp="2024-12-30T00:00:00",
            raw_text="Title\n\nContent",
            sections=sections,
            char_count=14,
            token_estimate=5
        )
        data = original.to_dict()
        restored = OCRResult.from_dict(data)
        assert restored.file_path == original.file_path
        assert restored.file_hash == original.file_hash
        assert restored.page_number == original.page_number
        assert restored.raw_text == original.raw_text
        assert len(restored.sections) == len(original.sections)
        assert restored.char_count == original.char_count
        assert restored.token_estimate == original.token_estimate

    def test_ocr_result_save_and_load(self, tmp_path):
        """Test saving and loading OCRResult to/from file."""
        sections = [
            OCRSection(type=SectionType.HEADING, content="Title", level=1),
        ]
        original = OCRResult(
            file_path="/test/image.jpg",
            file_hash="abc123",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Title",
            sections=sections,
            char_count=5,
            token_estimate=2
        )

        # Save
        save_path = tmp_path / "ocr_result.json"
        original.save(save_path)

        # Load
        loaded = OCRResult.load(save_path)

        assert loaded.file_path == original.file_path
        assert loaded.file_hash == original.file_hash
        assert loaded.page_number == original.page_number
        assert loaded.raw_text == original.raw_text
        assert len(loaded.sections) == len(original.sections)


class TestPDFCache:
    """Tests for PDFCache dataclass."""

    def test_pdf_cache_creation(self):
        """Test creating a PDFCache."""
        page_results = [
            OCRResult(
                file_path="/test/page_0.jpg",
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 1",
                sections=[],
                char_count=6,
                token_estimate=2
            ),
            OCRResult(
                file_path="/test/page_1.jpg",
                file_hash="hash2",
                page_number=1,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 2",
                sections=[],
                char_count=6,
                token_estimate=2
            ),
        ]
        cache = PDFCache(
            file_path="/test/document.pdf",
            file_hash="pdf_hash",
            timestamp="2024-12-30T00:00:00",
            page_results=page_results,
            full_text="Page 1\n\nPage 2"
        )
        assert cache.file_path == "/test/document.pdf"
        assert cache.file_hash == "pdf_hash"
        assert len(cache.page_results) == 2
        assert cache.full_text == "Page 1\n\nPage 2"

    def test_pdf_cache_to_dict(self):
        """Test converting PDFCache to dictionary."""
        page_results = [
            OCRResult(
                file_path="/test/page_0.jpg",
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 1",
                sections=[],
                char_count=6,
                token_estimate=2
            ),
        ]
        cache = PDFCache(
            file_path="/test/document.pdf",
            file_hash="pdf_hash",
            timestamp="2024-12-30T00:00:00",
            page_results=page_results,
            full_text="Page 1"
        )
        data = cache.to_dict()
        assert data['file_path'] == "/test/document.pdf"
        assert data['file_hash'] == "pdf_hash"
        assert len(data['page_results']) == 1
        assert data['full_text'] == "Page 1"

    def test_pdf_cache_from_dict(self):
        """Test creating PDFCache from dictionary."""
        data = {
            'file_path': '/test/document.pdf',
            'file_hash': 'pdf_hash',
            'timestamp': '2024-12-30T00:00:00',
            'page_results': [
                {
                    'file_path': '/test/page_0.jpg',
                    'file_hash': 'hash1',
                    'page_number': 0,
                    'timestamp': '2024-12-30T00:00:00',
                    'raw_text': 'Page 1',
                    'sections': [],
                    'char_count': 6,
                    'token_estimate': 2
                }
            ],
            'full_text': 'Page 1'
        }
        cache = PDFCache.from_dict(data)
        assert cache.file_path == '/test/document.pdf'
        assert cache.file_hash == 'pdf_hash'
        assert len(cache.page_results) == 1
        assert cache.full_text == 'Page 1'

    def test_pdf_cache_roundtrip(self):
        """Test roundtrip conversion (to_dict and from_dict)."""
        page_results = [
            OCRResult(
                file_path="/test/page_0.jpg",
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 1",
                sections=[],
                char_count=6,
                token_estimate=2
            ),
        ]
        original = PDFCache(
            file_path="/test/document.pdf",
            file_hash="pdf_hash",
            timestamp="2024-12-30T00:00:00",
            page_results=page_results,
            full_text="Page 1"
        )

        data = original.to_dict()
        restored = PDFCache.from_dict(data)

        assert restored.file_path == original.file_path
        assert restored.file_hash == original.file_hash
        assert len(restored.page_results) == len(original.page_results)
        assert restored.full_text == original.full_text

    def test_pdf_cache_save_and_load(self, tmp_path):
        """Test saving and loading PDFCache to/from file."""
        page_results = [
            OCRResult(
                file_path="/test/page_0.jpg",
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 1",
                sections=[],
                char_count=6,
                token_estimate=2
            ),
        ]
        original = PDFCache(
            file_path="/test/document.pdf",
            file_hash="pdf_hash",
            timestamp="2024-12-30T00:00:00",
            page_results=page_results,
            full_text="Page 1"
        )

        # Save
        save_path = tmp_path / "pdf_cache.json"
        original.save(save_path)

        # Load
        loaded = PDFCache.load(save_path)

        assert loaded.file_path == original.file_path
        assert loaded.file_hash == original.file_hash
        assert len(loaded.page_results) == len(original.page_results)
        assert loaded.full_text == original.full_text


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_hash_consistent(self, tmp_path):
        """Test that hash is consistent for the same file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash is 32 hex characters

    def test_compute_hash_different_files(self, tmp_path):
        """Test that different files have different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    def test_compute_hash_empty_file(self, tmp_path):
        """Test hash of empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        hash_value = compute_file_hash(empty_file)
        # MD5 of empty string is d41d8cd98f00b204e9800998ecf8427e
        assert hash_value == "d41d8cd98f00b204e9800998ecf8427e"

    def test_compute_hash_binary_file(self, tmp_path):
        """Test hash of binary file."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe\xfd')

        hash_value = compute_file_hash(binary_file)
        assert len(hash_value) == 32
        assert hash_value.isalnum() or any(c in hash_value for c in 'abcdef')


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_estimate_empty_string(self):
        """Test token estimate for empty string."""
        assert estimate_tokens("") == 1  # min(1, 0//3) = 1

    def test_estimate_short_text(self):
        """Test token estimate for short text."""
        # 30 chars / 3 = 10 tokens
        assert estimate_tokens("123456789012345678901234567890") == 10

    def test_estimate_chinese_text(self):
        """Test token estimate for Chinese text."""
        # Chinese text (15 chars) / 3 = 5
        chinese = "这是一段中文文本用于测试"
        assert estimate_tokens(chinese) == len(chinese) // 3

    def test_estimate_mixed_text(self):
        """Test token estimate for mixed text."""
        mixed = "Hello世界123测试"
        # 4 + 2 + 3 + 2 = 11 chars / 3 = 3
        assert estimate_tokens(mixed) == len(mixed) // 3

    def test_estimate_long_text(self):
        """Test token estimate for long text."""
        long_text = "a" * 300
        assert estimate_tokens(long_text) == 100
