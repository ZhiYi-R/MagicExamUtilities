"""
Tests for OCRCache manager.
"""

import json
import time
import pytest
from pathlib import Path

from utilities.cache import OCRCache
from utilities.models import OCRResult, OCRSection, SectionType, PDFCache


class TestOCRCache:
    """Tests for OCRCache class."""

    def test_cache_initialization(self, tmp_path):
        """Test OCRCache initialization creates cache directory."""
        cache_dir = tmp_path / "cache"
        cache = OCRCache(cache_dir=cache_dir)

        assert cache._cache_dir == cache_dir
        assert cache._ocr_cache_dir == cache_dir / "ocr"
        assert cache._ocr_cache_dir.exists()

    def test_get_pdf_cache_no_cache(self, tmp_path):
        """Test get_pdf_cache when cache doesn't exist."""
        cache = OCRCache(cache_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake pdf content")

        result = cache.get_pdf_cache(pdf_path)
        assert result is None

    def test_save_and_get_pdf_cache(self, tmp_path):
        """Test saving and retrieving PDF cache."""
        cache = OCRCache(cache_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("fake pdf content")

        # Create page results
        page_results = [
            OCRResult(
                file_path=str(tmp_path / "page_0.jpg"),
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 1 content",
                sections=[],
                char_count=15,
                token_estimate=5
            ),
            OCRResult(
                file_path=str(tmp_path / "page_1.jpg"),
                file_hash="hash2",
                page_number=1,
                timestamp="2024-12-30T00:00:00",
                raw_text="Page 2 content",
                sections=[],
                char_count=15,
                token_estimate=5
            ),
        ]
        full_text = "Page 1 content\n\nPage 2 content"

        # Save cache
        saved_cache = cache.save_pdf_cache(pdf_path, page_results, full_text)

        assert saved_cache.file_path == str(pdf_path)
        assert saved_cache.full_text == full_text
        assert len(saved_cache.page_results) == 2

        # Verify cache file exists
        from utilities.models import compute_file_hash
        pdf_hash = compute_file_hash(pdf_path)
        cache_file = tmp_path / "ocr" / f"{pdf_hash}.json"
        assert cache_file.exists()

        # Retrieve cache
        retrieved_cache = cache.get_pdf_cache(pdf_path)
        assert retrieved_cache is not None
        assert retrieved_cache.file_path == str(pdf_path)
        assert retrieved_cache.full_text == full_text
        assert len(retrieved_cache.page_results) == 2
        assert retrieved_cache.page_results[0].raw_text == "Page 1 content"
        assert retrieved_cache.page_results[1].raw_text == "Page 2 content"

    def test_get_pdf_cache_outdated(self, tmp_path):
        """Test get_pdf_cache returns None when cache is older than PDF."""
        cache = OCRCache(cache_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Create old cache
        page_results = [
            OCRResult(
                file_path=str(tmp_path / "page_0.jpg"),
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Old content",
                sections=[],
                char_count=11,
                token_estimate=4
            ),
        ]
        pdf_path.write_text("initial content")
        cache.save_pdf_cache(pdf_path, page_results, "Old content")

        # Wait a bit and modify the PDF
        time.sleep(0.01)
        pdf_path.write_text("newer content")

        # Cache should be considered outdated
        result = cache.get_pdf_cache(pdf_path)
        assert result is None

    def test_get_pdf_cache_valid(self, tmp_path):
        """Test get_pdf_cache returns cache when it's newer than PDF."""
        cache = OCRCache(cache_dir=tmp_path)
        pdf_path = tmp_path / "test.pdf"

        # Create PDF first
        pdf_path.write_text("content")

        # Wait a bit then create cache
        time.sleep(0.01)
        page_results = [
            OCRResult(
                file_path=str(tmp_path / "page_0.jpg"),
                file_hash="hash1",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Cached content",
                sections=[],
                char_count=14,
                token_estimate=5
            ),
        ]
        cache.save_pdf_cache(pdf_path, page_results, "Cached content")

        # Cache should be valid
        result = cache.get_pdf_cache(pdf_path)
        assert result is not None
        assert result.full_text == "Cached content"

    def test_save_page_result(self, tmp_path):
        """Test saving a single page result."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake image")

        ocr_result = OCRResult(
            file_path=str(image_path),
            file_hash="abc123",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="OCR text",
            sections=[],
            char_count=8,
            token_estimate=3
        )

        cache.save_page_result(image_path, ocr_result)

        # Verify cache file exists
        cache_file = tmp_path / "ocr" / "abc123.json"
        assert cache_file.exists()

        # Verify content
        loaded = OCRResult.load(cache_file)
        assert loaded.raw_text == "OCR text"
        assert loaded.file_hash == "abc123"

    def test_get_page_result(self, tmp_path):
        """Test retrieving a single page result."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake image")

        # Save a page result
        ocr_result = OCRResult(
            file_path=str(image_path),
            file_hash="xyz789",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Page text",
            sections=[],
            char_count=9,
            token_estimate=3
        )
        cache.save_page_result(image_path, ocr_result)

        # Retrieve it
        retrieved = cache.get_page_result(image_path, "xyz789")
        assert retrieved is not None
        assert retrieved.raw_text == "Page text"
        assert retrieved.file_hash == "xyz789"

    def test_get_page_result_no_cache(self, tmp_path):
        """Test get_page_result when cache doesn't exist."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake image")

        result = cache.get_page_result(image_path, "nonexistent")
        assert result is None

    def test_get_page_result_outdated(self, tmp_path):
        """Test get_page_result returns None when cache is older than image."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"

        # Create old cache
        image_path.write_bytes(b"old content")
        ocr_result = OCRResult(
            file_path=str(image_path),
            file_hash="old_hash",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Old OCR",
            sections=[],
            char_count=7,
            token_estimate=2
        )
        cache.save_page_result(image_path, ocr_result)

        # Wait and modify image
        time.sleep(0.01)
        image_path.write_bytes(b"new content")

        # Recompute hash would give different result, but let's test with old hash
        result = cache.get_page_result(image_path, "old_hash")
        # Cache should be outdated since image mtime > cache mtime
        assert result is None

    def test_get_page_result_valid_cache(self, tmp_path):
        """Test get_page_result returns valid cache."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"

        # Create image first
        image_path.write_bytes(b"image content")

        # Wait then create cache
        time.sleep(0.01)
        ocr_result = OCRResult(
            file_path=str(image_path),
            file_hash="valid_hash",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Valid OCR",
            sections=[],
            char_count=10,
            token_estimate=3
        )
        cache.save_page_result(image_path, ocr_result)

        # Cache should be valid
        result = cache.get_page_result(image_path, "valid_hash")
        assert result is not None
        assert result.raw_text == "Valid OCR"

    def test_clear_cache_specific_pdf(self, tmp_path):
        """Test clearing cache for a specific PDF."""
        cache = OCRCache(cache_dir=tmp_path)

        # Create caches for two PDFs
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_text("content1")
        pdf2.write_text("content2")

        page_results = [
            OCRResult(
                file_path=str(tmp_path / "page.jpg"),
                file_hash="hash",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Text",
                sections=[],
                char_count=4,
                token_estimate=1
            ),
        ]

        cache.save_pdf_cache(pdf1, page_results, "Text1")
        cache.save_pdf_cache(pdf2, page_results, "Text2")

        # Clear cache for pdf1
        cache.clear_cache(pdf1)

        # pdf1 cache should be gone
        assert cache.get_pdf_cache(pdf1) is None

        # pdf2 cache should still exist
        assert cache.get_pdf_cache(pdf2) is not None

    def test_clear_cache_all(self, tmp_path):
        """Test clearing all cache."""
        cache = OCRCache(cache_dir=tmp_path)

        # Create caches for multiple PDFs
        for i in range(3):
            pdf = tmp_path / f"doc{i}.pdf"
            pdf.write_text(f"content{i}")
            page_results = [
                OCRResult(
                    file_path=str(tmp_path / "page.jpg"),
                    file_hash=f"hash{i}",
                    page_number=0,
                    timestamp="2024-12-30T00:00:00",
                    raw_text="Text",
                    sections=[],
                    char_count=4,
                    token_estimate=1
                ),
            ]
            cache.save_pdf_cache(pdf, page_results, f"Text{i}")

        # Clear all caches
        cache.clear_cache()

        # All caches should be gone
        for i in range(3):
            pdf = tmp_path / f"doc{i}.pdf"
            assert cache.get_pdf_cache(pdf) is None

    def test_clear_cache_no_files(self, tmp_path):
        """Test clear_cache when cache directory is empty."""
        cache = OCRCache(cache_dir=tmp_path)
        # Should not raise any errors
        cache.clear_cache()

    def test_multiple_pdfs_different_hashes(self, tmp_path):
        """Test that different PDFs get different cache files."""
        cache = OCRCache(cache_dir=tmp_path)

        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_text("content A")
        pdf2.write_text("content B")

        page_results = [
            OCRResult(
                file_path=str(tmp_path / "page.jpg"),
                file_hash="hash",
                page_number=0,
                timestamp="2024-12-30T00:00:00",
                raw_text="Text",
                sections=[],
                char_count=4,
                token_estimate=1
            ),
        ]

        cache.save_pdf_cache(pdf1, page_results, "Text1")
        cache.save_pdf_cache(pdf2, page_results, "Text2")

        # Get cache files
        cache_files = list((tmp_path / "ocr").glob("*.json"))
        assert len(cache_files) == 2

        # Each PDF should get its own cached result
        assert cache.get_pdf_cache(pdf1).full_text == "Text1"
        assert cache.get_pdf_cache(pdf2).full_text == "Text2"
