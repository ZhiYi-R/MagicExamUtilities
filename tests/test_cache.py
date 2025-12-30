"""
Tests for OCRCache and STTCache managers.
"""

import json
import time
import pytest
from pathlib import Path

from utilities.cache import OCRCache, STTCache
from utilities.models import OCRResult, OCRSection, SectionType, PDFCache, AudioCache, compute_file_hash


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
                page_hash="hash1",
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
                page_hash="hash2",
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

        # Verify cache file exists (new structure: ocr/stored/{pdf_hash}/cache.json)
        from utilities.models import compute_file_hash
        pdf_hash = compute_file_hash(pdf_path)
        cache_dir = tmp_path / "ocr" / "stored" / pdf_hash
        cache_file = cache_dir / "cache.json"
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
                page_hash="hash1",
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
                page_hash="hash1",
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
                page_hash="abc123",
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
                page_hash="xyz789",
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

    def test_get_page_result_hash_based(self, tmp_path):
        """Test get_page_result uses hash-based validation."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"

        # Create image with initial content
        image_path.write_bytes(b"image content")

        # Create and save OCR result
        ocr_result = OCRResult(
            file_path=str(image_path),
            file_hash="content_hash",
                page_hash="content_hash",
            page_number=0,
            timestamp="2024-12-30T00:00:00",
            raw_text="Valid OCR",
            sections=[],
            char_count=10,
            token_estimate=3
        )
        cache.save_page_result(image_path, ocr_result)

        # Cache should be valid when hash matches
        result = cache.get_page_result(image_path, "content_hash")
        assert result is not None
        assert result.raw_text == "Valid OCR"

        # Cache should return None when hash doesn't match
        result = cache.get_page_result(image_path, "different_hash")
        assert result is None

    def test_get_page_result_handles_corrupted_cache(self, tmp_path):
        """Test get_page_result handles corrupted cache file gracefully."""
        cache = OCRCache(cache_dir=tmp_path)
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake image")

        # Create corrupted cache file
        cache_path = tmp_path / 'ocr' / 'test_hash.json'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("invalid json content")

        result = cache.get_page_result(image_path, "test_hash")
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
                page_hash="valid_hash",
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
                page_hash="hash",
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
                    page_hash=f"hash{i}",
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
                page_hash="hash",
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

        # Get cache directories (new structure: ocr/stored/{pdf_hash}/)
        cache_dirs = list((tmp_path / "ocr" / "stored").glob("*"))
        cache_dirs = [d for d in cache_dirs if d.is_dir()]
        assert len(cache_dirs) == 2

        # Each PDF should get its own cached result
        assert cache.get_pdf_cache(pdf1).full_text == "Text1"
        assert cache.get_pdf_cache(pdf2).full_text == "Text2"


class TestSTTCache:
    """Tests for STTCache class."""

    def test_cache_initialization(self, tmp_path):
        """Test STTCache initialization creates cache directory."""
        cache_dir = tmp_path / "cache"
        cache = STTCache(cache_dir=cache_dir)

        assert cache._cache_dir == cache_dir
        assert cache._stt_cache_dir == cache_dir / "stt"
        assert cache._stt_cache_dir.exists()

    def test_get_audio_cache_no_cache(self, tmp_path):
        """Test get_audio_cache when cache doesn't exist."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        result = cache.get_audio_cache(audio_path)
        assert result is None

    def test_save_and_get_audio_cache(self, tmp_path):
        """Test saving and retrieving audio cache."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio content")

        transcribed_text = "This is a transcribed text from the audio."

        # Save cache
        saved_cache = cache.save_audio_cache(audio_path, transcribed_text)

        assert saved_cache.file_path == str(audio_path)
        assert saved_cache.raw_text == transcribed_text
        assert saved_cache.char_count == len(transcribed_text)

        # Verify cache file exists (stt/{audio_hash}/cache.json)
        audio_hash = compute_file_hash(audio_path)
        cache_dir = tmp_path / "stt" / audio_hash
        cache_file = cache_dir / "cache.json"
        assert cache_file.exists()

        # Retrieve cache
        retrieved_cache = cache.get_audio_cache(audio_path)
        assert retrieved_cache is not None
        assert retrieved_cache.file_path == str(audio_path)
        assert retrieved_cache.raw_text == transcribed_text

    def test_get_audio_cache_hash_mismatch(self, tmp_path):
        """Test get_audio_cache returns None when hash doesn't match."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"

        # Create initial audio and cache
        audio_path.write_bytes(b"initial content")
        cache.save_audio_cache(audio_path, "Initial text")

        # Modify the audio file (hash will change)
        audio_path.write_bytes(b"modified content")

        # Cache should be invalid due to hash mismatch
        result = cache.get_audio_cache(audio_path)
        assert result is None

    def test_save_audio_cache_creates_metadata(self, tmp_path):
        """Test save_audio_cache creates proper metadata."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"audio content")

        text = "Transcribed text with enough content for token estimation."

        saved_cache = cache.save_audio_cache(audio_path, text)

        # Verify metadata
        assert saved_cache.char_count == len(text)
        assert saved_cache.token_estimate > 0
        assert saved_cache.file_hash == compute_file_hash(audio_path)

    def test_get_audio_cache_dir(self, tmp_path):
        """Test get_audio_cache_dir returns correct path."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"content")

        audio_hash = compute_file_hash(audio_path)
        expected_dir = tmp_path / "stt" / audio_hash

        assert cache.get_audio_cache_dir(audio_path) == expected_dir

    def test_clear_cache_specific_audio(self, tmp_path):
        """Test clearing cache for a specific audio file."""
        cache = STTCache(cache_dir=tmp_path)

        # Create caches for two audio files
        audio1 = tmp_path / "audio1.mp3"
        audio2 = tmp_path / "audio2.mp3"
        audio1.write_bytes(b"content1")
        audio2.write_bytes(b"content2")

        cache.save_audio_cache(audio1, "Text1")
        cache.save_audio_cache(audio2, "Text2")

        # Clear cache for audio1
        cache.clear_cache(audio1)

        # audio1 cache should be gone
        assert cache.get_audio_cache(audio1) is None

        # audio2 cache should still exist
        assert cache.get_audio_cache(audio2) is not None

    def test_clear_cache_all(self, tmp_path):
        """Test clearing all STT cache."""
        cache = STTCache(cache_dir=tmp_path)

        # Create caches for multiple audio files
        for i in range(3):
            audio = tmp_path / f"audio{i}.mp3"
            audio.write_bytes(f"content{i}".encode())
            cache.save_audio_cache(audio, f"Text{i}")

        # Clear all caches
        cache.clear_cache()

        # All caches should be gone
        for i in range(3):
            audio = tmp_path / f"audio{i}.mp3"
            assert cache.get_audio_cache(audio) is None

    def test_clear_cache_no_files(self, tmp_path):
        """Test clear_cache when cache directory is empty."""
        cache = STTCache(cache_dir=tmp_path)
        # Should not raise any errors
        cache.clear_cache()

    def test_multiple_audios_different_hashes(self, tmp_path):
        """Test that different audio files get different cache directories."""
        cache = STTCache(cache_dir=tmp_path)

        audio1 = tmp_path / "audio1.mp3"
        audio2 = tmp_path / "audio2.mp3"
        audio1.write_bytes(b"content A")
        audio2.write_bytes(b"content B")

        cache.save_audio_cache(audio1, "Text1")
        cache.save_audio_cache(audio2, "Text2")

        # Get cache directories
        cache_dirs = list((tmp_path / "stt").glob("*"))
        cache_dirs = [d for d in cache_dirs if d.is_dir()]
        assert len(cache_dirs) == 2

        # Each audio should get its own cached result
        assert cache.get_audio_cache(audio1).raw_text == "Text1"
        assert cache.get_audio_cache(audio2).raw_text == "Text2"

    def test_get_audio_cache_handles_corrupted_cache(self, tmp_path):
        """Test get_audio_cache handles corrupted cache file gracefully."""
        cache = STTCache(cache_dir=tmp_path)
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio")

        # Create corrupted cache file
        from utilities.models import compute_file_hash
        audio_hash = compute_file_hash(audio_path)
        cache_path = tmp_path / 'stt' / audio_hash / 'cache.json'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("invalid json content")

        result = cache.get_audio_cache(audio_path)
        assert result is None


class TestAudioCacheModel:
    """Tests for AudioCache data model."""

    def test_audio_cache_serialization(self, tmp_path):
        """Test AudioCache save and load."""
        cache_path = tmp_path / "cache.json"

        original = AudioCache(
            file_path="/path/to/audio.mp3",
            file_hash="abc123",
            timestamp="2024-12-30T00:00:00",
            raw_text="Transcribed text",
            char_count=17,
            token_estimate=6
        )

        # Save
        original.save(cache_path)

        # Load
        loaded = AudioCache.load(cache_path)

        assert loaded.file_path == original.file_path
        assert loaded.file_hash == original.file_hash
        assert loaded.raw_text == original.raw_text
        assert loaded.char_count == original.char_count
        assert loaded.token_estimate == original.token_estimate

    def test_audio_cache_to_dict(self):
        """Test AudioCache to_dict conversion."""
        cache = AudioCache(
            file_path="/path/to/audio.mp3",
            file_hash="abc123",
            timestamp="2024-12-30T00:00:00",
            raw_text="Text",
            char_count=4,
            token_estimate=1
        )

        data = cache.to_dict()

        assert data["file_path"] == "/path/to/audio.mp3"
        assert data["file_hash"] == "abc123"
        assert data["raw_text"] == "Text"
        assert data["char_count"] == 4

    def test_audio_cache_from_dict(self):
        """Test AudioCache from_dict creation."""
        data = {
            "file_path": "/path/to/audio.mp3",
            "file_hash": "xyz789",
            "timestamp": "2024-12-30T00:00:00",
            "raw_text": "More text",
            "char_count": 9,
            "token_estimate": 3
        }

        cache = AudioCache.from_dict(data)

        assert cache.file_path == "/path/to/audio.mp3"
        assert cache.file_hash == "xyz789"
        assert cache.raw_text == "More text"
        assert cache.char_count == 9
