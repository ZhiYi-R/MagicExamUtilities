"""
Unit tests for retrieval module.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from utilities.retrieval.cache_loader import CacheLoader
from utilities.retrieval.tools import (
    _get_retriever, _get_kb_manager,
    search_cache, get_sections_by_type, list_cached_documents,
    list_knowledge_bases, get_available_pdfs, get_all_tools
)
from utilities.models import OCRResult, PDFCache, OCRSection, SectionType


@pytest.fixture
def mock_cache_dir(temp_dir):
    """Create a mock cache directory with sample data."""
    ocr_dir = temp_dir / 'ocr' / 'stored'
    ocr_dir.mkdir(parents=True)

    # Create a sample PDF cache file (new structure: ocr/stored/{pdf_hash}/cache.json)
    pdf_hash_dir = ocr_dir / 'abc123'
    pdf_hash_dir.mkdir(parents=True)

    cache_data = {
        'file_path': '/path/to/test.pdf',
        'file_hash': 'abc123',
        'timestamp': '1234567890.0',
        'page_results': [
            {
                'file_path': '/path/to/test.pdf',
                'file_hash': 'abc123',
                'page_number': 0,
                'timestamp': '1234567890.0',
                'raw_text': 'Sample content for testing',
                'sections': [],
                'char_count': 25,
                'token_estimate': 10
            }
        ],
        'full_text': 'Sample content for testing'
    }

    cache_file = pdf_hash_dir / 'cache.json'
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f)

    # Create another cache file without page_results (single page OCR)
    # Use legacy structure for this one
    ocr_root_dir = temp_dir / 'ocr'
    page_data = {
        'file_path': '/path/to/page.jpg',
        'file_hash': 'def456',
        'page_number': 0,
        'timestamp': '1234567890.0',
        'raw_text': 'Single page OCR result',
        'sections': [],
        'char_count': 22,
        'token_estimate': 8
    }

    page_file = ocr_root_dir / 'page.json'
    with open(page_file, 'w', encoding='utf-8') as f:
        json.dump(page_data, f)

    # Also create a legacy cache for test.pdf (to support stem-based lookup)
    test_legacy_data = {
        'file_path': '/path/to/test.pdf',
        'file_hash': 'abc123',
        'timestamp': '1234567890.0',
        'page_results': [
            {
                'file_path': '/path/to/test.pdf',
                'file_hash': 'abc123',
                'page_number': 0,
                'timestamp': '1234567890.0',
                'raw_text': 'Sample content for testing',
                'sections': [],
                'char_count': 25,
                'token_estimate': 10
            }
        ],
        'full_text': 'Sample content for testing'
    }

    test_legacy_file = ocr_root_dir / 'test.json'
    with open(test_legacy_file, 'w', encoding='utf-8') as f:
        json.dump(test_legacy_data, f)

    # Create an invalid cache file
    invalid_file = ocr_root_dir / 'invalid.json'
    invalid_file.write_text('invalid json content')

    return temp_dir


@pytest.fixture
def cache_loader(mock_cache_dir):
    """Create a CacheLoader instance with mock cache directory."""
    return CacheLoader(mock_cache_dir)


@pytest.mark.unit
class TestCacheLoader:
    """Test cases for CacheLoader class."""

    def test_cache_loader_initialization(self, cache_loader, mock_cache_dir):
        """Test CacheLoader initialization."""
        assert cache_loader._cache_dir == mock_cache_dir
        assert cache_loader._ocr_cache_dir == mock_cache_dir / 'ocr'

    def test_list_cached_pdfs(self, cache_loader):
        """Test listing cached PDFs."""
        pdfs = cache_loader.list_cached_pdfs()
        assert len(pdfs) == 1
        assert 'test.pdf' in pdfs

    def test_list_cached_pdfs_nonexistent_cache(self, temp_dir):
        """Test listing cached PDFs when cache dir doesn't exist."""
        loader = CacheLoader(temp_dir)
        pdfs = loader.list_cached_pdfs()
        assert pdfs == []

    def test_load_pdf_cache_exists(self, cache_loader):
        """Test loading an existing PDF cache."""
        cache = cache_loader.load_pdf_cache(Path('test.pdf'))
        assert cache is not None
        assert cache.file_path == '/path/to/test.pdf'

    def test_load_pdf_cache_not_exists(self, cache_loader):
        """Test loading a non-existent PDF cache."""
        cache = cache_loader.load_pdf_cache(Path('nonexistent.pdf'))
        assert cache is None

    def test_load_all_caches(self, cache_loader):
        """Test loading all caches."""
        caches = cache_loader.load_all_caches()
        assert len(caches) == 1
        assert caches[0].file_path == '/path/to/test.pdf'

    def test_load_all_caches_nonexistent_dir(self, temp_dir):
        """Test loading all caches when cache dir doesn't exist."""
        loader = CacheLoader(temp_dir)
        caches = loader.load_all_caches()
        assert caches == []

    def test_get_all_text_with_caches(self, cache_loader):
        """Test getting all text from provided caches."""
        caches = cache_loader.load_all_caches()
        text = cache_loader.get_all_text(caches)
        assert 'test.pdf' in text
        assert 'Sample content for testing' in text

    def test_get_all_text_without_caches(self, cache_loader):
        """Test getting all text by loading all caches."""
        text = cache_loader.get_all_text()
        assert 'test.pdf' in text

    def test_get_all_text_empty_caches(self, cache_loader):
        """Test getting all text with empty cache list."""
        text = cache_loader.get_all_text([])
        assert text == ''

    def test_get_sections_by_type(self, cache_loader, mock_cache_dir):
        """Test getting sections by type."""
        # Create a cache with sections (new structure: ocr/stored/{pdf_hash}/cache.json)
        ocr_stored_dir = mock_cache_dir / 'ocr' / 'stored'
        pdf_hash_dir = ocr_stored_dir / 'xyz789'
        pdf_hash_dir.mkdir(parents=True)

        cache_data = {
            'file_path': '/path/to/sections.pdf',
            'file_hash': 'xyz789',
            'timestamp': '1234567890.0',
            'page_results': [
                {
                    'file_path': '/path/to/sections.pdf',
                    'file_hash': 'xyz789',
                    'page_number': 0,
                    'timestamp': '1234567890.0',
                    'raw_text': 'Content with sections',
                    'sections': [
                        {
                            'type': 'heading',
                            'content': 'Test Heading',
                            'level': None,
                            'language': None
                        },
                        {
                            'type': 'code',
                            'content': 'def test(): pass',
                            'level': None,
                            'language': None
                        }
                    ],
                    'char_count': 20,
                    'token_estimate': 8
                }
            ],
            'full_text': 'Content with sections'
        }

        cache_file = pdf_hash_dir / 'cache.json'
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)

        loader = CacheLoader(mock_cache_dir)
        code_sections = loader.get_sections_by_type(SectionType.CODE)
        assert len(code_sections) == 1
        assert 'def test(): pass' in code_sections[0]

        heading_sections = loader.get_sections_by_type(SectionType.HEADING)
        assert len(heading_sections) == 1
        assert 'Test Heading' in heading_sections[0]

    def test_search_keyword_case_insensitive(self, cache_loader):
        """Test keyword search case insensitive."""
        results = cache_loader.search_keyword('SAMPLE')
        assert len(results) == 1
        assert 'test.pdf' in results[0]

    def test_search_keyword_case_sensitive(self, cache_loader):
        """Test keyword search case sensitive."""
        results = cache_loader.search_keyword('Sample', case_sensitive=True)
        assert len(results) == 1
        assert 'test.pdf' in results[0]

        results = cache_loader.search_keyword('SAMPLE', case_sensitive=True)
        assert len(results) == 0

    def test_search_keyword_with_context(self, cache_loader, mock_cache_dir):
        """Test keyword search returns context around match."""
        # Create a cache with multiple lines
        ocr_dir = mock_cache_dir / 'ocr'
        cache_data = {
            'file_path': '/path/to/context.pdf',
            'file_hash': 'ctx123',
            'timestamp': '1234567890.0',
            'page_results': [
                {
                    'file_path': '/path/to/context.pdf',
                    'file_hash': 'ctx123',
                    'page_number': 0,
                    'timestamp': '1234567890.0',
                    'raw_text': 'Line 1\nLine 2\nKeyword here\nLine 4\nLine 5',
                    'sections': [],
                    'char_count': 50,
                    'token_estimate': 20
                }
            ],
            'full_text': 'Line 1\nLine 2\nKeyword here\nLine 4\nLine 5'
        }

        cache_file = ocr_dir / 'context.json'
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f)

        loader = CacheLoader(mock_cache_dir)
        results = loader.search_keyword('Keyword')
        assert len(results) == 1
        # Should include context lines
        assert 'Line 2' in results[0] or 'Line 4' in results[0]


@pytest.mark.unit
class TestRetrievalTools:
    """Test cases for retrieval tools."""

    @patch('utilities.retrieval.tools.CacheRetriever')
    @patch('utilities.retrieval.tools.KnowledgeBaseManager')
    def test_get_retriever_initializes_on_first_call(self, mock_kb_manager, mock_retriever, temp_dir):
        """Test that _get_retriever initializes on first call."""
        mock_kb_instance = Mock()
        mock_retriever_instance = Mock()

        mock_kb_manager.return_value = mock_kb_instance
        mock_retriever.return_value = mock_retriever_instance

        # Set environment variable
        os.environ['CACHE_DIR'] = str(temp_dir)

        # Reset global variables
        import utilities.retrieval.tools as tools_module
        tools_module._retriever = None
        tools_module._kb_manager = None

        retriever = _get_retriever()

        assert retriever == mock_retriever_instance
        mock_kb_manager.assert_called_once()
        mock_retriever.assert_called_once()
        mock_retriever_instance.load_or_build_index.assert_called_once()

        # Cleanup
        del os.environ['CACHE_DIR']

    @patch('utilities.retrieval.tools.KnowledgeBaseManager')
    def test_get_kb_manager_initializes_on_first_call(self, mock_kb_manager, temp_dir):
        """Test that _get_kb_manager initializes on first call."""
        mock_kb_instance = Mock()
        mock_kb_manager.return_value = mock_kb_instance

        # Set environment variable
        os.environ['CACHE_DIR'] = str(temp_dir)

        # Reset global variables
        import utilities.retrieval.tools as tools_module
        tools_module._kb_manager = None

        manager = _get_kb_manager()

        assert manager == mock_kb_instance
        mock_kb_manager.assert_called_once()

        # Cleanup
        del os.environ['CACHE_DIR']

    def test_get_all_tools(self):
        """Test get_all_tools returns all tools."""
        tools = get_all_tools()
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert 'search_cache' in tool_names
        assert 'get_sections_by_type' in tool_names
        assert 'list_cached_documents' in tool_names
        assert 'list_knowledge_bases' in tool_names
        assert 'get_available_pdfs' in tool_names

    @patch('utilities.retrieval.tools._get_retriever')
    def test_search_cache_with_results(self, mock_get_retriever):
        """Test search_cache returns results."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = ['Result 1', 'Result 2']
        mock_get_retriever.return_value = mock_retriever

        result = search_cache.invoke({'query': 'test query'})

        assert 'Result 1' in result
        assert 'Result 2' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_search_cache_no_results(self, mock_get_retriever):
        """Test search_cache returns message when no results."""
        mock_retriever = Mock()
        mock_retriever.search.return_value = []
        mock_get_retriever.return_value = mock_retriever

        result = search_cache.invoke({'query': 'test query'})

        assert '未找到相关内容' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_search_cache_with_kb_id_error(self, mock_get_retriever):
        """Test search_cache handles knowledge base errors."""
        mock_retriever = Mock()
        mock_retriever.search_in_kb.side_effect = ValueError('Invalid KB ID')
        mock_get_retriever.return_value = mock_retriever

        result = search_cache.invoke({'query': 'test query', 'kb_id': 'invalid'})

        assert 'Invalid KB ID' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_get_sections_by_type_with_results(self, mock_get_retriever):
        """Test get_sections_by_type returns sections."""
        mock_retriever = Mock()
        mock_retriever.get_sections_by_type.return_value = ['Section 1', 'Section 2']
        mock_get_retriever.return_value = mock_retriever

        result = get_sections_by_type.invoke({'section_type': 'code'})

        assert 'Section 1' in result
        assert 'Section 2' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_get_sections_by_type_no_results(self, mock_get_retriever):
        """Test get_sections_by_type returns message when no sections."""
        mock_retriever = Mock()
        mock_retriever.get_sections_by_type.return_value = []
        mock_get_retriever.return_value = mock_retriever

        result = get_sections_by_type.invoke({'section_type': 'code'})

        assert '未找到类型' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_get_sections_by_type_invalid_type(self, mock_get_retriever):
        """Test get_sections_by_type handles invalid type error."""
        mock_retriever = Mock()
        mock_retriever.get_sections_by_type.side_effect = ValueError('Invalid section type')
        mock_get_retriever.return_value = mock_retriever

        result = get_sections_by_type.invoke({'section_type': 'invalid'})

        assert 'Invalid section type' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_list_cached_documents_with_results(self, mock_get_retriever):
        """Test list_cached_documents returns documents."""
        mock_retriever = Mock()
        mock_retriever.list_cached_documents.return_value = ['doc1.pdf', 'doc2.pdf']
        mock_get_retriever.return_value = mock_retriever

        result = list_cached_documents.invoke({})

        assert 'doc1.pdf' in result
        assert 'doc2.pdf' in result

    @patch('utilities.retrieval.tools._get_retriever')
    def test_list_cached_documents_no_results(self, mock_get_retriever):
        """Test list_cached_documents returns message when no documents."""
        mock_retriever = Mock()
        mock_retriever.list_cached_documents.return_value = []
        mock_get_retriever.return_value = mock_retriever

        result = list_cached_documents.invoke({})

        assert '目前没有已缓存的文档' in result

    @patch('utilities.retrieval.tools._get_kb_manager')
    def test_list_knowledge_bases_with_results(self, mock_get_kb_manager):
        """Test list_knowledge_bases returns knowledge bases."""
        mock_kb = Mock()
        mock_kb.id = 'kb1'
        mock_kb.name = 'Test KB'
        mock_kb.description = 'A test knowledge base'
        mock_kb.pdf_files = []

        mock_manager = Mock()
        mock_manager.list_knowledge_bases.return_value = [mock_kb]
        mock_get_kb_manager.return_value = mock_manager

        result = list_knowledge_bases.invoke({})

        assert 'kb1' in result
        assert 'Test KB' in result

    @patch('utilities.retrieval.tools._get_kb_manager')
    def test_list_knowledge_bases_no_results(self, mock_get_kb_manager):
        """Test list_knowledge_bases returns message when no KBs."""
        mock_manager = Mock()
        mock_manager.list_knowledge_bases.return_value = []
        mock_get_kb_manager.return_value = mock_manager

        result = list_knowledge_bases.invoke({})

        assert '目前没有创建任何知识库' in result

    @patch('utilities.retrieval.tools._get_kb_manager')
    def test_get_available_pdfs_with_results(self, mock_get_kb_manager):
        """Test get_available_pdfs returns PDFs."""
        mock_manager = Mock()
        mock_manager.get_available_pdfs.return_value = ['doc1.pdf', 'doc2.pdf']
        mock_get_kb_manager.return_value = mock_manager

        result = get_available_pdfs.invoke({})

        assert 'doc1.pdf' in result
        assert 'doc2.pdf' in result

    @patch('utilities.retrieval.tools._get_kb_manager')
    def test_get_available_pdfs_no_results(self, mock_get_kb_manager):
        """Test get_available_pdfs returns message when no PDFs."""
        mock_manager = Mock()
        mock_manager.get_available_pdfs.return_value = []
        mock_get_kb_manager.return_value = mock_manager

        result = get_available_pdfs.invoke({})

        assert '目前没有已缓存的 PDF 文档' in result
