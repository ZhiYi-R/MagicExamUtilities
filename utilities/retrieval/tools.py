"""
LangChain tools for cache retrieval.

Defines tools that can be used by LLMs for function calling.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from .retriever import CacheRetriever
from ..knowledge_base.manager import KnowledgeBaseManager

# Global retriever instance
_retriever: Optional[CacheRetriever] = None
_kb_manager: Optional[KnowledgeBaseManager] = None


def _get_retriever() -> CacheRetriever:
    """Get or create the global retriever instance."""
    global _retriever, _kb_manager
    if _retriever is None:
        cache_dir = Path(os.environ.get('CACHE_DIR', './cache/'))
        _kb_manager = KnowledgeBaseManager(cache_dir)
        _retriever = CacheRetriever(cache_dir, kb_manager=_kb_manager)
        _retriever.load_or_build_index()
    return _retriever


def _get_kb_manager() -> KnowledgeBaseManager:
    """Get or create the global knowledge base manager."""
    global _kb_manager
    if _kb_manager is None:
        cache_dir = Path(os.environ.get('CACHE_DIR', './cache/'))
        _kb_manager = KnowledgeBaseManager(cache_dir)
    return _kb_manager


@tool
def search_cache(query: str, method: str = "hybrid", limit: int = 5, kb_id: Optional[str] = None) -> str:
    """
    搜索已缓存的 PDF 内容。

    支持关键词搜索、语义搜索或混合搜索。
    可以指定知识库 ID 进行限定搜索。

    Args:
        query: 搜索关键词或问题
        method: 搜索方式，可选值：keyword（关键词）、semantic（语义）、hybrid（混合）
        limit: 返回结果数量，默认 5 条
        kb_id: 知识库 ID，为空时搜索所有缓存

    Returns:
        匹配的内容片段，包含来源信息
    """
    retriever = _get_retriever()

    if kb_id:
        try:
            results = retriever.search_in_kb(query, kb_id=kb_id, method=method, limit=limit)
        except ValueError as e:
            return str(e)
    else:
        results = retriever.search(query, method=method, limit=limit)

    if not results:
        return "未找到相关内容。请确保已通过 PDF 或音频处理生成缓存。"

    return "\n\n".join(results)


@tool
def get_sections_by_type(section_type: str, limit: int = 10) -> str:
    """
    按类型获取缓存中的内容段落。

    支持的类型：table（表格）、formula（公式）、code（代码）、heading（标题）、paragraph（段落）、list（列表）

    Args:
        section_type: 内容类型
        limit: 返回结果数量，默认 10 条

    Returns:
        匹配的段落内容列表
    """
    retriever = _get_retriever()

    try:
        sections = retriever.get_sections_by_type(section_type, limit=limit)

        if not sections:
            return f"未找到类型为 '{section_type}' 的内容。"

        return "\n\n".join(sections)
    except ValueError as e:
        return str(e)


@tool
def list_cached_documents() -> str:
    """
    列出所有已缓存的 PDF 文档。

    Returns:
        已缓存文档的列表，包含文件名
    """
    retriever = _get_retriever()
    documents = retriever.list_cached_documents()

    if not documents:
        return "目前没有已缓存的文档。请先通过 PDF 或音频处理生成内容。"

    return "已缓存的文档：\n" + "\n".join(f"- {doc}" for doc in documents)


@tool
def list_knowledge_bases() -> str:
    """
    列出所有知识库。

    Returns:
        知识库列表，包含 ID、名称和描述
    """
    kb_manager = _get_kb_manager()
    kbs = kb_manager.list_knowledge_bases()

    if not kbs:
        return "目前没有创建任何知识库。"

    lines = ["可用的知识库："]
    for kb in kbs:
        pdf_count = len(kb.pdf_files)
        lines.append(f"- {kb.id}: {kb.name} ({pdf_count} 个文档)")
        if kb.description:
            lines.append(f"  描述: {kb.description}")

    return "\n".join(lines)


@tool
def get_available_pdfs() -> str:
    """
    列出所有有缓存的 PDF 文档。

    Returns:
        可用的 PDF 文档列表
    """
    kb_manager = _get_kb_manager()
    pdfs = kb_manager.get_available_pdfs()

    if not pdfs:
        return "目前没有已缓存的 PDF 文档。"

    return "可用的 PDF 文档：\n" + "\n".join(f"- {pdf}" for pdf in pdfs)


def get_all_tools():
    """
    Get all LangChain tools including knowledge base management.

    Returns:
        List of LangChain tools
    """
    return [search_cache, get_sections_by_type, list_cached_documents, list_knowledge_bases, get_available_pdfs]
