"""
LangChain tools for cache retrieval.

Defines tools that can be used by LLMs for function calling.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from .retriever import CacheRetriever

# Global retriever instance
_retriever: Optional[CacheRetriever] = None


def _get_retriever() -> CacheRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        cache_dir = Path(os.environ.get('CACHE_DIR', './cache/'))
        _retriever = CacheRetriever(cache_dir)
        _retriever.load_or_build_index()
    return _retriever


@tool
def search_cache(query: str, method: str = "hybrid", limit: int = 5) -> str:
    """
    搜索已缓存的 PDF 内容。

    支持关键词搜索、语义搜索或混合搜索。

    Args:
        query: 搜索关键词或问题
        method: 搜索方式，可选值：keyword（关键词）、semantic（语义）、hybrid（混合）
        limit: 返回结果数量，默认 5 条

    Returns:
        匹配的内容片段，包含来源信息
    """
    retriever = _get_retriever()
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


def get_all_tools():
    """
    Get all LangChain tools for cache retrieval.

    Returns:
        List of LangChain tools
    """
    return [search_cache, get_sections_by_type, list_cached_documents]


# Import os for environment variable
import os
