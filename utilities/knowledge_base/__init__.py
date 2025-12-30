"""
Knowledge base management module.

Provides functionality to organize cached PDF content into separate knowledge bases
for isolated retrieval.
"""

from .manager import KnowledgeBaseManager
from .models import KnowledgeBase

__all__ = ['KnowledgeBaseManager', 'KnowledgeBase']
