"""
Ask AI Worker for answering questions using cached content.

Uses LangChain with tool calling to search and retrieve information
from cached OCR results.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config, get_retry_config
from ..retrieval.tools import get_all_tools

logger = logging.getLogger(__name__)


class AskAIWorker(BaseWorker):
    """
    Worker for AI-powered question answering.

    Uses LLM with tool calling to answer questions based on
    cached OCR content from PDFs and audio files.
    """

    def __init__(self,
                 cache_dir: Path = Path('./cache'),
                 dump_ask_ai_response: bool = True,
                 task_timeout: Optional[float] = None):
        """
        Initialize the Ask AI worker.

        Args:
            cache_dir: Directory for cache and dumps
            dump_ask_ai_response: Whether to dump Ask AI responses to JSON
            task_timeout: Timeout for individual ask tasks in seconds (None = no timeout)
        """
        rpm, tpm = get_rate_limit_config('ASK_AI')
        max_retries, retry_delay = get_retry_config('ASK_AI')

        super().__init__(name='AskAIWorker', rpm=rpm, tpm=tpm, pricing_prefix='ASK_AI',
                        max_retries=max_retries, retry_delay=retry_delay, task_timeout=task_timeout)

        self._cache_dir = cache_dir
        self._dump_dir = cache_dir.joinpath('ask_ai')
        self._dump_ask_ai_response = dump_ask_ai_response

        # Use ASK_AI specific config if available, otherwise fall back
        api_url = os.environ.get('ASK_AI_API_URL',
                                 os.environ.get('OPENAI_LIKE_API_URL'))
        api_key = os.environ.get('ASK_AI_API_KEY',
                                  os.environ.get('OPENAI_LIKE_API_KEY'))
        model = os.environ.get('ASK_AI_MODEL',
                               'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')

        # Create LangChain LLM
        self._llm = ChatOpenAI(
            base_url=api_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
        )

        # Create OpenAI client for direct API calls
        self._client = OpenAI(base_url=api_url, api_key=api_key)
        self._model = model

        # Get tools
        self._tools = get_all_tools()
        self._llm_with_tools = self._llm.bind_tools(self._tools)

        # Register methods
        self._register_methods()

        ocr_type = 'ask_ai'
        print(f"[AskAIWorker] Initialized with model: {model}")
        print(f"[AskAIWorker] Rate limits: RPM={rpm}, TPM={tpm}")
        print(f"[AskAIWorker] Tools: {[t.name for t in self._tools]}")

    def _register_methods(self) -> None:
        """Register Ask AI methods."""
        self._methods = {
            'ask': self._ask,
        }

    def get_model_name(self) -> str:
        """Get the model name used by this worker."""
        return self._model

    def _track_langchain_usage(self, response, call_type: str) -> None:
        """
        Track usage from LangChain response.

        Args:
            response: LangChain ChatMessage response
            call_type: Type of call ('initial', 'final', 'direct')
        """
        # LangChain may store usage in different places depending on version
        input_tokens = 0
        output_tokens = 0
        found_usage = False

        # Try usage_metadata first (newer LangChain versions)
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            if usage:
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                found_usage = True

        # Try response_metadata as fallback
        elif hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                found_usage = True

        # Track usage if tokens were found
        if input_tokens > 0 or output_tokens > 0:
            cost = self._track_usage(input_tokens, output_tokens)
            cost_str = self._cost_tracker.format_cost(cost) if cost > 0 else "N/A"
            print(f'[AskAIWorker] {call_type} call - usage: {input_tokens} in + {output_tokens} out, cost: {cost_str}')
        elif not found_usage:
            logger.warning(f'[AskAIWorker] {call_type} call - LangChain response did not include usage information, cost tracking disabled for this request')

    def _ask(self, question: str, kb_id: Optional[str] = None, _timeout: Optional[float] = None) -> str:
        """
        Process a question with tool calling.

        Args:
            question: User's question
            kb_id: Optional knowledge base ID for scoped search
            _timeout: Maximum time to wait for API calls (internal parameter)

        Returns:
            Answer to the question
        """
        print(f'[AskAIWorker] Processing question: {question}' + (f' (KB: {kb_id})' if kb_id else ''))

        # Build system prompt
        kb_context = f"\n\n当前搜索范围：知识库 '{kb_id}'" if kb_id else ""
        system_prompt = f"""你是一个学习助手，帮助学生从已经处理的课件和笔记中回答问题。{kb_context}

你可以使用以下工具来搜索缓存的内容：
- search_cache: 搜索关键词相关的内容（可指定知识库 ID 进行限定搜索）
- get_sections_by_type: 按类型获取内容（表格、公式、代码等）
- list_cached_documents: 列出所有已缓存的文档
- list_knowledge_bases: 列出所有知识库
- get_available_pdfs: 列出所有有缓存的 PDF 文档

请根据用户的问题，使用合适的工具搜索相关信息，然后给出准确的答案。

回答时请注意：
1. 如果找到相关内容，请引用来源
2. 如果没有找到相关内容，请诚实告知
3. 回答要简洁明了"""

        # Try tool calling first
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]

            response = self._llm_with_tools.invoke(messages)

            # Check if tool calls are needed
            if hasattr(response, 'response_metadata') and 'tool_calls' in response.response_metadata:
                # Handle tool calls
                tool_calls = response.response_metadata['tool_calls']

                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})

                    # Auto-inject kb_id for search_cache if specified
                    if tool_name == 'search_cache' and kb_id:
                        tool_args['kb_id'] = kb_id

                    # Find and execute the tool
                    for tool in self._tools:
                        if tool.name == tool_name:
                            result = tool.invoke(tool_args)
                            tool_results.append(ToolMessage(
                                content=result,
                                tool_call_id=tool_call.get('id', '')
                            ))
                            break

                # Get final response with tool results
                messages.extend(tool_results)
                final_response = self._llm.invoke(messages)
                answer = final_response.content

                # Track costs for both calls (initial + final)
                self._track_langchain_usage(response, 'initial')
                self._track_langchain_usage(final_response, 'final')

            else:
                # No tool calls needed, direct answer
                answer = response.content
                # Track costs for single call
                self._track_langchain_usage(response, 'direct')

        except Exception as e:
            print(f'[AskAIWorker] Tool calling failed: {e}')
            # Fallback to direct API call
            try:
                api_response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': question}
                    ],
                    temperature=0.0,
                    timeout=_timeout
                )

                if not api_response.choices:
                    raise RuntimeError('No response from API')

                answer = api_response.choices[0].message.content

                # Track costs for fallback API call
                if api_response.usage:
                    input_tokens = api_response.usage.prompt_tokens
                    output_tokens = api_response.usage.completion_tokens
                    cost = self._track_usage(input_tokens, output_tokens)
                    cost_str = self._cost_tracker.format_cost(cost) if cost > 0 else "N/A"
                    print(f'[AskAIWorker] Fallback API call - usage: {input_tokens} in + {output_tokens} out, cost: {cost_str}')
                else:
                    logger.warning('[AskAIWorker] Fallback API call did not include usage information, cost tracking disabled for this request')

            except Exception as e2:
                print(f'[AskAIWorker] Direct API call also failed: {e2}')
                answer = f"抱歉，处理您的问题时出现错误：{str(e)}"

        print(f'[AskAIWorker] Answer generated, length: {len(answer)}')

        # Dump response if needed
        if self._dump_ask_ai_response:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            import time
            dump_file = self._dump_dir.joinpath(f'ask_{int(time.time())}.json')
            with open(dump_file, 'w', encoding='utf-8') as f:
                f.write(f'{{"question": "{question}", "answer": "{answer}"}}')

        return answer

    def ask(self, question: str, kb_id: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """
        Ask a question (public interface).

        Args:
            question: User's question
            kb_id: Optional knowledge base ID for scoped search
            timeout: Maximum time to wait for result (seconds) - enforced at worker level

        Returns:
            Answer to the question
        """
        future = self.submit('ask', question, kb_id, _task_timeout=timeout)
        return future.get()

    def _estimate_tokens(self, task) -> int:
        """Estimate tokens for Ask AI requests."""
        # Conservative estimate: question + response ~2000 tokens
        return 2000
