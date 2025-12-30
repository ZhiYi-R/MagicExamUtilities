"""
Ask AI Worker for answering questions using cached content.

Uses LangChain with tool calling to search and retrieve information
from cached OCR results.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import OpenAI

from ..worker import BaseWorker, get_rate_limit_config
from ..retrieval.tools import get_all_tools


class AskAIWorker(BaseWorker):
    """
    Worker for AI-powered question answering.

    Uses LLM with tool calling to answer questions based on
    cached OCR content from PDFs and audio files.
    """

    def __init__(self,
                 cache_dir: Path = Path('./cache'),
                 dump_ask_ai_response: bool = True):
        """
        Initialize the Ask AI worker.

        Args:
            cache_dir: Directory for cache and dumps
            dump_ask_ai_response: Whether to dump Ask AI responses to JSON
        """
        rpm, tpm = get_rate_limit_config('ASK_AI')
        super().__init__(name='AskAIWorker', rpm=rpm, tpm=tpm)

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

    def _ask(self, question: str, timeout: Optional[float] = None) -> str:
        """
        Process a question with tool calling.

        Args:
            question: User's question
            timeout: Maximum time to wait for result

        Returns:
            Answer to the question
        """
        print(f'[AskAIWorker] Processing question: {question}')

        # Build system prompt
        system_prompt = """你是一个学习助手，帮助学生从已经处理的课件和笔记中回答问题。

你可以使用以下工具来搜索缓存的内容：
- search_cache: 搜索关键词相关的内容
- get_sections_by_type: 按类型获取内容（表格、公式、代码等）
- list_cached_documents: 列出所有已缓存的文档

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

            else:
                # No tool calls needed, direct answer
                answer = response.content

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
                )

                if not api_response.choices:
                    raise RuntimeError('No response from API')

                answer = api_response.choices[0].message.content

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

    def ask(self, question: str, timeout: Optional[float] = None) -> str:
        """
        Ask a question (public interface).

        Args:
            question: User's question
            timeout: Maximum time to wait for result

        Returns:
            Answer to the question
        """
        future = self.submit('ask', question)
        return future.get(timeout=timeout)

    def _estimate_tokens(self, task) -> int:
        """Estimate tokens for Ask AI requests."""
        # Conservative estimate: question + response ~2000 tokens
        return 2000
