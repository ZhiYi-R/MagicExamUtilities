"""
Gradio GUI for MagicExamUtilities.

Web interface for processing PDFs and audio files to generate study notes.
"""

import os
import sys
import time
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import gradio as gr
from dotenv import load_dotenv

from utilities.workers import OCRWorker, STTWorker, SummarizationWorker, TextSource, AskAIWorker
from utilities.cache import OCRCache, STTCache
from utilities.worker import CostTracker


@dataclass
class WorkerCostInfo:
    """Cost information for a single worker with model name."""
    tracker: CostTracker = field(default_factory=CostTracker)
    model_name: str = ""
    chinese_name: str = ""


@dataclass
class ProcessingCostAccumulator:
    """Accumulate cost tracking across multiple workers during a processing session."""

    ocr: WorkerCostInfo = field(default_factory=lambda: WorkerCostInfo(chinese_name="文字识别"))
    stt: WorkerCostInfo = field(default_factory=lambda: WorkerCostInfo(chinese_name="语音转录"))
    summarization: WorkerCostInfo = field(default_factory=lambda: WorkerCostInfo(chinese_name="笔记生成"))
    ask_ai: WorkerCostInfo = field(default_factory=lambda: WorkerCostInfo(chinese_name="智能问答"))

    def reset(self):
        """Reset all cost trackers."""
        self.ocr = WorkerCostInfo(chinese_name="文字识别")
        self.stt = WorkerCostInfo(chinese_name="语音转录")
        self.summarization = WorkerCostInfo(chinese_name="笔记生成")
        self.ask_ai = WorkerCostInfo(chinese_name="智能问答")

    def get_realtime_display(self) -> str:
        """
        Get a real-time cost display for showing during processing.

        Shows current call counts and costs as they accumulate.
        """
        lines = ["## API 调用统计\n"]

        total_requests = 0
        total_cost = 0.0
        has_data = False

        # Display each active service
        services = [self.ocr, self.stt, self.summarization, self.ask_ai]
        for service in services:
            if service.tracker.request_count > 0:
                has_data = True
                total_requests += service.tracker.request_count
                total_cost += service.tracker.total_cost

                cost_str = service.tracker.format_cost(service.tracker.total_cost) if service.tracker.total_cost > 0 else "未配置"
                model_display = service.model_name if service.model_name else "未知模型"

                lines.append(
                    f"**{service.chinese_name}** ({model_display})\n"
                    f"- 调用次数: {service.tracker.request_count}\n"
                    f"- 费用: {cost_str}\n"
                )

        if not has_data:
            lines.append("*等待处理中...*")

        return "".join(lines)

    def get_total_summary(self) -> str:
        """Get a formatted summary of all costs (final summary)."""
        lines = ["## 本次处理费用统计\n"]

        total_input_tokens = 0
        total_output_tokens = 0
        total_requests = 0
        total_cost = 0.0

        sections = [
            (self.ocr.chinese_name, self.ocr.tracker, self.ocr.model_name),
            (self.stt.chinese_name, self.stt.tracker, self.stt.model_name),
            (self.summarization.chinese_name, self.summarization.tracker, self.summarization.model_name),
            (self.ask_ai.chinese_name, self.ask_ai.tracker, self.ask_ai.model_name),
        ]

        for chinese_name, tracker, model_name in sections:
            if tracker.request_count > 0:
                total_input_tokens += tracker.total_input_tokens
                total_output_tokens += tracker.total_output_tokens
                total_requests += tracker.request_count
                total_cost += tracker.total_cost

                cost_str = tracker.format_cost(tracker.total_cost) if tracker.total_cost > 0 else "N/A"
                model_display = model_name if model_name else "未知模型"

                lines.append(
                    f"**{chinese_name}** ({model_display}): {tracker.request_count} 次调用, "
                    f"{tracker.total_input_tokens:,} 输入 tokens, "
                    f"{tracker.total_output_tokens:,} 输出 tokens, "
                    f"费用: {cost_str}\n"
                )

        lines.append("\n---\n")

        if total_requests > 0:
            total_cost_str = f"${total_cost:.6f}" if total_cost > 0 else "N/A"
            lines.extend([
                f"### 总计\n",
                f"- **总调用次数**: {total_requests}\n",
                f"- **输入 tokens**: {total_input_tokens:,}\n",
                f"- **输出 tokens**: {total_output_tokens:,}\n",
                f"- **总费用**: {total_cost_str}\n"
            ])
        else:
            lines.append("*暂无费用数据（请确保在 .env 中配置了价格参数）*")

        return "".join(lines)

    def sync_from_workers(self, ocr_worker=None, stt_worker=None, summarization_worker=None, ask_ai_worker=None):
        """Sync cost data from active workers including model names."""
        if ocr_worker:
            tracker = ocr_worker.get_cost_tracker()
            self.ocr.tracker = CostTracker(
                total_input_tokens=tracker.total_input_tokens,
                total_output_tokens=tracker.total_output_tokens,
                total_cost=tracker.total_cost,
                request_count=tracker.request_count,
                input_price_per_m=tracker.input_price_per_m,
                output_price_per_m=tracker.output_price_per_m,
            )
            self.ocr.model_name = ocr_worker.get_model_name()

        if stt_worker:
            tracker = stt_worker.get_cost_tracker()
            self.stt.tracker = CostTracker(
                total_input_tokens=tracker.total_input_tokens,
                total_output_tokens=tracker.total_output_tokens,
                total_cost=tracker.total_cost,
                request_count=tracker.request_count,
                input_price_per_m=tracker.input_price_per_m,
                output_price_per_m=tracker.output_price_per_m,
            )
            self.stt.model_name = stt_worker.get_model_name()

        if summarization_worker:
            tracker = summarization_worker.get_cost_tracker()
            self.summarization.tracker = CostTracker(
                total_input_tokens=tracker.total_input_tokens,
                total_output_tokens=tracker.total_output_tokens,
                total_cost=tracker.total_cost,
                request_count=tracker.request_count,
                input_price_per_m=tracker.input_price_per_m,
                output_price_per_m=tracker.output_price_per_m,
            )
            self.summarization.model_name = summarization_worker.get_model_name()

        if ask_ai_worker:
            tracker = ask_ai_worker.get_cost_tracker()
            self.ask_ai.tracker = CostTracker(
                total_input_tokens=tracker.total_input_tokens,
                total_output_tokens=tracker.total_output_tokens,
                total_cost=tracker.total_cost,
                request_count=tracker.request_count,
                input_price_per_m=tracker.input_price_per_m,
                output_price_per_m=tracker.output_price_per_m,
            )
            self.ask_ai.model_name = ask_ai_worker.get_model_name()


class GradioApp:
    """
    Gradio application for MagicExamUtilities.

    Provides a web interface for PDF and audio processing.
    """

    def __init__(self, dump_dir: Path = Path('./cache')):
        """
        Initialize the Gradio app.

        Args:
            dump_dir: Directory for cache dumps
        """
        self._dump_dir = dump_dir
        self._dump_dir.mkdir(parents=True, exist_ok=True)

        # Workers (initialized when needed)
        self._ocr_worker = None
        self._stt_worker = None
        self._summarization_worker = None
        self._ask_ai_worker = None
        self._ocr_cache = None
        self._stt_cache = None

        # Processing state
        self._is_processing = False

        # Cost accumulator for tracking processing costs
        self._cost_accumulator = ProcessingCostAccumulator()

        print("[GUI] Initializing Gradio app...")

    def _ensure_workers(self):
        """Ensure workers are initialized."""
        if not self._ocr_worker:
            ocr_dump_path = self._dump_dir.joinpath('ocr')
            ocr_dump_path.mkdir(parents=True, exist_ok=True)
            self._ocr_worker = OCRWorker(dump_dir=ocr_dump_path, dump_ocr_response=True)
            self._ocr_worker.start()
            print("[GUI] OCR worker started")

        if not self._ocr_cache:
            self._ocr_cache = OCRCache(cache_dir=self._dump_dir)
            print("[GUI] OCR cache initialized")

        if not self._stt_cache:
            self._stt_cache = STTCache(cache_dir=self._dump_dir)
            print("[GUI] STT cache initialized")

        if not self._summarization_worker:
            summarization_dump_path = self._dump_dir.joinpath('summarization')
            summarization_dump_path.mkdir(parents=True, exist_ok=True)
            self._summarization_worker = SummarizationWorker(
                text_source=TextSource.OCR,
                dump_dir=summarization_dump_path,
                dump_summarization_response=True
            )
            self._summarization_worker.start()
            print("[GUI] Summarization worker started")

    def _get_kb_manager(self):
        """Get or create the knowledge base manager."""
        if not hasattr(self, '_kb_manager') or self._kb_manager is None:
            from utilities.knowledge_base.manager import KnowledgeBaseManager
            self._kb_manager = KnowledgeBaseManager(self._dump_dir)
        return self._kb_manager

    # Knowledge base management methods
    def _list_kbs(self) -> str:
        """List all knowledge bases."""
        kb_manager = self._get_kb_manager()
        kbs = kb_manager.list_knowledge_bases()

        if not kbs:
            return "目前没有创建任何知识库。"

        lines = ["## 知识库列表\n"]
        for kb in kbs:
            lines.append(f"### {kb.name} (`{kb.id}`)")
            if kb.description:
                lines.append(f"**描述**: {kb.description}")

            # Count total documents
            total_docs = len(kb.pdf_files) + len(kb.audio_files)
            lines.append(f"**文档数量**: {total_docs}")

            if kb.pdf_files:
                lines.append("**PDF 文档**:")
                for pdf in kb.pdf_files:
                    lines.append(f"  - {pdf}")

            if kb.audio_files:
                lines.append("**音频文档**:")
                for audio in kb.audio_files:
                    lines.append(f"  - {audio}")

            lines.append(f"**创建时间**: {kb.created_at}")
            lines.append(f"**更新时间**: {kb.updated_at}")
            lines.append("")

        return "\n".join(lines)

    def _create_kb(
        self,
        kb_id: str,
        name: str,
        description: str,
        pdf_files: List[str],
        audio_files: List[str]
    ) -> str:
        """Create a new knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb = kb_manager.create_knowledge_base(
                kb_id=kb_id,
                name=name,
                description=description,
                pdf_files=pdf_files if pdf_files else None,
                audio_files=audio_files if audio_files else None
            )
            total_docs = len(kb.pdf_files) + len(kb.audio_files)
            return f"成功创建知识库: {kb.name} (`{kb.id}`)\n\n包含 {total_docs} 个文档。"
        except ValueError as e:
            return f"创建失败: {str(e)}"

    def _delete_kb(self, kb_id: str) -> str:
        """Delete a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb_manager.delete_knowledge_base(kb_id)
            return f"成功删除知识库: `{kb_id}`"
        except ValueError as e:
            return f"删除失败: {str(e)}"

    def _add_file_to_kb(self, kb_id: str, file_type: str, file_name: str) -> str:
        """Add a file (PDF or audio) to a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            if file_type == "pdf":
                kb_manager.add_pdf_to_kb(kb_id, file_name)
                file_type_display = "PDF"
            else:
                kb_manager.add_audio_to_kb(kb_id, file_name)
                file_type_display = "音频"

            return f"成功将 {file_type_display} `{file_name}` 添加到知识库 `{kb_id}`"
        except ValueError as e:
            return f"添加失败: {str(e)}"

    def _remove_file_from_kb(self, kb_id: str, file_type: str, file_name: str) -> str:
        """Remove a file (PDF or audio) from a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            if file_type == "pdf":
                kb_manager.remove_pdf_from_kb(kb_id, file_name)
                file_type_display = "PDF"
            else:
                kb_manager.remove_audio_from_kb(kb_id, file_name)
                file_type_display = "音频"

            return f"成功从知识库 `{kb_id}` 移除 {file_type_display} `{file_name}`"
        except ValueError as e:
            return f"移除失败: {str(e)}"

    def _get_available_pdfs(self) -> str:
        """Get list of available PDFs with cache."""
        kb_manager = self._get_kb_manager()
        pdfs = kb_manager.get_available_pdfs()

        if not pdfs:
            return "目前没有已缓存的 PDF 文档。"

        return "可用的 PDF 文档：\n" + "\n".join(f"- {pdf}" for doc in pdfs)

    def _rebuild_kb_index(self, kb_id: str) -> str:
        """Rebuild index for a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb_manager.build_kb_index(kb_id)
            return f"成功重建知识库 `{kb_id}` 的索引。"
        except ValueError as e:
            return f"重建失败: {str(e)}"

    def _cleanup_workers(self):
        """Cleanup workers on shutdown."""
        print("[GUI] Cleaning up workers...")

        if self._ocr_worker:
            self._ocr_worker.shutdown(wait=True, timeout=30)
            self._ocr_worker = None
            print("[GUI] OCR worker stopped")

        if self._stt_worker:
            self._stt_worker.shutdown(wait=True, timeout=30)
            self._stt_worker = None
            print("[GUI] STT worker stopped")

        if self._summarization_worker:
            self._summarization_worker.shutdown(wait=True, timeout=30)
            self._summarization_worker = None
            print("[GUI] Summarization worker stopped")

        if self._ask_ai_worker:
            self._ask_ai_worker.shutdown(wait=True, timeout=30)
            self._ask_ai_worker = None
            print("[GUI] Ask AI worker stopped")

    def _process_pdf(
        self,
        files: List,
        output_dir: str,
        use_cache: bool,
        use_chunking: bool,
        max_chars: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str, str]:
        """
        Process PDF files.

        Args:
            files: List of uploaded file paths
            output_dir: Output directory path
            use_cache: Whether to use cache
            use_chunking: Whether to use chunking for long texts
            max_chars: Maximum characters before chunking
            progress: Gradio progress tracker

        Returns:
            Tuple of (output_message, output_file_path, preview_markdown, cost_summary)
        """
        if self._is_processing:
            return "错误: 另一个任务正在处理中", "", "", ""

        if not files:
            return "错误: 请上传至少一个 PDF 文件", "", "", ""

        # Reset cost accumulator for new processing session
        self._cost_accumulator.reset()

        self._is_processing = True
        try:
            self._ensure_workers()

            # Setup output directory
            output_path = Path(output_dir) if output_dir else self._dump_dir.joinpath('output')
            output_path.mkdir(parents=True, exist_ok=True)

            # Convert uploaded files to Path objects
            pdf_paths = []
            for file_info in files:
                src_path = Path(file_info.name)
                if src_path.suffix.lower() != '.pdf':
                    continue
                # Copy to temp location for processing
                temp_path = self._dump_dir.joinpath('uploads', src_path.name)
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, temp_path)
                pdf_paths.append(temp_path)

            if not pdf_paths:
                return "错误: 没有有效的 PDF 文件", "", "", ""

            progress(0.1, desc="初始化...")

            full_text = ''
            total_pages = 0

            # First pass: count total pages and convert all PDFs to images
            all_image_paths = []  # List of (pdf_path, image_paths) tuples
            pdf_page_counts = []  # Track page count per PDF

            for i, pdf_path in enumerate(pdf_paths):
                progress(
                    0.1 + (i / len(pdf_paths)) * 0.1,
                    desc=f"准备 {pdf_path.name} ({i+1}/{len(pdf_paths)})..."
                )

                # Check cache
                if use_cache:
                    cached = self._ocr_cache.get_pdf_cache(pdf_path)
                    if cached:
                        gr.Info(f"使用缓存: {pdf_path.name}")
                        all_image_paths.append((pdf_path, None, cached))  # None indicates cached
                        pdf_page_counts.append(len(cached.page_results))
                        total_pages += len(cached.page_results)
                        continue

                # Convert PDF to images (or reuse existing ones)
                # Import and reuse the same function from main.py
                import main
                image_paths, converted = main.convert_pdf_to_images(pdf_path, self._ocr_cache)
                if converted:
                    gr.Info(f"已转换: {pdf_path.name} ({len(image_paths)} 页)")
                else:
                    gr.Info(f"复用图片: {pdf_path.name} ({len(image_paths)} 页)")

                all_image_paths.append((pdf_path, image_paths, None))
                pdf_page_counts.append(len(image_paths))
                total_pages += len(image_paths)

            # Second pass: OCR processing with per-page progress and page-level caching
            progress(0.2, desc=f"OCR 处理中... (共 {total_pages} 页)")

            pages_processed = 0
            pdf_index = 0

            for pdf_path, image_paths, cached in all_image_paths:
                # If cached, use cached data
                if cached is not None:
                    full_text += cached.full_text
                    pages_processed += len(cached.page_results)
                    pdf_index += 1
                    continue

                # Process each page with page-level caching (allows resuming from interruption)
                page_results = []
                pdf_text_parts = []

                for i, image_path in enumerate(image_paths):
                    # Check page-level cache first (uses image existence as key)
                    cached_page = self._ocr_cache.get_page_cache(image_path)
                    if cached_page:
                        gr.Info(f"使用页面缓存: {pdf_path.name} 页 {i}")
                        page_results.append(cached_page)
                        pdf_text_parts.append(cached_page.raw_text)
                        pages_processed += 1
                        # Update progress
                        page_fraction = pages_processed / total_pages
                        current_progress = 0.2 + (page_fraction * 0.55)
                        progress(current_progress, desc=f"使用缓存: {pdf_path.name} 页 {i}/{len(image_paths)} (全局 {pages_processed}/{total_pages})")
                        continue

                    # Page cache miss - process with OCR
                    result = self._ocr_worker.process_image_structured(image_path, timeout=300)
                    page_results.append(result)
                    pdf_text_parts.append(result.raw_text)

                    # Save page-level cache immediately (for resumability)
                    if use_cache:
                        self._ocr_cache.save_page_cache(result)

                    pages_processed += 1

                    # Sync cost data from worker and update progress
                    self._cost_accumulator.sync_from_workers(ocr_worker=self._ocr_worker)
                    ocr_cost = self._cost_accumulator.ocr.tracker
                    cost_str = ocr_cost.format_cost(ocr_cost.total_cost) if ocr_cost.total_cost > 0 else "未配置"

                    page_fraction = pages_processed / total_pages
                    current_progress = 0.2 + (page_fraction * 0.55)
                    progress(current_progress, desc=f"OCR: {pdf_path.name} 页 {i}/{len(image_paths)} (全局 {pages_processed}/{total_pages}) | OCR: {ocr_cost.request_count}次 ${cost_str}")

                # Extract text
                pdf_text = ''.join(pdf_text_parts)
                full_text += pdf_text
                pdf_index += 1

                # Save full PDF cache for future fast access
                if use_cache:
                    self._ocr_cache.save_pdf_cache(pdf_path, page_results, pdf_text)

            progress(0.75, desc="生成总结...")

            # Progress callback for chunked summarization with cost tracking in description
            def chunk_progress(current, total, message):
                # Sync cost data from summarization worker
                self._cost_accumulator.sync_from_workers(summarization_worker=self._summarization_worker)
                sum_cost = self._cost_accumulator.summarization.tracker
                cost_str = sum_cost.format_cost(sum_cost.total_cost) if sum_cost.total_cost > 0 else "未配置"
                desc = f"{message} ({current}/{total}) | 笔记生成: {sum_cost.request_count}次 ${cost_str}"
                progress(0.75 + 0.15 * (current / total), desc=desc)

            # Summarize
            summary = self._summarization_worker.summarize(
                full_text,
                timeout=1200,  # Longer timeout for chunking
                use_chunking=use_chunking,
                max_chars=max_chars,
                progress_callback=chunk_progress
            )

            progress(0.93, desc="生成标题...")

            # Generate title from summary
            title = self._summarization_worker.generate_title(summary, timeout=60)

            progress(0.95, desc="保存结果...")

            # Final cost sync (including title generation)
            self._cost_accumulator.sync_from_workers(
                ocr_worker=self._ocr_worker,
                summarization_worker=self._summarization_worker
            )

            # Save output with AI-generated title
            output_file = output_path.joinpath(f'{title}.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            progress(1.0, desc="完成!")

            # Generate cost summary
            cost_summary = self._cost_accumulator.get_total_summary()

            stats = f"""
## 处理完成

- **文件数**: {len(pdf_paths)}
- **总页数**: {total_pages}
- **文本长度**: {len(full_text)} 字符
- **总结长度**: {len(summary)} 字符
- **输出文件**: {output_file.name}
"""
            return (
                stats,
                str(output_file),
                summary,
                cost_summary
            )

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", "", ""

        finally:
            self._is_processing = False

    def _process_audio(
        self,
        files: List,
        output_dir: str,
        use_cache: bool,
        use_chunking: bool,
        max_chars: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str, str]:
        """
        Process audio files.

        Args:
            files: List of uploaded file paths
            output_dir: Output directory path
            use_cache: Whether to use cache
            use_chunking: Whether to use chunking for long texts
            max_chars: Maximum characters before chunking
            progress: Gradio progress tracker

        Returns:
            Tuple of (output_message, output_file_path, preview_markdown, cost_summary)
        """
        if self._is_processing:
            return "错误: 另一个任务正在处理中", "", "", ""

        if not files:
            return "错误: 请上传至少一个音频文件", "", "", ""

        # Reset cost accumulator for new processing session
        self._cost_accumulator.reset()

        self._is_processing = True
        try:
            self._ensure_workers()

            # Initialize STT worker
            if not self._stt_worker:
                stt_dump_path = self._dump_dir.joinpath('stt')
                stt_dump_path.mkdir(parents=True, exist_ok=True)
                self._stt_worker = STTWorker(dump_dir=stt_dump_path, dump_stt_response=True)
                self._stt_worker.start()
                print("[GUI] STT worker started")

            # Setup output directory
            output_path = Path(output_dir) if output_dir else self._dump_dir.joinpath('output')
            output_path.mkdir(parents=True, exist_ok=True)

            # Convert uploaded files to Path objects
            audio_paths = []
            for file_info in files:
                src_path = Path(file_info.name)
                if src_path.suffix.lower() != '.mp3':
                    continue
                # Copy to temp location for processing
                temp_path = self._dump_dir.joinpath('uploads', src_path.name)
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, temp_path)
                audio_paths.append(temp_path)

            if not audio_paths:
                return "错误: 没有有效的音频文件", "", "", ""

            progress(0.1, desc="初始化...")

            # Process audio files with STT (using cache when enabled)
            full_text = ''
            for i, audio_path in enumerate(audio_paths):
                # Sync cost before processing for display
                self._cost_accumulator.sync_from_workers(stt_worker=self._stt_worker)
                stt_cost = self._cost_accumulator.stt.tracker
                cost_str = stt_cost.format_cost(stt_cost.total_cost) if stt_cost.total_cost > 0 else "未配置"

                progress(
                    0.1 + (i / len(audio_paths)) * 0.5,
                    desc=f"转录 {audio_path.name} ({i+1}/{len(audio_paths)}) | STT: {stt_cost.request_count}次 ${cost_str}"
                )

                # Check cache first
                if use_cache:
                    cached = self._stt_cache.get_audio_cache(audio_path)
                    if cached:
                        gr.Info(f"使用缓存: {audio_path.name}")
                        full_text += cached.raw_text
                        continue

                # 30 minutes timeout for large audio files (1800 seconds)
                text = self._stt_worker.process_audio(audio_path, timeout=1800)
                full_text += text

                # Sync cost after processing
                self._cost_accumulator.sync_from_workers(stt_worker=self._stt_worker)

                # Save to cache
                if use_cache:
                    self._stt_cache.save_audio_cache(audio_path, text)

            # Show final STT cost
            self._cost_accumulator.sync_from_workers(stt_worker=self._stt_worker)
            stt_cost = self._cost_accumulator.stt.tracker
            cost_str = stt_cost.format_cost(stt_cost.total_cost) if stt_cost.total_cost > 0 else "未配置"
            progress(0.6, desc=f"纠错文字稿... | STT: {stt_cost.request_count}次 ${cost_str}")

            # Switch summarization worker to STT mode for transcript correction
            self._summarization_worker._text_source = TextSource.STT

            # Correct the transcript
            corrected_transcript = self._summarization_worker.correct_transcript(
                full_text,
                timeout=600
            )

            progress(0.75, desc="生成总结...")

            # Progress callback for chunked summarization with cost tracking in description
            def chunk_progress(current, total, message):
                # Sync cost data from summarization worker
                self._cost_accumulator.sync_from_workers(summarization_worker=self._summarization_worker)
                sum_cost = self._cost_accumulator.summarization.tracker
                cost_str = sum_cost.format_cost(sum_cost.total_cost) if sum_cost.total_cost > 0 else "未配置"
                desc = f"{message} ({current}/{total}) | 笔记生成: {sum_cost.request_count}次 ${cost_str}"
                progress(0.75 + 0.15 * (current / total), desc=desc)

            # Summarize (using the corrected transcript)
            summary = self._summarization_worker.summarize(
                corrected_transcript,
                timeout=1200,
                use_chunking=use_chunking,
                max_chars=max_chars,
                progress_callback=chunk_progress
            )

            # Reset to OCR mode
            self._summarization_worker._text_source = TextSource.OCR

            # Show summarization cost before title generation
            self._cost_accumulator.sync_from_workers(summarization_worker=self._summarization_worker)
            sum_cost = self._cost_accumulator.summarization.tracker
            cost_str = sum_cost.format_cost(sum_cost.total_cost) if sum_cost.total_cost > 0 else "未配置"
            progress(0.93, desc=f"生成标题... | 笔记生成: {sum_cost.request_count}次 ${cost_str}")

            # Generate title from summary
            title = self._summarization_worker.generate_title(summary, timeout=60)

            progress(0.95, desc="保存结果...")

            # Final cost sync (including title generation)
            self._cost_accumulator.sync_from_workers(
                stt_worker=self._stt_worker,
                summarization_worker=self._summarization_worker
            )

            # Save outputs
            timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

            # Save corrected transcript (keep timestamp for transcript)
            transcript_file = output_path.joinpath(f'{timestamp}_transcript.md')
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(corrected_transcript)

            # Save summary with AI-generated title
            summary_file = output_path.joinpath(f'{title}.md')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            progress(1.0, desc="完成!")

            # Generate cost summary
            cost_summary = self._cost_accumulator.get_total_summary()

            stats = f"""
## 处理完成

- **文件数**: {len(audio_paths)}
- **原始转录长度**: {len(full_text)} 字符
- **纠错后长度**: {len(corrected_transcript)} 字符
- **总结长度**: {len(summary)} 字符
- **文字稿文件**: {transcript_file.name}
- **笔记文件**: {summary_file.name}
"""
            # Return summary as preview, but note that transcript is also available
            return (
                stats,
                f"{transcript_file.name}\n{summary_file.name}",  # Both files
                summary,  # Preview shows summary
                cost_summary
            )

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", "", ""

        finally:
            self._is_processing = False

    def _ask_ai(
        self,
        question: str,
        kb_id: str = "",
        progress=gr.Progress()
    ) -> str:
        """
        Process a question using Ask AI.

        Args:
            question: User's question
            kb_id: Optional knowledge base ID for scoped search
            progress: Gradio progress tracker

        Returns:
            Answer to the question
        """
        if not question or not question.strip():
            return "错误: 请输入问题"

        try:
            progress(0.1, desc="初始化 Ask AI...")

            # Initialize Ask AI worker
            if not self._ask_ai_worker:
                self._ask_ai_worker = AskAIWorker(cache_dir=self._dump_dir, dump_ask_ai_response=True)
                self._ask_ai_worker.start()
                print("[GUI] Ask AI worker started")

            progress(0.3, desc="正在思考...")

            # Ask the question with optional kb_id
            answer = self._ask_ai_worker.ask(question, kb_id=kb_id if kb_id else None, timeout=300)

            progress(1.0, desc="完成!")

            return answer

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def _get_notes_dir(self, output_dir: str) -> Path:
        """
        Get the notes directory, using default if not specified.

        Args:
            output_dir: User specified output directory

        Returns:
            Path to the notes directory
        """
        if output_dir and output_dir.strip():
            return Path(output_dir)
        return self._dump_dir.joinpath('output')

    def _list_notes(self, output_dir: str) -> gr.Radio:
        """
        List all available markdown notes in the output directory.

        Args:
            output_dir: User specified output directory

        Returns:
            Updated gr.Radio with note choices
        """
        notes_dir = self._get_notes_dir(output_dir)
        if not notes_dir.exists():
            return gr.Radio(choices=[], value=None)

        # Find all .md files (excluding transcript files which end with _transcript.md)
        note_files = []
        for file in sorted(notes_dir.glob('*.md'), key=lambda f: f.stat().st_mtime, reverse=True):
            # Skip transcript files
            if not file.name.endswith('_transcript.md'):
                note_files.append(file.name)

        return gr.Radio(choices=note_files, value=None)

    def _load_note(self, output_dir: str, note_name: str) -> Tuple[str, str]:
        """
        Load and display a note's content.

        Args:
            output_dir: User specified output directory
            note_name: Name of the note file to load

        Returns:
            Tuple of (status_message, note_content)
        """
        if not note_name:
            return "*请选择一个笔记*", ""

        notes_dir = self._get_notes_dir(output_dir)
        note_path = notes_dir.joinpath(note_name)

        if not note_path.exists():
            return f"错误: 笔记文件不存在: {note_name}", ""

        try:
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get file modification time
            import os
            mtime = os.path.getmtime(note_path)
            import time
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

            status = f"**文件**: {note_name}\n\n**修改时间**: {mtime_str}\n\n**字符数**: {len(content)}"
            return status, content
        except Exception as e:
            return f"错误: 读取笔记失败 - {str(e)}", ""

    def _save_note(self, output_dir: str, note_name: str, content: str) -> str:
        """
        Save edited content to a note file.

        Args:
            output_dir: User specified output directory
            note_name: Name of the note file to save
            content: Edited content to save

        Returns:
            Status message
        """
        if not note_name:
            return "错误: 未选择笔记文件"

        notes_dir = self._get_notes_dir(output_dir)
        note_path = notes_dir.joinpath(note_name)

        if not note_path.exists():
            return f"错误: 笔记文件不存在: {note_name}"

        try:
            # Create backup
            import shutil
            backup_path = note_path.with_suffix('.md.bak')
            shutil.copy2(note_path, backup_path)

            # Save edited content
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"成功: 笔记已保存，备份文件: {backup_path.name}"
        except Exception as e:
            return f"错误: 保存笔记失败 - {str(e)}"

    def _delete_note(self, output_dir: str, note_name: str) -> str:
        """
        Delete a note file.

        Args:
            output_dir: User specified output directory
            note_name: Name of the note file to delete

        Returns:
            Status message
        """
        if not note_name:
            return "错误: 未选择笔记文件"

        notes_dir = self._get_notes_dir(output_dir)
        note_path = notes_dir.joinpath(note_name)

        if not note_path.exists():
            return f"错误: 笔记文件不存在: {note_name}"

        try:
            # Move to trash instead of permanent delete
            import shutil
            trash_dir = notes_dir.joinpath('.trash')
            trash_dir.mkdir(exist_ok=True)

            # Add timestamp to avoid conflicts
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            trash_path = trash_dir.joinpath(f"{note_name}.{timestamp}")

            shutil.move(str(note_path), str(trash_path))

            return f"成功: 笔记已移至回收站: {trash_path.name}"
        except Exception as e:
            return f"错误: 删除笔记失败 - {str(e)}"

    def build_ui(self):
        """Build the Gradio UI with centered vertical layout."""
        with gr.Blocks(title="妙妙期末小工具") as app:
            gr.Markdown(
                """
                # 妙妙期末小工具

                从 PDF 或音频文件生成复习笔记的 AI 工具。
                """
            )

            with gr.Tabs():
                # PDF Processing Tab
                with gr.Tab("PDF 处理"):
                    with gr.Group():
                        pdf_files = gr.File(
                            label="上传 PDF 文件",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        pdf_output_dir = gr.Textbox(
                            label="输出目录",
                            placeholder="留空使用默认目录",
                            value=""
                        )
                        pdf_use_cache = gr.Checkbox(
                            label="使用缓存",
                            value=True,
                            info="启用后可跳过已处理的 PDF"
                        )
                        with gr.Accordion("高级选项", open=False):
                            pdf_use_chunking = gr.Checkbox(
                                label="启用分块总结 (处理超长文本)",
                                value=False,
                                info="文本过长时自动分块处理，可突破上下文窗口限制"
                            )
                            pdf_max_chars = gr.Number(
                                label="分块阈值 (字符数)",
                                value=8000,
                                minimum=1000,
                                maximum=50000,
                                step=1000,
                                info="超过此长度时启用分块处理"
                            )
                        with gr.Row():
                            pdf_clear_btn = gr.Button("清空", variant="secondary")
                            pdf_process_btn = gr.Button("开始处理", variant="primary", size="lg")

                    pdf_status = gr.Markdown(label="处理状态")
                    pdf_output_path = gr.Textbox(label="输出文件路径", interactive=False)

                    # Cost display section (above preview for real-time updates)
                    pdf_cost_display = gr.Markdown(
                        label="API 调用统计",
                        value="## API 调用统计\n\n*等待处理中...*"
                    )

                    # Summary preview section
                    with gr.Accordion("生成笔记预览", open=False):
                        pdf_preview = gr.Markdown(label="笔记内容")
                        with gr.Row():
                            pdf_copy_btn = gr.Button("复制笔记", size="sm", variant="secondary")

                    pdf_download = gr.DownloadButton(label="下载结果", visible=False, variant="secondary")

                    # Event handlers
                    def clear_pdf_inputs():
                        return None, "", "", "", "## API 调用统计\n\n*等待处理中...*"

                    def copy_pdf_preview():
                        """Copy the generated summary to clipboard."""
                        return gr.Info("笔记已复制到剪贴板（需浏览器支持）")

                    pdf_clear_btn.click(
                        fn=clear_pdf_inputs,
                        outputs=[pdf_files, pdf_output_dir, pdf_status, pdf_preview, pdf_cost_display]
                    )

                    pdf_copy_btn.click(
                        fn=copy_pdf_preview,
                    )

                    pdf_process_btn.click(
                        fn=self._process_pdf,
                        inputs=[pdf_files, pdf_output_dir, pdf_use_cache, pdf_use_chunking, pdf_max_chars],
                        outputs=[pdf_status, pdf_output_path, pdf_preview, pdf_cost_display],
                        show_progress="full"
                    ).then(
                        fn=lambda x: gr.DownloadButton(visible=bool(x), value=x) if x else gr.DownloadButton(visible=False),
                        inputs=pdf_output_path,
                        outputs=pdf_download
                    )

                # Audio Processing Tab
                with gr.Tab("音频处理"):
                    with gr.Group():
                        audio_files = gr.File(
                            label="上传 MP3 文件",
                            file_types=[".mp3"],
                            file_count="multiple"
                        )
                        audio_output_dir = gr.Textbox(
                            label="输出目录",
                            placeholder="留空使用默认目录",
                            value=""
                        )
                        audio_use_cache = gr.Checkbox(
                            label="使用缓存",
                            value=True,
                            info="启用后可跳过已处理的音频文件"
                        )
                        with gr.Accordion("高级选项", open=False):
                            audio_use_chunking = gr.Checkbox(
                                label="启用分块总结 (处理超长文本)",
                                value=False,
                                info="文本过长时自动分块处理，可突破上下文窗口限制"
                            )
                            audio_max_chars = gr.Number(
                                label="分块阈值 (字符数)",
                                value=8000,
                                minimum=1000,
                                maximum=50000,
                                step=1000,
                                info="超过此长度时启用分块处理"
                            )
                        with gr.Row():
                            audio_clear_btn = gr.Button("清空", variant="secondary")
                            audio_process_btn = gr.Button("开始处理", variant="primary", size="lg")

                    audio_status = gr.Markdown(label="处理状态")
                    audio_output_path = gr.Textbox(label="输出文件路径", interactive=False, visible=False)

                    # Cost display section (above preview for real-time updates)
                    audio_cost_display = gr.Markdown(
                        label="API 调用统计",
                        value="## API 调用统计\n\n*等待处理中...*"
                    )

                    # Summary preview section
                    with gr.Accordion("生成笔记预览", open=False):
                        audio_preview = gr.Markdown(label="笔记内容")
                        with gr.Row():
                            audio_copy_btn = gr.Button("复制笔记", size="sm", variant="secondary")

                    # Download buttons
                    with gr.Row():
                        audio_transcript_download = gr.DownloadButton(label="下载文字稿", visible=False, variant="secondary")
                        audio_summary_download = gr.DownloadButton(label="下载笔记", visible=False, variant="primary")

                    # Event handlers
                    def clear_audio_inputs():
                        return None, "", "", "", "## API 调用统计\n\n*等待处理中...*"

                    def copy_audio_preview():
                        """Copy the generated summary to clipboard."""
                        return gr.Info("笔记已复制到剪贴板（需浏览器支持）")

                    def update_audio_downloads(files_string):
                        """Update download buttons based on output files."""
                        if not files_string:
                            return (
                                gr.DownloadButton(visible=False),
                                gr.DownloadButton(visible=False)
                            )
                        files = files_string.strip().split('\n')
                        # Get output directory from first file path
                        if len(files) > 0:
                            first_file_path = Path(files[0])
                            output_dir = first_file_path.parent
                        else:
                            output_dir = self._dump_dir.joinpath('output')
                        return (
                            gr.DownloadButton(visible=True, value=str(output_dir / files[0])) if len(files) > 0 else gr.DownloadButton(visible=False),
                            gr.DownloadButton(visible=True, value=str(output_dir / files[1])) if len(files) > 1 else gr.DownloadButton(visible=False),
                        )

                    audio_clear_btn.click(
                        fn=clear_audio_inputs,
                        outputs=[audio_files, audio_output_dir, audio_status, audio_preview, audio_cost_display]
                    )

                    audio_copy_btn.click(
                        fn=copy_audio_preview,
                    )

                    audio_process_btn.click(
                        fn=self._process_audio,
                        inputs=[audio_files, audio_output_dir, audio_use_cache, audio_use_chunking, audio_max_chars],
                        outputs=[audio_status, audio_output_path, audio_preview, audio_cost_display],
                        show_progress="full"
                    ).then(
                        fn=update_audio_downloads,
                        inputs=audio_output_path,
                        outputs=[audio_transcript_download, audio_summary_download]
                    )

                # Ask AI Tab
                with gr.Tab("Ask AI"):
                    with gr.Group():
                        ask_kb_dropdown = gr.Dropdown(
                            label="知识库 (可选)",
                            choices=[],
                            value="",
                            allow_custom_value=False,
                            info="留空则搜索所有缓存"
                        )
                        ask_question = gr.Textbox(
                            label="问题",
                            placeholder="请输入您的问题...",
                            lines=3
                        )
                        with gr.Row():
                            ask_clear_btn = gr.Button("清空", variant="secondary")
                            ask_process_btn = gr.Button("提问", variant="primary", size="lg")
                        gr.Examples(
                            examples=[
                                "MySQL 的安装步骤是什么？",
                                "Java 中 JDBC 和 PreparedStatement 有什么区别？",
                                "总结一下多线程的核心概念",
                                "找出所有关于数据库配置的内容"
                            ],
                            inputs=ask_question
                        )

                    ask_answer = gr.Markdown(label="回答")

                    # Event handlers
                    def clear_ask_inputs():
                        return "", ""

                    ask_clear_btn.click(
                        fn=clear_ask_inputs,
                        outputs=[ask_question, ask_answer]
                    )

                    ask_process_btn.click(
                        fn=self._ask_ai,
                        inputs=[ask_question, ask_kb_dropdown],
                        outputs=[ask_answer],
                        show_progress="full"
                    )

                # Note Management Tab
                with gr.Tab("笔记管理"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 笔记列表")
                            notes_output_dir = gr.Textbox(
                                label="笔记目录",
                                placeholder="留空使用默认输出目录",
                                value=""
                            )
                            refresh_notes_btn = gr.Button("刷新列表", variant="primary", size="sm")
                            notes_list = gr.Radio(
                                label="选择笔记",
                                choices=[],
                                info="选择要预览/编辑的笔记"
                            )
                            delete_note_btn = gr.Button("删除选中笔记", variant="stop", size="sm")

                        with gr.Column(scale=2):
                            gr.Markdown("### 笔记预览与编辑")
                            note_preview = gr.Markdown(
                                label="笔记内容",
                                value="*请从左侧选择要查看的笔记*"
                            )
                            note_editor = gr.Textbox(
                                label="编辑笔记内容",
                                lines=20,
                                visible=False,
                                interactive=True
                            )
                            with gr.Row():
                                edit_mode_btn = gr.Button("进入编辑模式", variant="secondary")
                                save_note_btn = gr.Button("保存修改", variant="primary", visible=False)
                                cancel_edit_btn = gr.Button("取消编辑", variant="secondary", visible=False)

                            note_status = gr.Markdown(label="操作状态", value="")

                    # Event handlers for note management
                    def refresh_notes_list(output_dir):
                        """Refresh the list of available notes."""
                        return self._list_notes(output_dir)

                    def select_note(output_dir, note_name):
                        """Select and display a note."""
                        return self._load_note(output_dir, note_name)

                    def enter_edit_mode(current_note):
                        """Enter edit mode for the current note."""
                        return (
                            gr.Textbox(visible=True, interactive=True, value=current_note),
                            gr.Button(visible=True),
                            gr.Button(visible=True),
                            gr.Button(visible=False),
                            "*编辑模式下可直接修改上方内容，修改完成后点击「保存修改」*"
                        )

                    def save_note(output_dir, note_name, edited_content):
                        """Save the edited note."""
                        result = self._save_note(output_dir, note_name, edited_content)
                        # Refresh the note preview
                        _, updated_content = self._load_note(output_dir, note_name)
                        return (
                            result,
                            updated_content,
                            gr.Textbox(visible=False, interactive=False),
                            gr.Button(visible=False),
                            gr.Button(visible=False),
                            gr.Button(visible=True),
                            result
                        )

                    def cancel_edit():
                        """Cancel edit mode."""
                        return (
                            gr.Textbox(visible=False, interactive=False),
                            gr.Button(visible=False),
                            gr.Button(visible=False),
                            gr.Button(visible=True),
                            "*已取消编辑*"
                        )

                    def delete_selected_note(output_dir, note_name):
                        """Delete the selected note."""
                        result = self._delete_note(output_dir, note_name)
                        # Refresh the notes list
                        updated_choices = self._list_notes(output_dir)
                        return (
                            result,
                            updated_choices,
                            gr.Radio(value=None),
                            "*请从左侧选择要查看的笔记*",
                            result
                        )

                    refresh_notes_btn.click(
                        fn=refresh_notes_list,
                        inputs=[notes_output_dir],
                        outputs=[notes_list]
                    )

                    notes_list.change(
                        fn=select_note,
                        inputs=[notes_output_dir, notes_list],
                        outputs=[note_preview, note_status]
                    )

                    edit_mode_btn.click(
                        fn=enter_edit_mode,
                        inputs=[note_preview],
                        outputs=[note_editor, save_note_btn, cancel_edit_btn, edit_mode_btn, note_status]
                    )

                    save_note_btn.click(
                        fn=save_note,
                        inputs=[notes_output_dir, notes_list, note_editor],
                        outputs=[note_status, note_preview, note_editor, save_note_btn, cancel_edit_btn, edit_mode_btn]
                    )

                    cancel_edit_btn.click(
                        fn=cancel_edit,
                        outputs=[note_editor, save_note_btn, cancel_edit_btn, edit_mode_btn, note_status]
                    )

                    delete_note_btn.click(
                        fn=delete_selected_note,
                        inputs=[notes_output_dir, notes_list],
                        outputs=[note_status, notes_list, notes_list, note_preview, note_status]
                    )

                # Knowledge Base Management Tab
                with gr.Tab("知识库管理"):
                    # Create KB section
                    with gr.Group():
                        gr.Markdown("### 创建知识库")
                        create_kb_id = gr.Textbox(
                            label="知识库 ID",
                            placeholder="例如: database_sys (仅限字母、数字、下划线、连字符)"
                        )
                        create_kb_name = gr.Textbox(
                            label="知识库名称",
                            placeholder="例如: 数据库系统"
                        )
                        create_kb_desc = gr.Textbox(
                            label="描述",
                            placeholder="选填",
                            lines=2
                        )
                        with gr.Row():
                            create_kb_pdfs = gr.CheckboxGroup(
                                label="选择 PDF 文档",
                                choices=[],
                                info="可多选"
                            )
                            create_kb_audios = gr.CheckboxGroup(
                                label="选择音频文档",
                                choices=[],
                                info="可多选"
                            )
                        with gr.Row():
                            create_kb_btn = gr.Button("创建知识库", variant="primary")
                            refresh_create_files_btn = gr.Button("刷新文件列表", variant="secondary")

                    # KB list section
                    with gr.Group():
                        with gr.Row():
                            gr.Markdown("### 知识库列表")
                            refresh_kb_list_btn = gr.Button("刷新", size="sm")
                        kb_list_output = gr.Markdown(label="知识库列表")

                    # Management actions in accordion
                    with gr.Accordion("管理操作", open=False):
                        with gr.Group():
                            gr.Markdown("#### 添加/移除文档")
                            manage_kb_id = gr.Dropdown(label="选择知识库", choices=[])
                            manage_file_type = gr.Radio(
                                label="文档类型",
                                choices=[("PDF", "pdf"), ("音频", "audio")],
                                value="pdf"
                            )
                            manage_file_name = gr.Dropdown(label="文档", choices=[])
                            with gr.Row():
                                add_file_btn = gr.Button("添加文档", variant="secondary")
                                remove_file_btn = gr.Button("移除文档", variant="stop")
                                refresh_manage_files_btn = gr.Button("刷新文档", variant="secondary")
                            manage_output = gr.Markdown(label="操作状态")

                        with gr.Group():
                            gr.Markdown("#### 重建索引")
                            rebuild_kb_id = gr.Dropdown(label="选择知识库", choices=[])
                            rebuild_index_btn = gr.Button("重建索引", variant="primary")
                            rebuild_output = gr.Markdown(label="状态")

                    # Event handlers
                    def create_kb_handler(kb_id, name, desc, pdfs, audios):
                        result = self._create_kb(kb_id, name, desc, pdfs, audios)
                        kb_manager = self._get_kb_manager()
                        kbs = kb_manager.list_knowledge_bases()
                        kb_choices = [kb.id for kb in kbs]
                        pdfs_list = kb_manager.get_available_pdfs()
                        audios_list = kb_manager.get_available_audios()
                        return (
                            result,
                            gr.Dropdown(choices=[""] + kb_choices),
                            gr.Dropdown(choices=kb_choices),
                            gr.Dropdown(choices=kb_choices),
                            gr.Dropdown(choices=pdfs_list + audios_list),
                            gr.CheckboxGroup(choices=pdfs_list),
                            gr.CheckboxGroup(choices=audios_list),
                        )

                    def refresh_create_files_handler():
                        kb_manager = self._get_kb_manager()
                        pdfs = kb_manager.get_available_pdfs()
                        audios = kb_manager.get_available_audios()
                        return (
                            gr.CheckboxGroup(choices=pdfs),
                            gr.CheckboxGroup(choices=audios),
                        )

                    def manage_file_type_change(file_type):
                        kb_manager = self._get_kb_manager()
                        if file_type == "pdf":
                            return gr.Dropdown(choices=kb_manager.get_available_pdfs())
                        else:
                            return gr.Dropdown(choices=kb_manager.get_available_audios())

                    def refresh_manage_files_handler():
                        kb_manager = self._get_kb_manager()
                        return gr.Dropdown(choices=kb_manager.get_available_pdfs() + kb_manager.get_available_audios())

                    create_kb_btn.click(
                        fn=create_kb_handler,
                        inputs=[create_kb_id, create_kb_name, create_kb_desc, create_kb_pdfs, create_kb_audios],
                        outputs=[kb_list_output, ask_kb_dropdown, manage_kb_id, rebuild_kb_id, manage_file_name, create_kb_pdfs, create_kb_audios]
                    )

                    refresh_create_files_btn.click(
                        fn=refresh_create_files_handler,
                        outputs=[create_kb_pdfs, create_kb_audios]
                    )

                    manage_file_type.change(
                        fn=manage_file_type_change,
                        inputs=[manage_file_type],
                        outputs=[manage_file_name]
                    )

                    refresh_manage_files_btn.click(
                        fn=refresh_manage_files_handler,
                        outputs=[manage_file_name]
                    )

                    refresh_kb_list_btn.click(
                        fn=self._list_kbs,
                        outputs=kb_list_output
                    )

                    add_file_btn.click(
                        fn=self._add_file_to_kb,
                        inputs=[manage_kb_id, manage_file_type, manage_file_name],
                        outputs=manage_output
                    )

                    remove_file_btn.click(
                        fn=self._remove_file_from_kb,
                        inputs=[manage_kb_id, manage_file_type, manage_file_name],
                        outputs=manage_output
                    )

                    rebuild_index_btn.click(
                        fn=self._rebuild_kb_index,
                        inputs=[rebuild_kb_id],
                        outputs=rebuild_output
                    )

            gr.Markdown(
                """
                ---

                **提示**:
                - PDF 处理: 支持多文件上传，自动使用缓存加速重复处理
                - 音频处理: 支持 MP3 格式录音文件，支持缓存加速重复处理
                - Ask AI: 基于已缓存的 PDF/音频内容回答问题，可选择知识库限定搜索范围
                - 知识库管理: 按科目分组管理 PDF，支持独立索引和搜索
                - 处理时间取决于文件大小和 API 响应速度
                """
            )

        return app

    def launch(self, server_port: int = 7860):
        """
        Launch the Gradio app.

        Args:
            server_port: Port to run the server on
        """
        app = self.build_ui()

        # Register cleanup on close
        app.close = lambda: self._cleanup_workers()

        app.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            share=False,
            show_error=True,
            quiet=False,
        )


def main():
    """Main entry point for GUI mode."""
    load_dotenv()

    # Get port from environment or use default
    port = int(os.environ.get('GRADIO_PORT', 7860))

    app = GradioApp()
    app.launch(server_port=port)


if __name__ == '__main__':
    main()
