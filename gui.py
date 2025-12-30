"""
Gradio GUI for MagicExamUtilities.

Web interface for processing PDFs and audio files to generate study notes.
"""

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import gradio as gr
from dotenv import load_dotenv

from utilities.workers import OCRWorker, STTWorker, SummarizationWorker, TextSource, AskAIWorker
from utilities.cache import OCRCache


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

        # Processing state
        self._is_processing = False

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
            lines.append(f"**文档数量**: {len(kb.pdf_files)}")
            if kb.pdf_files:
                lines.append("**包含文档**:")
                for pdf in kb.pdf_files:
                    lines.append(f"  - {pdf}")
            lines.append(f"**创建时间**: {kb.created_at}")
            lines.append(f"**更新时间**: {kb.updated_at}")
            lines.append("")

        return "\n".join(lines)

    def _create_kb(
        self,
        kb_id: str,
        name: str,
        description: str,
        pdf_files: List[str]
    ) -> str:
        """Create a new knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb = kb_manager.create_knowledge_base(
                kb_id=kb_id,
                name=name,
                description=description,
                pdf_files=pdf_files if pdf_files else None
            )
            return f"成功创建知识库: {kb.name} (`{kb.id}`)\n\n包含 {len(kb.pdf_files)} 个文档。"
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

    def _add_pdf_to_kb(self, kb_id: str, pdf_name: str) -> str:
        """Add a PDF to a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb_manager.add_pdf_to_kb(kb_id, pdf_name)
            return f"成功将 `{pdf_name}` 添加到知识库 `{kb_id}`"
        except ValueError as e:
            return f"添加失败: {str(e)}"

    def _remove_pdf_from_kb(self, kb_id: str, pdf_name: str) -> str:
        """Remove a PDF from a knowledge base."""
        kb_manager = self._get_kb_manager()

        try:
            kb_manager.remove_pdf_from_kb(kb_id, pdf_name)
            return f"成功从知识库 `{kb_id}` 移除 `{pdf_name}`"
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
    ) -> Tuple[str, str, str]:
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
            Tuple of (output_message, output_file_path, preview_markdown)
        """
        if self._is_processing:
            return "错误: 另一个任务正在处理中", "", ""

        if not files:
            return "错误: 请上传至少一个 PDF 文件", "", ""

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
                return "错误: 没有有效的 PDF 文件", "", ""

            progress(0.1, desc="初始化...")

            full_text = ''
            total_pages = 0

            for i, pdf_path in enumerate(pdf_paths):
                progress(
                    0.1 + (i / len(pdf_paths)) * 0.6,
                    desc=f"处理 {pdf_path.name} ({i+1}/{len(pdf_paths)})..."
                )

                # Check cache
                if use_cache:
                    cached = self._ocr_cache.get_pdf_cache(pdf_path)
                    if cached:
                        gr.Info(f"使用缓存: {pdf_path.name}")
                        full_text += cached.full_text
                        total_pages += len(cached.page_results)
                        continue

                # Convert PDF to images
                from pdf2image import convert_from_path

                images = convert_from_path(
                    pdf_path=pdf_path,
                    dpi=300,
                    fmt='jpeg',
                    grayscale=False
                )

                # Save images to temp directory
                image_dir = self._dump_dir.joinpath(pdf_path.stem)
                image_dir.mkdir(parents=True, exist_ok=True)
                image_paths = []
                for j, image in enumerate(images):
                    image_path = image_dir.joinpath(f'{pdf_path.stem}_{j}.jpg')
                    image.save(image_path, 'JPEG')
                    image_paths.append(image_path)

                # Process images with OCR
                page_results = self._ocr_worker.process_images_structured(
                    image_paths,
                    timeout_per_image=300
                )

                # Extract text
                pdf_text = ''.join([r.raw_text for r in page_results])
                full_text += pdf_text
                total_pages += len(page_results)

                # Save to cache
                if use_cache:
                    self._ocr_cache.save_pdf_cache(pdf_path, page_results, pdf_text)

            progress(0.75, desc="生成总结...")

            # Progress callback for chunked summarization
            def chunk_progress(current, total, message):
                progress(0.75 + 0.15 * (current / total), desc=message)

            # Summarize
            summary = self._summarization_worker.summarize(
                full_text,
                timeout=1200,  # Longer timeout for chunking
                use_chunking=use_chunking,
                max_chars=max_chars,
                progress_callback=chunk_progress
            )

            progress(0.95, desc="保存结果...")

            # Save output
            timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
            output_file = output_path.joinpath(f'{timestamp}.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            progress(1.0, desc="完成!")

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
                summary
            )

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", ""

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
    ) -> Tuple[str, str, str]:
        """
        Process audio files.

        Args:
            files: List of uploaded file paths
            output_dir: Output directory path
            use_cache: Whether to use cache (not used for audio currently)
            use_chunking: Whether to use chunking for long texts
            max_chars: Maximum characters before chunking
            progress: Gradio progress tracker

        Returns:
            Tuple of (output_message, output_file_path, preview_markdown)
        """
        if self._is_processing:
            return "错误: 另一个任务正在处理中", "", ""

        if not files:
            return "错误: 请上传至少一个音频文件", "", ""

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
                return "错误: 没有有效的音频文件", "", ""

            progress(0.1, desc="初始化...")

            # Process audio files with STT
            full_text = ''
            for i, audio_path in enumerate(audio_paths):
                progress(
                    0.1 + (i / len(audio_paths)) * 0.7,
                    desc=f"转录 {audio_path.name} ({i+1}/{len(audio_paths)})..."
                )
                text = self._stt_worker.process_audio(audio_path, timeout=600)
                full_text += text

            progress(0.85, desc="生成总结...")

            # Switch summarization worker to STT mode
            self._summarization_worker._text_source = TextSource.STT

            # Progress callback for chunked summarization
            def chunk_progress(current, total, message):
                progress(0.85 + 0.08 * (current / total), desc=message)

            # Summarize
            summary = self._summarization_worker.summarize(
                full_text,
                timeout=1200,
                use_chunking=use_chunking,
                max_chars=max_chars,
                progress_callback=chunk_progress
            )

            # Reset to OCR mode
            self._summarization_worker._text_source = TextSource.OCR

            progress(0.95, desc="保存结果...")

            # Save output
            timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
            output_file = output_path.joinpath(f'{timestamp}.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            progress(1.0, desc="完成!")

            stats = f"""
## 处理完成

- **文件数**: {len(audio_paths)}
- **文本长度**: {len(full_text)} 字符
- **总结长度**: {len(summary)} 字符
- **输出文件**: {output_file.name}
"""
            return (
                stats,
                str(output_file),
                summary
            )

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", ""

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

                    # Summary preview section
                    with gr.Accordion("生成笔记预览", open=False):
                        pdf_preview = gr.Markdown(label="笔记内容")
                        with gr.Row():
                            pdf_copy_btn = gr.Button("复制笔记", size="sm", variant="secondary")

                    pdf_download = gr.DownloadButton(label="下载结果", visible=False, variant="secondary")

                    # Event handlers
                    def clear_pdf_inputs():
                        return None, "", "", ""

                    def copy_pdf_preview():
                        """Copy the generated summary to clipboard."""
                        return gr.Info("笔记已复制到剪贴板（需浏览器支持）")

                    pdf_clear_btn.click(
                        fn=clear_pdf_inputs,
                        outputs=[pdf_files, pdf_output_dir, pdf_status, pdf_preview]
                    )

                    pdf_copy_btn.click(
                        fn=copy_pdf_preview,
                    )

                    pdf_process_btn.click(
                        fn=self._process_pdf,
                        inputs=[pdf_files, pdf_output_dir, pdf_use_cache, pdf_use_chunking, pdf_max_chars],
                        outputs=[pdf_status, pdf_output_path, pdf_preview],
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
                            value=False,
                            info="音频暂不支持缓存",
                            interactive=False
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
                    audio_output_path = gr.Textbox(label="输出文件路径", interactive=False)

                    # Summary preview section
                    with gr.Accordion("生成笔记预览", open=False):
                        audio_preview = gr.Markdown(label="笔记内容")
                        with gr.Row():
                            audio_copy_btn = gr.Button("复制笔记", size="sm", variant="secondary")

                    audio_download = gr.DownloadButton(label="下载结果", visible=False, variant="secondary")

                    # Event handlers
                    def clear_audio_inputs():
                        return None, "", "", ""

                    def copy_audio_preview():
                        """Copy the generated summary to clipboard."""
                        return gr.Info("笔记已复制到剪贴板（需浏览器支持）")

                    audio_clear_btn.click(
                        fn=clear_audio_inputs,
                        outputs=[audio_files, audio_output_dir, audio_status, audio_preview]
                    )

                    audio_copy_btn.click(
                        fn=copy_audio_preview,
                    )

                    audio_process_btn.click(
                        fn=self._process_audio,
                        inputs=[audio_files, audio_output_dir, audio_use_cache, audio_use_chunking, audio_max_chars],
                        outputs=[audio_status, audio_output_path, audio_preview],
                        show_progress="full"
                    ).then(
                        fn=lambda x: gr.DownloadButton(visible=bool(x), value=x) if x else gr.DownloadButton(visible=False),
                        inputs=audio_output_path,
                        outputs=audio_download
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
                        create_kb_pdfs = gr.CheckboxGroup(
                            label="选择 PDF 文档",
                            choices=[],
                            info="可多选，留空创建空知识库"
                        )
                        with gr.Row():
                            create_kb_btn = gr.Button("创建知识库", variant="primary")
                            refresh_create_pdfs_btn = gr.Button("刷新 PDF 列表", variant="secondary")

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
                            manage_pdf_name = gr.Dropdown(label="PDF 文档", choices=[])
                            with gr.Row():
                                add_pdf_btn = gr.Button("添加文档", variant="secondary")
                                remove_pdf_btn = gr.Button("移除文档", variant="stop")
                                refresh_manage_pdfs_btn = gr.Button("刷新 PDF", variant="secondary")
                            manage_output = gr.Markdown(label="操作状态")

                        with gr.Group():
                            gr.Markdown("#### 重建索引")
                            rebuild_kb_id = gr.Dropdown(label="选择知识库", choices=[])
                            rebuild_index_btn = gr.Button("重建索引", variant="primary")
                            rebuild_output = gr.Markdown(label="状态")

                    # Event handlers
                    def create_kb_handler(kb_id, name, desc, pdfs):
                        result = self._create_kb(kb_id, name, desc, pdfs)
                        kb_manager = self._get_kb_manager()
                        kbs = kb_manager.list_knowledge_bases()
                        kb_choices = [kb.id for kb in kbs]
                        pdfs_list = kb_manager.get_available_pdfs()
                        return (
                            result,
                            gr.Dropdown(choices=[""] + kb_choices),
                            gr.Dropdown(choices=kb_choices),
                            gr.Dropdown(choices=kb_choices),
                            gr.Dropdown(choices=kb_choices),
                            gr.Dropdown(choices=pdfs_list),
                            gr.CheckboxGroup(choices=pdfs_list),
                        )

                    def refresh_pdfs_handler():
                        kb_manager = self._get_kb_manager()
                        pdfs = kb_manager.get_available_pdfs()
                        return gr.CheckboxGroup(choices=pdfs)

                    def refresh_manage_pdfs_handler():
                        kb_manager = self._get_kb_manager()
                        pdfs = kb_manager.get_available_pdfs()
                        return gr.Dropdown(choices=pdfs)

                    create_kb_btn.click(
                        fn=create_kb_handler,
                        inputs=[create_kb_id, create_kb_name, create_kb_desc, create_kb_pdfs],
                        outputs=[kb_list_output, ask_kb_dropdown, manage_kb_id, rebuild_kb_id, manage_pdf_name, manage_pdf_name, create_kb_pdfs]
                    )

                    refresh_create_pdfs_btn.click(
                        fn=refresh_pdfs_handler,
                        outputs=[create_kb_pdfs]
                    )

                    refresh_manage_pdfs_btn.click(
                        fn=refresh_manage_pdfs_handler,
                        outputs=[manage_pdf_name]
                    )

                    refresh_kb_list_btn.click(
                        fn=self._list_kbs,
                        outputs=kb_list_output
                    )

                    add_pdf_btn.click(
                        fn=self._add_pdf_to_kb,
                        inputs=[manage_kb_id, manage_pdf_name],
                        outputs=manage_output
                    )

                    remove_pdf_btn.click(
                        fn=self._remove_pdf_from_kb,
                        inputs=[manage_kb_id, manage_pdf_name],
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
                - 音频处理: 支持 MP3 格式录音文件
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
