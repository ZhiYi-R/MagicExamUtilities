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
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Process PDF files.

        Args:
            files: List of uploaded file paths
            output_dir: Output directory path
            use_cache: Whether to use cache
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

            # Summarize
            summary = self._summarization_worker.summarize(full_text, timeout=600)

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
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Process audio files.

        Args:
            files: List of uploaded file paths
            output_dir: Output directory path
            use_cache: Whether to use cache (not used for audio currently)
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

            # Summarize
            summary = self._summarization_worker.summarize(full_text, timeout=600)

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
        progress=gr.Progress()
    ) -> str:
        """
        Process a question using Ask AI.

        Args:
            question: User's question
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

            # Ask the question
            answer = self._ask_ai_worker.ask(question, timeout=300)

            progress(1.0, desc="完成!")

            return answer

        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def build_ui(self):
        """Build the Gradio UI."""
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
                    with gr.Row():
                        with gr.Column(scale=1):
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
                            pdf_process_btn = gr.Button(
                                "开始处理",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=1):
                            pdf_status = gr.Markdown(label="状态")
                            pdf_output_path = gr.Textbox(
                                label="输出文件路径",
                                interactive=False
                            )
                            pdf_preview = gr.Markdown(label="预览")
                            pdf_download = gr.DownloadButton(
                                label="下载结果",
                                visible=False,
                                variant="secondary"
                            )

                    pdf_process_btn.click(
                        fn=self._process_pdf,
                        inputs=[pdf_files, pdf_output_dir, pdf_use_cache],
                        outputs=[pdf_status, pdf_output_path, pdf_preview],
                        show_progress="full"
                    ).then(
                        fn=lambda x: gr.DownloadButton(visible=bool(x), value=x) if x else gr.DownloadButton(visible=False),
                        inputs=pdf_output_path,
                        outputs=pdf_download
                    )

                # Audio Processing Tab
                with gr.Tab("音频处理"):
                    with gr.Row():
                        with gr.Column(scale=1):
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
                                info="音频暂不支持缓存"
                            )
                            audio_process_btn = gr.Button(
                                "开始处理",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=1):
                            audio_status = gr.Markdown(label="状态")
                            audio_output_path = gr.Textbox(
                                label="输出文件路径",
                                interactive=False
                            )
                            audio_preview = gr.Markdown(label="预览")
                            audio_download = gr.DownloadButton(
                                label="下载结果",
                                visible=False,
                                variant="secondary"
                            )

                    audio_process_btn.click(
                        fn=self._process_audio,
                        inputs=[audio_files, audio_output_dir, audio_use_cache],
                        outputs=[audio_status, audio_output_path, audio_preview],
                        show_progress="full"
                    ).then(
                        fn=lambda x: gr.DownloadButton(visible=bool(x), value=x) if x else gr.DownloadButton(visible=False),
                        inputs=audio_output_path,
                        outputs=audio_download
                    )

                # Ask AI Tab
                with gr.Tab("Ask AI"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ask_question = gr.Textbox(
                                label="问题",
                                placeholder="请输入您的问题...",
                                lines=3
                            )
                            ask_process_btn = gr.Button(
                                "提问",
                                variant="primary",
                                size="lg"
                            )
                            gr.Examples(
                                examples=[
                                    "MySQL 的安装步骤是什么？",
                                    "Java 中 JDBC 和 PreparedStatement 有什么区别？",
                                    "总结一下多线程的核心概念",
                                    "找出所有关于数据库配置的内容"
                                ],
                                inputs=ask_question
                            )

                        with gr.Column(scale=1):
                            ask_answer = gr.Markdown(label="回答")

                    ask_process_btn.click(
                        fn=self._ask_ai,
                        inputs=[ask_question],
                        outputs=[ask_answer],
                        show_progress="full"
                    )

            gr.Markdown(
                """
                ---

                **提示**:
                - PDF 处理: 支持多文件上传，自动使用缓存加速重复处理
                - 音频处理: 支持 MP3 格式录音文件
                - Ask AI: 基于已缓存的 PDF/音频内容回答问题，需要先处理文档才能使用
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
