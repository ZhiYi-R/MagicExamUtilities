"""
Main entry point for MagicExamUtilities.

Supports processing PDFs (via OCR) and audio files (via STT)
to generate summarized study notes.
"""

import os
import sys
import time
import argparse
import atexit
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path

from utilities.workers import OCRWorker, STTWorker, SummarizationWorker, TextSource, AskAIWorker
from utilities.cache import OCRCache


# Global workers for cleanup
_ocr_worker = None
_stt_worker = None
_summarization_worker = None
_ask_ai_worker = None


def cleanup_workers():
    """Cleanup workers on exit."""
    global _ocr_worker, _stt_worker, _summarization_worker, _ask_ai_worker

    print("\n[Main] Shutting down workers...")

    if _ocr_worker:
        _ocr_worker.shutdown(wait=True, timeout=30)
    if _stt_worker:
        _stt_worker.shutdown(wait=True, timeout=30)
    if _summarization_worker:
        _summarization_worker.shutdown(wait=True, timeout=30)
    if _ask_ai_worker:
        _ask_ai_worker.shutdown(wait=True, timeout=30)

    print("[Main] Workers shut down complete")


def validate_files(file_paths: list[str], file_type: str) -> list[Path]:
    """
    Validate and filter input files.

    Args:
        file_paths: List of file paths to validate
        file_type: Expected file type ('pdf' or 'audio')

    Returns:
        List of valid Path objects
    """
    valid_files = []

    for path in file_paths:
        wrapped_path = Path(path)

        if not wrapped_path.exists():
            print(f'[Warning] File {path} does not exist')
            continue

        if not wrapped_path.is_file():
            print(f'[Warning] {path} is not a file')
            continue

        if file_type == 'pdf' and wrapped_path.suffix != '.pdf':
            print(f'[Warning] {path} is not a PDF file')
            continue

        if file_type == 'audio' and wrapped_path.suffix != '.mp3':
            print(f'[Warning] {path} is not an MP3 file')
            continue

        valid_files.append(wrapped_path)

    return valid_files


def convert_pdf_to_images(pdf_path: Path, dump_dir: Path) -> list[Path]:
    """
    Convert a PDF file to images.

    Args:
        pdf_path: Path to the PDF file
        dump_dir: Directory to save the images

    Returns:
        List of paths to the generated images
    """
    print(f'[Main] Converting PDF to images: {pdf_path}')
    dump_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(
        pdf_path=pdf_path,
        dpi=300,
        fmt='jpeg',
        grayscale=False
    )

    image_paths = []
    for i, image in enumerate(images):
        image_path = dump_dir.joinpath(f'{pdf_path.stem}_{i}.jpg')
        image.save(image_path, 'JPEG')
        print(f'[Main] Dumped image: {image_path}')
        image_paths.append(image_path)

    return image_paths


def process_pdf_pipeline(files: list[Path], dump_dir: Path, dump: bool, output_dir: Path) -> None:
    """
    Process PDF files through the OCR + Summarization pipeline.

    Args:
        files: List of PDF files to process
        dump_dir: Base directory for cache dumps
        dump: Whether to dump intermediate results
        output_dir: Directory for final output
    """
    global _ocr_worker, _summarization_worker

    print(f'[Main] Starting OCR summarization for {len(files)} PDF(s)')

    # Initialize cache manager
    ocr_cache = OCRCache(cache_dir=dump_dir)

    # Initialize OCR worker
    ocr_dump_path = dump_dir.joinpath('ocr')
    ocr_dump_path.mkdir(parents=True, exist_ok=True)
    _ocr_worker = OCRWorker(dump_dir=ocr_dump_path, dump_ocr_response=dump)
    _ocr_worker.start()

    # Process each PDF
    all_page_results = []
    full_text = ''

    for pdf_path in files:
        print(f'[Main] Processing PDF: {pdf_path.name}')

        # Check cache first
        cached = ocr_cache.get_pdf_cache(pdf_path)
        if cached:
            print(f'[Main] Using cached OCR results for {pdf_path.name}')
            all_page_results.extend(cached.page_results)
            full_text += cached.full_text
            continue

        # Convert PDF to images
        pdf_dump_path = dump_dir.joinpath(pdf_path.stem)
        image_paths = convert_pdf_to_images(pdf_path, pdf_dump_path)
        print(f'[Main] Converted {pdf_path.name} to {len(image_paths)} image(s)')

        # Process images with structured OCR
        print(f'[Main] Processing {len(image_paths)} image(s) with OCR...')
        page_results = _ocr_worker.process_images_structured(image_paths, timeout_per_image=300)

        # Extract full text from results
        pdf_text = ''.join([r.raw_text for r in page_results])
        full_text += pdf_text
        all_page_results.extend(page_results)
        print(f'[Main] OCR complete for {pdf_path.name}. Text length: {len(pdf_text)} characters')

        # Save to cache
        ocr_cache.save_pdf_cache(pdf_path, page_results, pdf_text)

    print(f'[Main] All OCR complete. Total text length: {len(full_text)} characters')

    # Initialize and start Summarization worker
    summarization_dump_path = dump_dir.joinpath('summarization')
    summarization_dump_path.mkdir(parents=True, exist_ok=True)
    _summarization_worker = SummarizationWorker(
        text_source=TextSource.OCR,
        dump_dir=summarization_dump_path,
        dump_summarization_response=dump
    )
    _summarization_worker.start()

    # Summarize the text
    print('[Main] Summarizing text...')
    summary = _summarization_worker.summarize(full_text, timeout=600)
    print(f'[Main] Summarization complete. Summary length: {len(summary)} characters')

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f'{time.strftime("%Y_%m_%d-%H_%M_%S")}.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'[Main] Output written to: {output_file}')


def process_audio_pipeline(files: list[Path], dump_dir: Path, dump: bool, output_dir: Path) -> None:
    """
    Process audio files through the STT + Summarization pipeline.

    Args:
        files: List of audio files to process
        dump_dir: Base directory for cache dumps
        dump: Whether to dump intermediate results
        output_dir: Directory for final output
    """
    global _stt_worker, _summarization_worker

    print(f'[Main] Starting STT summarization for {len(files)} audio file(s)')

    # Step 1: Initialize and start STT worker
    stt_dump_path = dump_dir.joinpath('stt')
    stt_dump_path.mkdir(parents=True, exist_ok=True)
    _stt_worker = STTWorker(dump_dir=stt_dump_path, dump_stt_response=dump)
    _stt_worker.start()

    # Step 2: Process audio files with STT
    print(f'[Main] Processing {len(files)} audio file(s) with STT...')
    full_text = ''
    for i, audio in enumerate(files, 1):
        print(f'[Main] STT progress: {i}/{len(files)} - {audio.name}')
        text = _stt_worker.process_audio(audio, timeout=600)
        full_text += text
    print(f'[Main] STT complete. Total text length: {len(full_text)} characters')

    # Step 3: Initialize and start Summarization worker
    summarization_dump_path = dump_dir.joinpath('summarization')
    summarization_dump_path.mkdir(parents=True, exist_ok=True)
    _summarization_worker = SummarizationWorker(
        text_source=TextSource.STT,
        dump_dir=summarization_dump_path,
        dump_summarization_response=dump
    )
    _summarization_worker.start()

    # Step 4: Summarize the text
    print('[Main] Summarizing text...')
    summary = _summarization_worker.summarize(full_text, timeout=600)
    print(f'[Main] Summarization complete. Summary length: {len(summary)} characters')

    # Step 5: Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f'{time.strftime("%Y_%m_%d-%H_%M_%S")}.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'[Main] Output written to: {output_file}')


def process_ask_pipeline(question: str, dump_dir: Path) -> None:
    """
    Process a question using Ask AI with cached content.

    Args:
        question: User's question
        dump_dir: Base directory for cache
    """
    global _ask_ai_worker

    print(f'[Main] Asking AI: {question}')

    # Initialize Ask AI worker
    _ask_ai_worker = AskAIWorker(cache_dir=dump_dir, dump_ask_ai_response=True)
    _ask_ai_worker.start()

    # Ask the question
    answer = _ask_ai_worker.ask(question, timeout=300)

    print(f'\n[Main] Answer:\n{answer}')


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Generate study notes from PDFs or audio recordings using AI'
    )
    parser.add_argument('--gui', action='store_true',
                        help='Launch GUI mode')
    parser.add_argument('--ask', type=str, metavar='QUESTION',
                        help='Ask a question using cached content (Ask AI mode)')
    parser.add_argument('--type', type=str, choices=['pdf', 'audio'],
                        help='Type of input file (required in CLI mode)')
    parser.add_argument('--input', type=str, nargs='+',
                        help='Input files (supports wildcards) (required in CLI mode)')
    parser.add_argument('--dump', action='store_true',
                        help='Dump intermediate results to cache', default=True)
    parser.add_argument('--no-dump', dest='dump', action='store_false',
                        help='Do not dump intermediate results')
    parser.add_argument('--dump-dir', type=str,
                        help='Cache directory', default='./cache/')
    parser.add_argument('--output', type=str,
                        help='Output directory (required in CLI mode)')

    args = parser.parse_args()

    # GUI mode
    if args.gui:
        from gui import GradioApp
        port = int(os.environ.get('GRADIO_PORT', 7860))
        app = GradioApp(dump_dir=Path(args.dump_dir))
        app.launch(server_port=port)
        return

    # Ask AI mode
    if args.ask:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(cleanup_workers)
        process_ask_pipeline(args.ask, dump_dir)
        return

    # CLI mode - validate required arguments
    if not args.type:
        parser.error('--type is required in CLI mode (or use --ask for Ask AI mode or --gui for GUI mode)')
    if not args.input:
        parser.error('--input is required in CLI mode (or use --ask for Ask AI mode or --gui for GUI mode)')
    if not args.output:
        parser.error('--output is required in CLI mode (or use --ask for Ask AI mode or --gui for GUI mode)')

    # Validate input files
    files = validate_files(args.input, args.type)
    if not files:
        print('[Error] No valid input files found')
        sys.exit(1)

    # Setup directories
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output)

    # Register cleanup handler
    atexit.register(cleanup_workers)

    # Process based on type
    try:
        if args.type == 'pdf':
            process_pdf_pipeline(files, dump_dir, args.dump, output_dir)
        elif args.type == 'audio':
            process_audio_pipeline(files, dump_dir, args.dump, output_dir)

        print('[Main] Processing complete!')

    except KeyboardInterrupt:
        print('\n[Main] Interrupted by user')
        sys.exit(130)
    except Exception as e:
        print(f'[Error] Processing failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
