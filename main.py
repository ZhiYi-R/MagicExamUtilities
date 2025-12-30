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

from utilities.DumpPDF import dump_pdf
from utilities.workers import OCRWorker, STTWorker, SummarizationWorker, TextSource


# Global workers for cleanup
_ocr_worker = None
_stt_worker = None
_summarization_worker = None


def cleanup_workers():
    """Cleanup workers on exit."""
    global _ocr_worker, _stt_worker, _summarization_worker

    print("\n[Main] Shutting down workers...")

    if _ocr_worker:
        _ocr_worker.shutdown(wait=True, timeout=30)
    if _stt_worker:
        _stt_worker.shutdown(wait=True, timeout=30)
    if _summarization_worker:
        _summarization_worker.shutdown(wait=True, timeout=30)

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

    # Step 1: Convert PDFs to images
    image_paths = []
    for pdf_path in files:
        dump_path = dump_dir.joinpath(pdf_path.stem)
        dump_path.mkdir(parents=True, exist_ok=True)
        image_paths.extend(dump_pdf(pdf_path, dump_path))
    print(f'[Main] Converted {len(files)} PDF(s) to {len(image_paths)} image(s)')

    # Step 2: Initialize and start OCR worker
    ocr_dump_path = dump_dir.joinpath('ocr')
    ocr_dump_path.mkdir(parents=True, exist_ok=True)
    _ocr_worker = OCRWorker(dump_dir=ocr_dump_path, dump_ocr_response=dump)
    _ocr_worker.start()

    # Step 3: Process images with OCR
    print(f'[Main] Processing {len(image_paths)} image(s) with OCR...')
    full_text = ''
    for i, image in enumerate(image_paths, 1):
        print(f'[Main] OCR progress: {i}/{len(image_paths)} - {image.name}')
        text = _ocr_worker.process_image(image, timeout=300)
        full_text += text
    print(f'[Main] OCR complete. Total text length: {len(full_text)} characters')

    # Step 4: Initialize and start Summarization worker
    summarization_dump_path = dump_dir.joinpath('summarization')
    summarization_dump_path.mkdir(parents=True, exist_ok=True)
    _summarization_worker = SummarizationWorker(
        text_source=TextSource.OCR,
        dump_dir=summarization_dump_path,
        dump_summarization_response=dump
    )
    _summarization_worker.start()

    # Step 5: Summarize the text
    print('[Main] Summarizing text...')
    summary = _summarization_worker.summarize(full_text, timeout=600)
    print(f'[Main] Summarization complete. Summary length: {len(summary)} characters')

    # Step 6: Write output
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


def main():
    """Main entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Generate study notes from PDFs or audio recordings using AI'
    )
    parser.add_argument('--type', type=str, choices=['pdf', 'audio'],
                        help='Type of input file', required=True)
    parser.add_argument('--input', type=str, nargs='+',
                        help='Input files (supports wildcards)', required=True)
    parser.add_argument('--dump', action='store_true',
                        help='Dump intermediate results to cache', default=True)
    parser.add_argument('--no-dump', dest='dump', action='store_false',
                        help='Do not dump intermediate results')
    parser.add_argument('--dump-dir', type=str,
                        help='Cache directory', default='./cache/')
    parser.add_argument('--output', type=str,
                        help='Output directory', required=True)

    args = parser.parse_args()

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
