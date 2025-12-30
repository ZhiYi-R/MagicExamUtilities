from utilities.STT import SpeechToText
from utilities.DumpPDF import dump_pdf
from utilities.DeepSeekOCR import DeepSeekOCR
from utilities.GenericVisionLanguageOCR import GenericVL
from utilities.Summarization import Summarization, TextSource


import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(description='Summarize a PDF file')
    parser.add_argument('--type', type=str, choices=['pdf', 'audio'], help='Type of input file')
    parser.add_argument('--input', type=str, nargs='+', help='Input files')
    parser.add_argument('--dump', action='store_true', help='Dump the results to JSON files', required=False, default=True)
    parser.add_argument('--dump-dir', type=str, help='Dump directory', required=False, default='./cache/')
    parser.add_argument('--output', type=str, help='Output directory')
    args = parser.parse_args()
    file_to_processs = []
    for path in args.input:
        wrapped_path = Path(path)
        if not wrapped_path.exists():
            print(f'File {path} does not exist')
            continue
        if not wrapped_path.is_file():
            print(f'{path} is not a file')
            continue
        if args.type == 'pdf' and wrapped_path.suffix != '.pdf':
            print(f'{path} is not a PDF file')
            continue
        if args.type == 'audio' and wrapped_path.suffix != '.mp3':
            print(f'{path} is not an MP3 file')
            continue
        file_to_processs.append(wrapped_path)
    dump_dir_base = Path(args.dump_dir)
    dump_dir_base.mkdir(parents=True, exist_ok=True)
    if args.type == 'pdf':
        print(f'Starting OCR summarization for PDFs')
        image_pathes = []
        for pdf_path in file_to_processs:
            dump_path = dump_dir_base.joinpath(pdf_path.stem)
            dump_path.mkdir(parents=True, exist_ok=True)
            image_pathes.extend(dump_pdf(pdf_path, dump_path))
        ocr_dump_path = dump_dir_base.joinpath('ocr')
        ocr_dump_path.mkdir(parents=True, exist_ok=True)
        json_dump = False
        if args.dump:
            json_dump = True
        else:
            json_dump = False
        if os.environ['OCR_USE_DEEPSEEK_OCR']:
            ocr = DeepSeekOCR(dump_dir=ocr_dump_path, dump_ocr_response=json_dump)
        else:
            ocr = GenericVL(dump_dir=ocr_dump_path, dump_ocr_response=json_dump)
        full_text = ''
        for image in image_pathes:
            full_text = full_text + ocr.ocr(image)
            print(f'OCR Done for image: {image}')
        summarization_dump_path = dump_dir_base.joinpath('summarization')
        summarization_dump_path.mkdir(parents=True, exist_ok=True)
        summarization = Summarization(TextSource.OCR, dump_dir=summarization_dump_path, dump_summarization_response=json_dump)
        summary = summarization.summarize(full_text)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath(f'{time.strftime("%Y_%m_%d-%H_%M_%S")}.md')
        with open(output_file, 'w') as f:
            f.write(summary)
        print(f'Summarization Done')
    elif args.type == 'audio':
        print(f'Starting STT summarization for MP3s')
        stt_dump_path = dump_dir_base.joinpath('stt')
        stt_dump_path.mkdir(parents=True, exist_ok=True)
        json_dump = False
        if args.dump:
            json_dump = True
        else:
            json_dump = False
        stt = SpeechToText(dump_dir=stt_dump_path, dump_stt_response=json_dump)
        full_text = ''
        for audio in file_to_processs:
            full_text = full_text + stt.stt(audio)
            print(f'STT Done for audio: {audio}')
        summarization_dump_path = dump_dir_base.joinpath('summarization')
        summarization_dump_path.mkdir(parents=True, exist_ok=True)
        summarization = Summarization(TextSource.STT, dump_dir=summarization_dump_path, dump_summarization_response=json_dump)
        summary = summarization.summarize(full_text)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath(f'{time.strftime("%Y_%m_%d-%H_%M_%S")}.md')
        with open(output_file, 'w') as f:
            f.write(summary)
        print(f'Summarization Done')
        
        
        
