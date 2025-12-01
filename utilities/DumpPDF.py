from typing import List
from pathlib import Path
from pdf2image import convert_from_path


def dump_pdf(pdf_path: Path, dump_dir: Path) -> List[Path]:
    print(f'Dumping PDF to images: {pdf_path}')
    dump_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(
        pdf_path = pdf_path,
        dpi = 300,
        fmt = 'jpeg',
        grayscale = False
    )
    image_paths = []
    for i, image in enumerate(images):
        image_path = dump_dir.joinpath(f'{pdf_path.stem}_{i}.jpg')
        image.save(image_path, 'JPEG')
        print(f'Dumped image: {image_path}')
        image_paths.append(image_path)
    return image_paths