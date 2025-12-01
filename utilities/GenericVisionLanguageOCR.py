import os
import json
import base64
import pathlib
from openai import OpenAI


class GenericVL:
    def __init__(self, dump_dir: pathlib.Path = pathlib.Path('.'), dump_ocr_response: bool = True) -> None:
        self.__client = OpenAI(
            base_url=f'{os.environ["OPENAI_LIKE_API_URL"]}',
            api_key=f'{os.environ["OPENAI_LIKE_API_KEY"]}'
        )
        if dump_ocr_response:
            dump_dir.mkdir(parents=True, exist_ok=True)
        self.__dump_dir = dump_dir
        self.__dump_ocr_response = dump_ocr_response
        self.__text_prompt = '''
        你是一个专业的图像抄录员，你需要抄录图像中的文本内容，并输出Markdown格式的文本。你需要注意以下几点：
        1. 抄录的文本内容必须与图像中的内容一致。
        2. 如果图像中包含表格，你需要将表格抄录为Markdown格式的表格。
        3. 如果图像中包含公式，你需要将公式抄录为Markdown格式的公式。
        4. 如果图像中包含流程图或是架构图，你需要将其转换为Mermaid图像嵌入到Markdown中。
        5. 如果图像中包含代码，你需要将代码抄录为Markdown格式的代码块。
        6. 如果图像中包含电路图，你需要将电路图转换为Tikz代码块（使用CircuiTikz宏包）嵌入到Markdown中。
        '''
    
    def ocr(self, image_path: pathlib.Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f'Image {image_path} does not exist')
        print(f'OCRing image: {image_path}')
        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            response = self.__client.chat.completions.create(
                model=os.environ['OCR_MODEL'],
                messages=[
                    {'role': 'system', 'content': self.__text_prompt},
                    {'role': 'user', 'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}',
                            'detail': 'high'
                        }
                    } , {
                        'type': 'text',
                        'text': '请抄录图像中的文本内容'
                    }]}
                ],
                temperature = 0.0,
                top_p = 1.0,
                max_tokens = 1024,
                frequency_penalty = 0.0,
                presence_penalty = 0.2,
                extra_body = {
                    'repetition_penalty': 1.02,
                    'presence_penalty': 0.2
                }
            )
            if not response.choices:
                raise RuntimeError(f'No choices returned from OpenAI API for image {image_path}')
            if not response.choices[0].message.content:
                raise RuntimeError(f'No content returned from OpenAI API for image {image_path}')
            if not response.usage:
                print(f'OCR Done for image: {image_path}, length: {len(response.choices[0].message.content)}')
            else:
                print(f'OCR Done for image: {image_path}, length: {len(response.choices[0].message.content)}, usage: {response.usage.total_tokens}')
            if not self.__dump_ocr_response:
                return response.choices[0].message.content
            else:
                dump_file_path = self.__dump_dir.joinpath(f'{image_path.stem}.json')
                with open(dump_file_path, 'w') as dump_file:
                    json.dump(response.model_dump(), dump_file, ensure_ascii=False, indent=4)
                return response.choices[0].message.content
