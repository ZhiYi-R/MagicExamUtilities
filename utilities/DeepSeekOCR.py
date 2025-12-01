import os
import json
import base64
import pathlib
from openai import OpenAI


class DeepSeekOCR:
    def __init__(self, dump_dir: pathlib.Path = pathlib.Path('.'), dump_ocr_response: bool = True) -> None:
        self.__client = OpenAI(
            base_url=f'{os.environ["OPENAI_LIKE_API_URL"]}',
            api_key=f'{os.environ["OPENAI_LIKE_API_KEY"]}'
        )
        # Do NOT change prompt, since deepseek ocr is extremely sensitive to prompt
        self.__text_prompt = '<image>\nOCR this image with Markdown format.'
        if dump_ocr_response:
            dump_dir.mkdir(parents=True, exist_ok=True)
        self.__dump_dir = dump_dir
        self.__dump_ocr_response = dump_ocr_response
    
    def ocr(self, image_path: pathlib.Path) -> str:
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            response = self.__client.chat.completions.create(
                model=os.environ['OCR_MODEL'],
                messages=[{'role': 'user', 'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{image_base64}',
                            'detail': 'high'
                        }
                    } , {
                        'type': 'text',
                        'text': self.__text_prompt
                    }
                ]}],
                temperature = 0.0,
                top_p = 1.0,
                max_tokens = 1024,
                frequency_penalty = 0.0,
                presence_penalty = 0.2,
                extra_body = {
                    'repetition_penalty': 1.02,
                    'presence_penalty': 0.2,
                }
            )
        if not response.choices:
            raise RuntimeError(f'No choices returned from OpenAI API for image {image_path}')
        if not response.choices[0].message.content:
            raise RuntimeError(f'No content returned from OpenAI API for image {image_path}')
        if not self.__dump_ocr_response:
            return response.choices[0].message.content
        else:
            dump_file_path = self.__dump_dir.joinpath(f'{image_path.stem}.json')
            with open(dump_file_path, 'w') as f:
                json.dump(response.model_dump_json(), f, ensure_ascii=False, indent=4)
                print(f'Dumped OCR response to {dump_file_path}')
            return response.choices[0].message.content
        