import os
import time
import json
from enum import Enum
from pathlib import Path
from openai import OpenAI

class TextSource(Enum):
    OCR = 'OCR',
    STT = 'STT'


class Summarization:
    def __init__(self, text_source: TextSource, dump_dir: Path = Path('.'), dump_summarization_response: bool = True) -> None:
        self.__client = OpenAI(
            base_url = os.environ['OPENAI_LIKE_API_URL'],
            api_key = os.environ['OPENAI_LIKE_API_KEY']
        )
        self.__dump_dir = dump_dir
        self.__dump_summarization_response = dump_summarization_response
        if text_source == TextSource.OCR:
            self.__text_prompt = '''
            你是一个专业的AI助手，专门为OCR识别出的文本进行纠错和总结，你的行为守则如下：
            1. 输出结果为Markdown格式。
            2. 输出结果中应当包含原始文本中的所有关键信息，不要缺少任何重要的信息。
            3. 确保你的输出只包含简体中文，如果原文是别的语言，确保你已经进行了翻译。
            4. 请确保你的公式输出正确，请使用LaTeX格式。
            5. 请确保你的表格输出正确，请使用Markdown格式。
            6. 如果内容是复习课，确保你输出的文本包含复习的所有内容，不要缺少任何重要的信息，如果其讲的不够详细，请补充其详细内容。
            7. 你的输出除了报告的Markdown文本外，请不要输出任何多余的文本（直接输出Markdown源码，不要包裹任何代码块之类的东西)。
            '''
        elif text_source == TextSource.STT:
            self.__text_prompt = '''
            你是一个专业的AI助手，专门为语音转录出的文本进行纠错和总结，你的行为守则如下：
            1. 确保你的输出是Markdown格式的，并只包含原始录音的内容。
            2. 确保你的输出是经过格式化、缩进和换行处理的，并符合Markdown语法。
            3. 确保你的输出是经过润色处理的，并符合原始录音的内容。
            4. 确保你的输出只包含简体中文，如果原文是别的语言，确保你已经进行了翻译。
            5. 确保你输出的文本包含原始录音的所有关键信息，不要缺少任何重要的信息。
            6. 如果录音内容是复习课，确保你输出的文本包含复习的所有内容，不要缺少任何重要的信息，如果其讲的不够详细，请补充其详细内容。
            7. 你的输出除了报告的Markdown文本外，请不要输出任何多余的文本（直接输出Markdown源码，不要包裹任何代码块之类的东西)。
            '''
        else:
            raise ValueError(f'Invalid text source: {text_source}')
    
    def summarize(self, text: str) -> str:
        print(f'Summarizing text of length {len(text)}')
        if 'qwen3' in os.environ['SUMMARIZATION_MODEL'].lower():
            user_prompt = f'\\no_think 请对以下文本进行总结，请使用Markdown格式输出：\n{text}'
        else:
            user_prompt = f'请对以下文本进行总结，请使用Markdown格式输出：\n{text}'
        response = self.__client.chat.completions.create(
            model = os.environ['SUMMARIZATION_MODEL'],
            messages = [
                {'role': 'system', 'content': self.__text_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature = 0.1,
            top_p = 1.0,
        )
        if not response.choices:
            raise RuntimeError(f'No choices returned from OpenAI API for text {text}')
        if not response.choices[0].message.content:
            raise RuntimeError(f'No content returned from OpenAI API for text {text}')
        if not response.usage:
            print(f'Summarization Done for text, length: {len(response.choices[0].message.content)}')
        else:
            print(f'Summarization Done for text, length: {len(response.choices[0].message.content)}, usage: {response.usage.total_tokens}')
        if self.__dump_summarization_response:
            dump_file_path = self.__dump_dir.joinpath(f'Summarization_{time.strftime("%Y_%m_%d-%H_%M_%S")}.json')
            with open(dump_file_path, 'w') as f:
                json.dump(response.model_dump_json(), f, ensure_ascii=False, indent=4)
                print(f'Dumped summarization response to {dump_file_path}')
        return response.choices[0].message.content