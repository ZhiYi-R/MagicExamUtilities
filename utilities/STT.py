import os
import json
import requests
from pathlib import Path

class SpeechToText:
    def __init__(self, dump_dir: Path = Path('.'), dump_stt_response: bool = True) -> None:
        self.__dump_dir = dump_dir
        self.__dump_stt_response = dump_stt_response
    
    def stt(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f'Audio {audio_path} does not exist')
        print(f'STTing audio: {audio_path}')
        with open(audio_path, 'rb') as f:
            response = requests.post(
                url = os.environ['ASR_API_URL'],
                headers = {
                    'Authorization': f'Bearer {os.environ["ASR_API_KEY"]}',
                },
                files = {
                    'file': (audio_path.name, f.read(), 'audio/mpeg'),
                    'model': (None, os.environ['ASR_MODEL'])
                }
            )
            if response.status_code != 200:
                raise RuntimeError(f'STT failed for audio {audio_path}, status code: {response.status_code}, response: {response.text}')
            if self.__dump_stt_response:
                dump_file_path = self.__dump_dir.joinpath(f'{audio_path.stem}.json')
                with open(dump_file_path, 'w') as f:
                    json.dump(response.json(), f, ensure_ascii=False, indent=4)
                    print(f'Dumped STT response to {dump_file_path}')
            if not response.json()['text']:
                raise RuntimeError(f'No text returned from ASR API for audio {audio_path}')
            return response.json()['text']
