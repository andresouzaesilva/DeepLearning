import whisper
from .base import Transcriber

class WhisperTranscriber(Transcriber):
    def __init__(self, model_size="base"):
        # O modelo é carregado apenas uma vez ao instanciar
        print(f"Carregando Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, language="pt")
        return result["text"]