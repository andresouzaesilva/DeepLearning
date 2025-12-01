from transformers import pipeline
import os
from .base import Transcriber

class Wav2VecTranscriber(Transcriber):
    def __init__(self, model_id="models/wav2vec-pt-br"):

        if not os.path.exists(model_id):
             print(f"Pasta local '{model_id}' não encontrada. Tentando download online...")
             model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"

        print(f"Carregando Wav2Vec2 ({model_id})...")
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=30)

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_path}")
        
        result = self.pipe(audio_path)
        
        return result.get("text", "")