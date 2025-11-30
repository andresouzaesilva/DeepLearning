import os
import wave
import json
from vosk import Model, KaldiRecognizer
from .base import Transcriber
from src.utils import prepare_audio_for_vosk

class VoskTranscriber(Transcriber):
    def __init__(self, model_path="models/vosk-model-small-pt-0.3"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo Vosk não encontrado em: {model_path}. "
                "Baixe em https://alphacephei.com/vosk/models e extraia na pasta models/"
            )
        print("Carregando Vosk...")
        self.model = Model(model_path)

    def transcribe(self, audio_path: str) -> str:
        # Converter áudio para formato aceito pelo Vosk
        wav_path = prepare_audio_for_vosk(audio_path)
        
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())
        
        final_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                final_text += res.get("text", "") + " "
        
        res = json.loads(rec.FinalResult())
        final_text += res.get("text", "")
        
        # Limpar arquivo temporário
        wf.close()
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return final_text.strip()