import os
from pydub import AudioSegment

def prepare_audio_for_vosk(input_path, output_path="temp_vosk.wav"):
    """Converte qualquer áudio para WAV Mono 16kHz (requisito do Vosk)"""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path