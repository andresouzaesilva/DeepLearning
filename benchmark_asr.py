import os
import time
import argparse
import pandas as pd
import jiwer
import sys
import librosa
from bert_score import score
import logging
import unicodedata
import re

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

try:
    from src.processors.vosk_engine import VoskTranscriber
    from src.processors.whisper_engine import WhisperTranscriber
    from src.processors.wav2vec_engine import Wav2VecTranscriber
except ImportError as e:
    print(e)
    sys.exit(1)

_re_spaces = re.compile(r"\s+")
def normalize_text_for_wer(text: str) -> str:
    """
    Normalização simples e segura para WER:
    - converte para str, lower
    - remove caracteres de controle
    - remove pontuação (mantém letras e dígitos)
    - normaliza acentuação (NFC)
    - colapsa espaços
    Retorna uma string pronta para passar ao jiwer.wer()
    """
    if text is None:
        return ""
    s = str(text).lower()
    # Normalize unicode (compose)
    s = unicodedata.normalize("NFC", s)
    # Remove control chars
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    # Remove punctuation: category starting with 'P'
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    # Replace non-breaking spaces etc -> normal space
    s = s.replace("\u00A0", " ")
    # Collapse multiple spaces
    s = _re_spaces.sub(" ", s).strip()
    return s

def normalize_text_for_cer(text: str) -> str:
    """
    Normalização leve para CER: lower + strip + collapse spaces + unicode normalize.
    We keep more characters so CER is measured on characters.
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = unicodedata.normalize("NFC", s)
    # Remove control chars
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    # Collapse spaces
    s = _re_spaces.sub(" ", s).strip()
    return s

def get_audio_duration(path):
    """Retorna a duração em segundos (librosa)."""
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def calcular_wer(ref: str, hyp: str) -> float:
    """Calcula WER via jiwer.wer após normalização segura."""
    if not ref or not hyp:
        return 1.0
    ref_n = normalize_text_for_wer(ref)
    hyp_n = normalize_text_for_wer(hyp)
    try:
        return float(jiwer.wer(ref_n, hyp_n))
    except TypeError:
        raise

def calcular_cer(ref: str, hyp: str) -> float:
    """Calcula CER via jiwer.cer após normalização leve."""
    if not ref or not hyp:
        return 1.0
    ref_n = normalize_text_for_cer(ref)
    hyp_n = normalize_text_for_cer(hyp)
    try:
        return float(jiwer.cer(ref_n, hyp_n))
    except AttributeError:
        ref_chars = " ".join(list(ref_n))
        hyp_chars = " ".join(list(hyp_n))
        return float(jiwer.wer(ref_chars, hyp_chars))

def main(audio_folder, csv_path, limit):
    if not os.path.exists(csv_path):
        print("CSV não encontrado:", csv_path)
        return

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(int(limit))

    print("Carregando modelos...")
    modelos = {}
    try:
        print("Carregando Vosk...")
        modelos["Vosk"] = VoskTranscriber(model_path="models/vosk-model-small-pt-0.3")
    except Exception as e:
        print("Falha ao carregar Vosk:", e)

    try:
        print("Carregando Whisper (base)...")
        modelos["Whisper"] = WhisperTranscriber(model_size="base")
    except Exception as e:
        print("Falha ao carregar Whisper:", e)

    try:
        print("Carregando Wav2Vec2 (pt-br)...")
        m_id = "models/wav2vec-pt-br" if os.path.exists("models/wav2vec-pt-br") else "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"
        modelos["Wav2Vec2"] = Wav2VecTranscriber(model_id=m_id)
    except Exception as e:
        print("Falha ao carregar Wav2Vec2:", e)

    print("Modelos carregados.\n")

    resultados = []
    total = len(df)

    for index, row in df.iterrows():
        full_csv_path = str(row.get("file_path", "")).strip()
        filename = os.path.basename(full_csv_path)
        if not filename:
            print(f"[{index+1}/{total}] Aviso: file_path vazio na linha {index}")
            continue

        abs_path = os.path.join(audio_folder, filename)
        ref = str(row.get("text", "")).strip()

        print(f"[{index+1}/{total}] {filename}...", end="\r", flush=True)

        if not os.path.exists(abs_path):
            print(f"\n[AVISO] Arquivo não encontrado: {abs_path}")
            continue

        duracao_audio = get_audio_duration(abs_path)
        if duracao_audio <= 0:
            print(f"\n[AVISO] Duração inválida (0): {abs_path}")
            continue

        for nome, engine in modelos.items():
            try:
                start = time.time()
                hyp = engine.transcribe(abs_path)
                tempo_proc = time.time() - start

                hyp = "" if hyp is None else str(hyp).strip()

                try:
                    wer = calcular_wer(ref, hyp)
                except Exception as e:
                    print(f"\nErro calculando WER para {nome} em {filename}: {e}")
                    wer = 1.0

                try:
                    cer = calcular_cer(ref, hyp)
                except Exception as e:
                    print(f"\nErro calculando CER para {nome} em {filename}: {e}")
                    cer = None

                rtf = tempo_proc / duracao_audio

                resultados.append({
                    "Arquivo": filename,
                    "Modelo": nome,
                    "Referencia": ref,
                    "Hipotese": hyp,
                    "WER": wer,
                    "CER": cer,
                    "RTF": rtf
                })

            except Exception as e:
                print(f"\nErro com modelo {nome} ao processar {filename}: {e}")
                continue

    if not resultados:
        print("Nenhum resultado foi gerado.")
        return

    df_res = pd.DataFrame(resultados)

    print("\nCalculando BERTScore...")
    candidatos = df_res["Hipotese"].fillna("").tolist()
    referencias = df_res["Referencia"].fillna("").tolist()

    try:
        _, _, F1 = score(candidatos, referencias, lang="pt", verbose=False)
        df_res["BERTScore"] = F1.numpy()
    except Exception as e:
        print("Falha ao calcular BERTScore:", e)
        df_res["BERTScore"] = 0.0

    agg_cols = ["WER", "CER", "RTF", "BERTScore"]
    final_df = df_res.groupby("Modelo")[agg_cols].mean().reset_index()

    exib = final_df.copy()
    exib["WER"] = (exib["WER"] * 100).map("{:.2f}%".format)
    exib["RTF"] = exib["RTF"].map("{:.4f}".format)
    exib["BERTScore"] = exib["BERTScore"].map("{:.4f}".format)
    exib["CER"] = exib["CER"].apply(lambda x: "{:.2f}%".format(x * 100) if pd.notna(x) else "N/A")

    print("\n" + "="*50)
    print(" RESULTADOS FINAIS")
    print("="*50)
    try:
        from tabulate import tabulate
        print(tabulate(exib, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(exib)

    out_csv = "metricas_finais.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nDados salvos em '{out_csv}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", required=True, help="Pasta onde estão os arquivos de áudio (cenário A).")
    parser.add_argument("--csv_path", required=True, help="CSV com coluna file_path e text.")
    parser.add_argument("--limit", type=int, default=None, help="Limitar número de linhas do CSV.")
    args = parser.parse_args()
    main(args.audio_folder, args.csv_path, args.limit)
