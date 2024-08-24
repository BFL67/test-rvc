import os, sys
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
import wget
import subprocess
from pydub import AudioSegment
import tempfile
from torch import nn

import logging
from transformers import HubertModel

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(now_dir)

base_path = os.path.join(now_dir, "rvc", "models", "formant", "stftpitchshift")
stft = base_path + ".exe" if sys.platform == "win32" else base_path


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file, sample_rate, formant_shifting, formant_qfrency, formant_timbre
):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File not found: {file}")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        if formant_shifting:
            audio = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
                audio_segment.export(temp_file_path, format="wav")

            command = [
                stft,
                "-i",
                temp_file_path,
                "-q",
                str(formant_qfrency),
                "-t",
                str(formant_timbre),
                "-o",
                f"{temp_file_path}_formatted.wav",
            ]
            subprocess.run(command, shell=True)
            formatted_audio_path = f"{temp_file_path}_formatted.wav"
            audio, sr = sf.read(formatted_audio_path)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.T)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")
    return audio.flatten()


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec_base"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }

    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/contentvec_base/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/korean_hubert_base/pytorch_model.bin",
    }

    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/contentvec_base/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/new_embedders/korean_hubert_base/config.json",
    }

    if embedder_model == "custom":
        model_path = custom_embedder
        if not custom_embedder and os.path.exists(custom_embedder):
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, 'pytorch_model.bin')
        json_file = os.path.join(model_path, 'config.json')
        if not os.path.exists(bin_file) or not os.path.exists(json_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)


    if custom_embedder:
        models = HubertModelWithFinalProj.from_pretrained(os.path.join(embedder_root, custom_embedder))
    else:
        models = HubertModelWithFinalProj.from_pretrained(model_path)

    return models
