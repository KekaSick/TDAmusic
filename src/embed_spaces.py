"""
embed_spaces.py
---------------
Единый интерфейс извлечения покадровых эмбеддингов для ВСЕХ пространств.
Ключевой принцип: extract(audio, space) -> np.ndarray (T, d).
Downstream-код после этого space-agnostic.

ВАЖНО:
  * Все DL-модели грузятся ЛЕНИВО (один раз) и кэшируются в _MODELS.
  * Результат КЭШИРУЕТСЯ на диск (.npy). Извлечение дорого — второй раз
    его делать нельзя. Скрипт возобновляемый: пропускает готовое.
  * MuQ = SSL-модель (OpenMuQ/MuQ-large-msd-iter), НЕ MuQ-MuLan (та
    выдаёт один вектор на клип и для топологии бесполезна).
  * Никакой агрегации по времени здесь не делается — это убило бы топологию.
"""
from __future__ import annotations
import os

# Фиксируем HF_HOME, чтобы перенаправление HOME не сломало кэш transformers
os.environ["HF_HOME"] = "/Users/mverzhbitskiy/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Перенаправляем HOME в локальную папку кэша ПЕРЕД импортом любых библиотек, 
# чтобы tensorflow/keras при ленивой загрузке внутри transformers 
# не пытались читать/писать в недоступный ~/.keras/keras.json
_fake_home = os.path.join(os.getcwd(), "cache")
os.makedirs(os.path.join(_fake_home, ".keras"), exist_ok=True)
os.environ["HOME"] = _fake_home
os.environ["KERAS_HOME"] = os.path.join(_fake_home, ".keras")

import numpy as np
import torch
import librosa

_MODELS: dict = {}   # ленивый кэш загруженных моделей


# ------------------------------------------------------------------ MERT
def _extract_mert(wav, sr, cfg):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    key = "mert"
    if key not in _MODELS:
        mid = cfg["spaces"]["mert"]["model_id"]
        # Обход бага transformers в offline-режиме: передаем прямой путь к кэшу,
        # чтобы он даже не пытался лезть в сеть за model.safetensors.
        import os
        cache_path = "/Users/mverzhbitskiy/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
        if os.path.exists(cache_path):
            mid = cache_path
        print("MID IS:", mid)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        print("LOADING PROC", flush=True)
        proc = Wav2Vec2FeatureExtractor.from_pretrained(
            mid, trust_remote_code=True, local_files_only=True)
        print("LOADING MODEL", flush=True)
        model = AutoModel.from_pretrained(
            mid, trust_remote_code=True, local_files_only=True, use_safetensors=False).eval()
        print("MODEL LOADED", flush=True)
        _MODELS[key] = (proc, model.to(_device()))
    proc, model = _MODELS[key]
    print("PREPARING INPUTS (MANUAL)", flush=True)
    # Bypass Wav2Vec2FeatureExtractor to avoid mutex lock crash on macOS
    # Feature extractor just does: (x - mean) / sqrt(var + 1e-7)
    wav_norm = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
    inputs = {"input_values": torch.tensor(wav_norm, dtype=torch.float32).unsqueeze(0).to(_device())}
    
    print("INFERENCE", flush=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    print("EXTRACTING LAYER", flush=True)
    layer = cfg["spaces"]["mert"]["layer"]
    h = out.hidden_states[layer].squeeze(0)          # (T, 768)
    print("RETURNING", flush=True)
    return h.cpu().numpy()


# ------------------------------------------------------------------ MuQ
def _extract_muq(wav, sr, cfg):
    from muq import MuQ                              # pip install muq
    key = "muq"
    if key not in _MODELS:
        mid = cfg["spaces"]["muq"]["model_id"]
        _MODELS[key] = MuQ.from_pretrained(mid).to(_device()).eval()
    model = _MODELS[key]
    wavs = torch.tensor(wav).unsqueeze(0).to(_device())
    with torch.no_grad():
        out = model(wavs, output_hidden_states=True)
    layer = cfg["spaces"]["muq"]["layer"]
    h = out.hidden_states[layer].squeeze(0)          # (T, 1024)
    return h.cpu().numpy()


# ------------------------------------------------------------------ Encodec
def _extract_encodec(wav, sr, cfg):
    """Непрерывный латент ДО квантизации (encoder output), не дискретные коды."""
    from transformers import EncodecModel
    key = "encodec"
    if key not in _MODELS:
        mid = cfg["spaces"]["encodec"]["model_id"]
        model = EncodecModel.from_pretrained(mid).eval()
        _MODELS[key] = model.to(_device())
    model = _MODELS[key]
    # Bypass AutoProcessor to avoid macOS mutex lock crashes
    inputs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(_device())
    with torch.no_grad():
        # Непрерывный латент ДО RVQ-квантизации.
        # model.encoder() — прямой вызов энкодера, возвращает (B, dim, T).
        # НЕ model.encode() — тот прогоняет encoder + quantizer и отдаёт
        # дискретные коды (audio_codes), а не непрерывный латент.
        # Проверено: transformers 4.40, facebook/encodec_24khz,
        #   encoder output shape = (1, 128, 2250) для 30s@24kHz = 75 fps.
        latent = model.encoder(inputs)   # (B, 128, T)
    z = latent.squeeze(0).transpose(0, 1)            # (T, 128)
    return z.cpu().numpy()


# ------------------------------------------------------------------ MIR
def _extract_mir(wav, sr, cfg):
    """Ручные покадровые фичи. БЕЗ агрегации по времени."""
    hop = cfg["spaces"]["mir"]["hop_length"]
    feats = []
    for name in cfg["spaces"]["mir"]["features"]:
        if name == "chroma_cens":
            f = librosa.feature.chroma_cens(y=wav, sr=sr, hop_length=hop)   # (12, T)
        elif name == "spectral_centroid":
            f = librosa.feature.spectral_centroid(y=wav, sr=sr, hop_length=hop)
        elif name == "spectral_flatness":
            f = librosa.feature.spectral_flatness(y=wav, hop_length=hop)
        elif name == "spectral_bandwidth":
            f = librosa.feature.spectral_bandwidth(y=wav, sr=sr, hop_length=hop)
        else:
            raise ValueError(f"unknown MIR feature {name}")
        feats.append(f)
    stacked = np.concatenate(feats, axis=0).T        # (T, sum_dims)
    return stacked


_DISPATCH = {"mert": _extract_mert, "muq": _extract_muq,
             "encodec": _extract_encodec, "mir": _extract_mir}


def _device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract(filepath: str, space: str, cfg: dict) -> np.ndarray:
    """Главная функция. Кэширует на диск. Возвращает (T, d)."""
    cache_dir = os.path.join(cfg["paths"]["cache"], space)
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    cache_path = os.path.join(cache_dir, f"{base}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)                   # возобновляемость

    sr = cfg["data"]["sample_rate"]
    wav, _ = librosa.load(filepath, sr=sr, mono=True)
    arr = _DISPATCH[space](wav, sr, cfg)
    np.save(cache_path, arr)
    if _device() == "cuda":
        torch.cuda.empty_cache()                     # Encodec/MuQ прожорливы
    elif _device() == "mps":
        torch.mps.empty_cache()
    return arr


if __name__ == "__main__":
    import yaml, glob, sys
    from tqdm import tqdm
    cfg = yaml.safe_load(open("config.yaml"))
    spaces = sys.argv[1:] or list(cfg["spaces"].keys())
    files = sorted(glob.glob(os.path.join(cfg["paths"]["audio"], "**/*.*"),
                             recursive=True))
    for space in spaces:
        print(f"=== extracting {space} ({len(files)} files) ===")
        for fp in tqdm(files):
            try:
                extract(fp, space, cfg)
            except Exception as e:
                print(f"  FAIL {fp}: {e}")
