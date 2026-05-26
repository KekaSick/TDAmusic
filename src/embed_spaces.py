"""
embed_spaces.py
---------------
Frame-level embedding extraction.

Each extractor returns an array with shape (T, d). The result is cached to disk
because model inference is the expensive step.
"""
from __future__ import annotations
import os

os.environ["HF_HOME"] = "/Users/mverzhbitskiy/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Keep libraries from writing into the user's global home during model loading.
_fake_home = os.path.join(os.getcwd(), "cache")
os.makedirs(os.path.join(_fake_home, ".keras"), exist_ok=True)
os.environ["HOME"] = _fake_home
os.environ["KERAS_HOME"] = os.path.join(_fake_home, ".keras")

import numpy as np
import torch
import librosa

_MODELS: dict = {}


# MERT
def _extract_mert(wav, sr, cfg):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    key = "mert"
    if key not in _MODELS:
        mid = cfg["spaces"]["mert"]["model_id"]
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
    # Avoid Wav2Vec2FeatureExtractor mutex crashes observed on macOS.
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


# MuQ
def _extract_muq(wav, sr, cfg):
    from muq import MuQ
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


# EnCodec
def _extract_encodec(wav, sr, cfg):
    """Return the continuous encoder latent before RVQ quantization."""
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
        latent = model.encoder(inputs)   # (B, 128, T)
    z = latent.squeeze(0).transpose(0, 1)            # (T, 128)
    return z.cpu().numpy()


# MIR
def _extract_mir(wav, sr, cfg):
    """Extract handcrafted frame-level MIR features without time pooling."""
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
    """Extract or load a cached frame-level representation."""
    cache_dir = os.path.join(cfg["paths"]["cache"], space)
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]
    cache_path = os.path.join(cache_dir, f"{base}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    sr = cfg["data"]["sample_rate"]
    wav, _ = librosa.load(filepath, sr=sr, mono=True)
    arr = _DISPATCH[space](wav, sr, cfg)
    np.save(cache_path, arr)
    if _device() == "cuda":
        torch.cuda.empty_cache()
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
