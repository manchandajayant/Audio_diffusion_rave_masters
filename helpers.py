import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
import argparse
import warnings
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import IPython.display as ipd


TARGET_SR = 44100
CLIP_DURATION_SECONDS = 5.0
TARGET_SAMPLES = int(TARGET_SR * CLIP_DURATION_SECONDS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_view_audio(out_file, sr=44100):
# Load saved audio
    sr = int(sr)
    if sr > 65535:
        sr = 44100
    audio_data, sr = sf.read(out_file)
    
    # If stereo, convert to mono for plotting
    if audio_data.ndim > 1:
        audio_mono = librosa.to_mono(audio_data.T)
    else:
        audio_mono = audio_data
    
    # Plot waveform
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio_mono, sr=sr)
    plt.title(f"Waveform of {out_file}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    
    # Play audio (in notebook)
    # ipd.display(ipd.Audio(audio_data, rate=44100))
    ipd.display(ipd.Audio(out_file))


def load_audio_to_numpy(audio_path):
    """
    Load stereo audio and pad/crop to TARGET_AUDIO_LEN samples.
    """
    audio, sr = sf.read(audio_path)  # stereo: shape (n_samples, 2)

    # Ensure stereo shape
    if audio.ndim == 1:
        raise ValueError("Input audio is mono. Expected stereo.")

    # Normalize to [-1, 1] range if not already
    if audio.dtype != np.float32:
        audio = audio / np.iinfo(audio.dtype).max
    audio = np.clip(audio, -1.0, 1.0)

    # Trim or pad
    if audio.shape[0] > TARGET_AUDIO_LEN:
        audio = audio[:TARGET_AUDIO_LEN, :]
    elif audio.shape[0] < TARGET_AUDIO_LEN:
        pad_width = TARGET_AUDIO_LEN - audio.shape[0]
        audio = np.pad(audio, ((0, pad_width), (0, 0)), mode="constant")

    return audio

# Helpers
def load_audio_to_numpy(path, sr=44100):
    wav, fs = sf.read(path)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=1)
    if fs != sr:
        import librosa
        wav = librosa.resample(wav.T, fs, sr).T
    L = wav.shape[0]
    if L < TARGET_SAMPLES:
        wav = np.pad(wav, ((0, TARGET_SAMPLES-L), (0,0)))
    else:
        wav = wav[:TARGET_SAMPLES]
    return wav.astype(np.float32)

def load_audio(path):
    arr = load_audio_to_numpy(path)
    return torch.from_numpy(arr).permute(1,0).unsqueeze(0).to(device)

def get_audio_files(folder):
    exts = (".wav", ".flac", ".mp3")
    return [os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.lower().endswith(exts)]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Function copied from diffusers.models.embeddings
def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
    use_real_unbind_dim: int = -2,
) -> torch.Tensor:
    if use_real:
        x_a = x.float()
        x_b = freqs_cis[0].squeeze(use_real_unbind_dim)
        x_c = freqs_cis[1].squeeze(use_real_unbind_dim)
        # the below is a simplification of complex multiplication, real and imag parts are separated
        x_out = (x_a * x_b) + (rotate_half(x_a) * x_c)
        return x_out.type_as(x)
    # in case of using complex tensors, the below is used
    x = x.float()
    freqs_cis = freqs_cis.squeeze(2)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_reshaped = torch.view_as_complex(x_reshaped)
    freqs_cis = torch.view_as_complex(freqs_cis)
    x_out = x_reshaped * freqs_cis
    x_out = torch.view_as_real(x_out)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)