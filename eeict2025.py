import os
import numpy as np
import librosa
import pandas as pd
import parselmouth
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.signal import correlate
#import torch
base_directory = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//recordings-20250306T192251Z-001"

# Function to estimate F0 using SWIPE' (via Parselmouth)
def extract_f0_swipe(filepath):
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  # Remove unvoiced frames
    return np.mean(f0), np.std(f0), compute_jitter(f0)

# Function to estimate F0 using YIN (Librosa)
def extract_f0_yin(y, sr):
    f0 = librosa.yin(y, fmin=75, fmax=300)
    f0 = f0[~np.isnan(f0)]
    return np.mean(f0), np.std(f0), compute_jitter(f0)

# Function to estimate F0 using pYIN (Librosa)
def extract_f0_pyin(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=300)
    f0 = f0[~np.isnan(f0)]
    return np.mean(f0), np.std(f0), compute_jitter(f0)

# Function to estimate F0 using RAPT (TorchAudio)
def extract_f0_rapt(y, sr):
    # Use librosa's piptrack for pitch tracking (this mimics RAPT's functionality)
    D = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.core.piptrack(S=D, sr=sr)
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])
    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]
    return np.mean(pitch), np.std(pitch), compute_jitter(pitch)
# Function to compute jitter
def compute_jitter(f0):
    if len(f0) < 2:
        return 0
    diff_f0 = np.abs(np.diff(f0))
    jitter = np.mean(diff_f0) / np.mean(f0)
    return jitter

# Process files in all nested folders
data_swipe, data_yin, data_pyin, data_rapt = [], [], [], []

for root, _, files in os.walk(base_directory):
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            folder_name = os.path.basename(root)  # Extract person ID (P... or K...)
            label = 1 if folder_name.startswith("P") else 0  # 1 for Parkinson’s, 0 for control

            # Load audio
            y, sr = librosa.load(filepath, sr=None)

            # Extract features for each pitch tracking method
            data_swipe.append([*extract_f0_swipe(filepath), label])
            data_yin.append([*extract_f0_yin(y, sr), label])
            data_pyin.append([*extract_f0_pyin(y, sr), label])
            data_rapt.append([*extract_f0_rapt(y, sr), label])

# Create DataFrames
columns = ["F0_mean", "F0_std", "Jitter", "Label"]
df_swipe = pd.DataFrame(data_swipe, columns=columns)
df_yin = pd.DataFrame(data_yin, columns=columns)
df_pyin = pd.DataFrame(data_pyin, columns=columns)
df_rapt = pd.DataFrame(data_rapt, columns=columns)

# Save to CSV
df_swipe.to_csv("features_swipe.csv", index=False)
df_yin.to_csv("features_yin.csv", index=False)
df_pyin.to_csv("features_pyin.csv", index=False)
df_rapt.to_csv("features_rapt.csv", index=False)
