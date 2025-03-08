import os
import numpy as np
import librosa
import pandas as pd
import parselmouth
import pysptk

base_directory = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//supershort_database"

# Function to estimate F0 using SWIPE' (via Parselmouth)
def extract_f0_parselmouth(filepath):
    snd = parselmouth.Sound(filepath)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  # Remove unvoiced frames
    relF0SD = np.std(f0) / np.mean(f0) if np.mean(f0) != 0 else 0
    jitter_3 = compute_jitter_apq(f0, 3)
    jitter_5 = compute_jitter_apq(f0, 5)
    jitter_11 = compute_jitter_apq(f0, 11)
    return  relF0SD, jitter_3, jitter_5, jitter_11

# Function to estimate F0 using YIN (Librosa)
def extract_f0_yin(y, sr):
    f0 = librosa.yin(y, fmin=75, fmax=300)
    f0 = f0[~np.isnan(f0)]
    relF0SD = np.std(f0) / np.mean(f0) if np.mean(f0) != 0 else 0
    jitter_3 = compute_jitter_apq(f0, 3)
    jitter_5 = compute_jitter_apq(f0, 5)
    jitter_11 = compute_jitter_apq(f0, 11)
    return  relF0SD, jitter_3, jitter_5, jitter_11

# Function to estimate F0 using pYIN (Librosa)
def extract_f0_pyin(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=300)
    f0 = f0[~np.isnan(f0)]
    relF0SD = np.std(f0) / np.mean(f0) if np.mean(f0) != 0 else 0
    jitter_3 = compute_jitter_apq(f0, 3)
    jitter_5 = compute_jitter_apq(f0, 5)
    jitter_11 = compute_jitter_apq(f0, 11)
    return  relF0SD, jitter_3, jitter_5, jitter_11

# Function to estimate F0 using RAPT (TorchAudio)
#
#def extract_f0_rapt(y, sr):
    # Use librosa's piptrack for pitch tracking (this mimics RAPT's functionality)
 #   D = np.abs(librosa.stft(y))
  #  pitches, magnitudes = librosa.core.piptrack(S=D, sr=sr)
  #  pitch = []
  #  for t in range(pitches.shape[1]):
  #      index = magnitudes[:, t].argmax()
  #      pitch.append(pitches[index, t])
  #  pitch = np.array(pitch)
  #  pitch = pitch[pitch > 0]
  #  relF0SD = np.std(pitch) / np.mean(pitch) if np.mean(pitch) != 0 else 0
  #  jitter_3 = compute_jitter_apq(pitch, 3)
  #  jitter_5 = compute_jitter_apq(pitch, 5)
  #  jitter_11 = compute_jitter_apq(pitch, 11)
  #  return  relF0SD, jitter_3, jitter_5, jitter_11
def extract_f0_rapt(y, sr):
    # Convert audio to the format required by pysptk (single-channel float64)
    y = librosa.util.normalize(y)* 32768   # Normalize audio for consistency

    # Frame shift in samples (default is 5 ms for RAPT, meaning 0.005 * sr)
    hopsize = int(0.005 * sr)

    # Compute pitch using pysptk's RAPT
    f0 = pysptk.sptk.rapt(y.astype(np.float32), sr, hopsize=hopsize, min=50, max=500, voice_bias=0.0, otype=1)

    # Remove zero values (unvoiced frames)
    f0 = f0[f0 > 0]

    # Calculate relative F0 SD
    relF0SD = np.std(f0) / np.mean(f0) if np.mean(f0) != 0 else 0

    # Compute jitter measures
    jitter_3 = compute_jitter_apq(f0, 3)
    jitter_5 = compute_jitter_apq(f0, 5)
    jitter_11 = compute_jitter_apq(f0, 11)

    return relF0SD, jitter_3, jitter_5, jitter_11

# Function to estimate F0 using pysptk's SWIPE'
def extract_f0_swipe(y, sr):
    # Normalize audio for numerical stability
    y = librosa.util.normalize(y)

    # Frame shift (default 5 ms for SWIPE')
    hop_size = int(0.005 * sr)

    # Compute F0 using pysptk's SWIPE'
    f0 = pysptk.sptk.swipe(y.astype(np.float64), sr, hopsize=hop_size, min=50, max=500, threshold=0.3, otype='f0')

    # Remove zero values (unvoiced frames)
    f0 = f0[f0 > 0]

    # Compute relative F0 standard deviation
    relF0SD = np.std(f0) / np.mean(f0) if np.mean(f0) != 0 else 0

    # Compute Jitter measures
    jitter_3 = compute_jitter_apq(f0, 3)
    jitter_5 = compute_jitter_apq(f0, 5)
    jitter_11 = compute_jitter_apq(f0, 11)

    return relF0SD, jitter_3, jitter_5, jitter_11
# Function to compute Jitter(APQ)
def compute_jitter_apq(f0, n_points):
    if len(f0) < n_points:
        return 0
    # Create an array to hold the APQ values
    jitter_values = []
    for i in range(len(f0) - n_points):
        avg_f0 = np.mean(f0[i:i+n_points])  # Average of the n_points f0 values
        jitter_values.append(np.abs(f0[i+n_points] - avg_f0))  # Jitter is the difference from the average
    return np.mean(jitter_values)

# Process files in all nested folders
data_parselmouth, data_yin, data_pyin, data_rapt, data_swipe = [], [], [], [], []

for root, _, files in os.walk(base_directory):
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            folder_name = os.path.basename(root)  # Extract person ID (P... or K...)
            label = 1 if folder_name.startswith("P") else 0  # 1 for Parkinson’s, 0 for control

            # Load audio
            y, sr = librosa.load(filepath, sr=None)

            # Extract features for each pitch tracking method
            data_parselmouth.append([*extract_f0_parselmouth(filepath), label])
            data_yin.append([*extract_f0_yin(y, sr), label])
            data_pyin.append([*extract_f0_pyin(y, sr), label])
            data_rapt.append([*extract_f0_rapt(y, sr), label])
            data_swipe.append([*extract_f0_swipe(y, sr), label])

# Create DataFrames
columns = [ "relF0SD", "Jitter(APQ3)", "Jitter(APQ5)", "Jitter(APQ11)", "Label"]
df_parselmouth = pd.DataFrame(data_parselmouth, columns=columns)
df_yin = pd.DataFrame(data_yin, columns=columns)
df_pyin = pd.DataFrame(data_pyin, columns=columns)
df_rapt = pd.DataFrame(data_rapt, columns=columns)
df_swipe = pd.DataFrame(data_swipe, columns=columns)

# Save to CSV
df_parselmouth.to_csv("features_parselmouth_rel_supershort.csv", index=False)
df_yin.to_csv("features_yin_rel_supershort.csv", index=False)
df_pyin.to_csv("features_pyin_rel_supershort.csv", index=False)
df_rapt.to_csv("features_rapt_rel_supershort.csv", index=False)
df_swipe.to_csv("features_swipe_rel_supershort.csv", index=False)

print("Feature extraction completed and saved to CSV files.")
