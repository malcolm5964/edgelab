#%% Import the required libraries

import numpy as np                      # For numerical operations
import matplotlib.pyplot as plt         # For visualizing plots
import librosa                          # Core library for audio feature extraction
import librosa.display                  # For advanced audio-specific plotting

#%% Load an audio file

# Load audio from 'test.wav'. sr=None keeps the original sample rate of the file
y, sr = librosa.load("test.wav", sr=None)

# y: audio time series (1D NumPy array)
# sr: sampling rate of the loaded audio

#%% Compute the Spectrogram

# Compute the complex Short-Time Fourier Transform (STFT)
# `S_full`: magnitude of spectrogram
# `phase`: phase component (discarded in visualizations)
S_full, phase = librosa.magphase(librosa.stft(y))

#%% Plot Waveform and Spectrogram (Time-Frequency Representation)

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# === Time-domain waveform plot ===
ax1.plot(y)
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set(title='Time Series')

# === Spectrogram plot ===
# Convert amplitude to decibels and display using logarithmic frequency scale
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=ax2)
fig.colorbar(img, ax=ax2)
ax2.set(title='Spectrogram')

# Show both plots
plt.show()

#%% Chroma Estimation (Pitch Class Energy over Time)

# Chroma represents intensity for each of 12 pitch classes (C, C#, ..., B)
# Useful for chord detection, harmony analysis, key estimation

# Compute power spectrogram with large FFT window
S = np.abs(librosa.stft(y, n_fft=4096))**2

# Extract chromagram from power spectrogram
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

# Plot power spectrogram and chroma
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Power spectrogram ===
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()
ax[0].set(title='Power Spectrogram')

# === Chroma plot ===
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='Chromogram')

plt.show()

#%% Compute Mel-Spectrogram

# Mel spectrogram compresses frequency axis to match human perception (Mel scale)
# Common in audio classification, music/speech modeling

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to dB scale for visualization
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)

# Plot Mel spectrogram
fig, ax = plt.subplots()
img = librosa.display.specshow(S_mel_dB, x_axis='time',
                               y_axis='mel', sr=sr,
                               fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

#%% Compute MFCC (Mel-Frequency Cepstral Coefficients)

# MFCCs represent short-term spectral shape and are widely used in speech recognition

# Compute MFCCs from audio signal (default 20, here we use 40)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Also compute mel spectrogram for reference in same plot
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Plot both Mel spectrogram and MFCCs
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Mel spectrogram ===
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()

# === MFCC plot ===
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

plt.show()


'''
Use Cases:
- Speech recognition and emotion detection
- Music genre classification
- Audio fingerprinting
- Deep learning audio preprocessing

Possible Additions:
- Add pitch estimation: librosa.yin()
- Add tempo estimation: librosa.beat.tempo()
- Save features as .npy or .csv for training ML models
- Segment features using windowing for real-time classification
'''