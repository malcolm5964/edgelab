#%% Import the required libraries

import sounddevice as sd  # For audio input/output using Python (easy and cross-platform)
# See: https://python-sounddevice.readthedocs.io/en/0.4.6/

import numpy as np  # For numerical processing and FFT computation
import matplotlib.pyplot as plt  # For real-time plotting of waveform and spectrum
import time  # For timing the FFT computation

#%% Parameters for audio and recording

BUFFER = 1024 * 16           # Number of samples per frame (chunk size). Larger = smoother but more delay
CHANNELS = 1                 # Mono recording (1 microphone)
RATE = 44100                 # Audio sample rate (samples per second)
RECORD_SECONDS = 30          # Duration of the recording session (in seconds)

#%% Setup matplotlib figure and line plots

# Create a figure with 2 vertically stacked subplots: waveform and frequency spectrum
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate X-axis values for waveform: sample indices (every 2 for 16-bit audio)
x = np.arange(0, 2 * BUFFER, 2)

# Generate X-axis values for frequency spectrum: frequency bins (up to Nyquist freq)
xf = np.fft.fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Initialize line plots with random data (placeholders)
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)            # Waveform
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)  # Spectrum

# === Configure waveform axis ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000)  # Adjust based on expected volume range
ax1.set_xlim(0, BUFFER)

# === Configure frequency spectrum axis ===
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)      # Adjust based on expected intensity
ax2.set_xlim(0, RATE / 2)  # Show up to Nyquist frequency

# Display the figure without blocking (so it can update in real-time)
plt.show(block=False)

#%% Start recording loop and update plots with real-time audio

exec_time = []  # List to record execution times for FFT processing

# Determine number of chunks to process based on total duration and buffer size
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Record a single chunk of audio ===
    # `blocking=True` ensures the data is fully captured before moving on
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
    data = np.squeeze(data)  # Remove unnecessary dimensions (shape: [BUFFER])

    # === Perform FFT to get frequency domain representation ===
    start_time = time.time()  # Start timer to measure FFT performance
    fft_data = np.fft.fft(data)  # Compute FFT of the audio chunk (complex output)
    fft_data = np.abs(fft_data[:BUFFER // 2])  # Get magnitude of positive frequencies only

    # Record FFT processing time
    exec_time.append(time.time() - start_time)

    # === Update the plots in real-time ===

    # Update waveform with new audio data
    line.set_ydata(data)

    # Update spectrum with scaled FFT magnitude
    line_fft.set_ydata(2.0 / BUFFER * fft_data)

    # Redraw the updated figure
    fig.canvas.draw()
    fig.canvas.flush_events()

#%% Post-recording summary and cleanup

print('stream stopped')  # Indicate end of recording
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))  # Print average FFT time
