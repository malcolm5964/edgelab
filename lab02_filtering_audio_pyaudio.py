#%% Import the required libraries

import pyaudio  # For real-time audio input/output from microphone
# See: https://people.csail.mit.edu/hubert/pyaudio/

import struct   # For converting byte stream data into Python int16 format
# See: https://docs.python.org/3/library/struct.html

import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For real-time plotting
from scipy.signal import butter, sosfilt  # For digital bandpass filter design and application
# See: https://docs.scipy.org/doc/scipy/reference/signal.html

import time  # For measuring filter execution time

#%% Audio and buffer parameters

BUFFER = 1024 * 16           # Number of samples per frame (chunk size)
FORMAT = pyaudio.paInt16     # 16-bit PCM format
CHANNELS = 1                 # Mono audio input
RATE = 44100                 # Sampling rate in Hz
RECORD_SECONDS = 20          # Duration of real-time processing session (in seconds)

#%% Create the initial plot for waveform and filtered output

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))  # Two vertically stacked plots

# X-axis: sample indices for waveform and filtered plots
x = np.arange(0, 2 * BUFFER, 2)

# Create placeholder lines with random values
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Raw waveform plot
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)  # Filtered signal plot

# === Configure waveform plot ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')
ax1.set_ylim(-5000, 5000)   # Amplitude range for display
ax1.set_xlim(0, BUFFER)

# === Configure filtered signal plot ===
ax2.set_title('FILTERED')
ax2.set_xlabel('samples')
ax2.set_ylabel('amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Show the plot window non-blocking (updates in real time)
plt.show(block=False)

#%% Function to design a digital bandpass filter

def design_filter(lowfreq, highfreq, fs, order=3):
    nyq = 0.5 * fs                  # Nyquist frequency = half of sample rate
    low = lowfreq / nyq            # Normalize low cutoff frequency
    high = highfreq / nyq          # Normalize high cutoff frequency
    sos = butter(order, [low, high], btype='band', output='sos')  # Second-order sections (stable)
    return sos

# Design the bandpass filter (e.g., between 19.4 kHz to 19.6 kHz)
# NOTE: RATE must match the design fs = 48000, or change design to match 44100
# lowfreq = 19400 || Lower cutoff, frequencies below this are blocked.
# highfreq = 19600 || Upper cutoff, frequencies above this are blocked.
# fs = 48000 || Sampling rate of the signal in Hz. Used to normalise the filter design.
# order = 3 || Filter order, controls how steeply frequencies are attenuated outside the passband.
sos = design_filter(19400, 19600, 48000, 3)

#%% Set up the PyAudio stream for real-time microphone input

audio = pyaudio.PyAudio()  # Initialize PyAudio

# Open audio input/output stream
stream = audio.open(
    format=FORMAT,             # 16-bit PCM
    channels=CHANNELS,         # Mono
    rate=RATE,                 # Sample rate
    input=True,                # Enable input (mic)
    output=True,               # Enable output (for monitoring, optional)
    frames_per_buffer=BUFFER   # Chunk size
)

print('stream started')

# List to store processing time per frame for performance profiling
exec_time = []

# Loop through frames for the duration of the recording session
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Read a chunk of audio data ===
    data = stream.read(BUFFER)  # Read raw bytes from microphone

    # === Convert binary data to NumPy int16 array ===
    data_int = struct.unpack(str(BUFFER) + 'h', data)  # Unpack BUFFER samples of 'h' (short int)
    
    # === Apply bandpass filter ===
    start_time = time.time()     # Start timer
    yf = sosfilt(sos, data_int)  # Apply the second-order sections filter
    exec_time.append(time.time() - start_time)  # Log execution time

    # === Update plots with real-time data ===
    line.set_ydata(data_int)     # Raw waveform
    line_filter.set_ydata(yf)    # Filtered signal
    fig.canvas.draw()            # Redraw the figure
    fig.canvas.flush_events()    # Ensure GUI updates

# Terminate audio stream after recording is complete
audio.terminate()

print('stream stopped')
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))
