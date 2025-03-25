#%% Import the required libraries

import sounddevice as sd  # For real-time audio input/output from microphone
# See: https://python-sounddevice.readthedocs.io/en/0.4.6/

import numpy as np  # For numerical processing of audio data
import matplotlib.pyplot as plt  # For real-time plotting of waveforms
import time  # For measuring time taken to process each frame

from scipy.signal import butter, sosfilt  # For creating and applying digital bandpass filters
# See: https://docs.scipy.org/doc/scipy/reference/signal.html

#%% Parameters for audio recording and buffer

BUFFER = 1024 * 16           # Number of samples to capture per frame
CHANNELS = 1                 # Mono recording (1 microphone)
RATE = 44100                 # Audio sample rate in Hz (CD quality)
RECORD_SECONDS = 20          # Total recording duration in seconds

#%% Set up Matplotlib figure and live plots

# Create a figure with two stacked subplots: raw audio and filtered audio
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate x-axis values (sample indices) for time-domain plots
x = np.arange(0, 2 * BUFFER, 2)

# Initialize both plots with random placeholder data
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)         # Plot for raw audio
line_filter, = ax2.plot(x, np.random.rand(BUFFER), '-', lw=2)  # Plot for filtered audio

# === Configure raw waveform plot ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')
ax1.set_ylim(-5000, 5000)   # Y-axis range for amplitude
ax1.set_xlim(0, BUFFER)     # X-axis range for number of samples

# === Configure filtered waveform plot ===
ax2.set_title('FILTERED')
ax2.set_xlabel('samples')
ax2.set_ylabel('amplitude')
ax2.set_ylim(-5000, 5000)
ax2.set_xlim(0, BUFFER)

# Display the figure window without blocking code execution
plt.show(block=False)

#%% Define a bandpass filter using Butterworth design

def design_filter(lowfreq, highfreq, fs, order=3):
    """
    Creates a digital bandpass filter using second-order sections (for stability).
    
    Parameters:
        lowfreq (float): Lower cutoff frequency in Hz
        highfreq (float): Upper cutoff frequency in Hz
        fs (int): Sampling rate in Hz
        order (int): Filter order (higher = sharper but more processing)
        
    Returns:
        sos (ndarray): Second-order section representation of filter
    """
    nyq = 0.5 * fs               # Nyquist frequency = half of sampling rate
    low = lowfreq / nyq          # Normalize low frequency
    high = highfreq / nyq        # Normalize high frequency
    sos = butter(order, [low, high], btype='band', output='sos')  # Create bandpass filter
    return sos

# Design a bandpass filter to isolate 19.4–19.6 kHz (high-frequency narrowband)
# ⚠️ Make sure filter sample rate (fs) matches RATE
sos = design_filter(19400, 19600, 44100, 3)

#%% Real-time audio processing loop

exec_time = []  # To store FFT/filter processing time per frame

# Calculate number of frames to capture for the full duration
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):   

    # === Record a chunk of audio ===
    # Blocking = True: waits until full BUFFER is recorded
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)

    # Remove singleton dimension (e.g., shape [BUFFER, 1] → [BUFFER])
    data = np.squeeze(data)

    # === Apply the bandpass filter ===
    start_time = time.time()         # Record start time
    yf = sosfilt(sos, data)          # Filter the audio using the designed SOS filter
    exec_time.append(time.time() - start_time)  # Log the execution time

    # === Update the plots ===
    line.set_ydata(data)             # Update raw waveform plot
    line_filter.set_ydata(yf)        # Update filtered waveform plot
    fig.canvas.draw()                # Redraw the figure with new data
    fig.canvas.flush_events()        # Push GUI updates immediately

#%% Cleanup and report

print('stream stopped')  # Notify user that capture ended
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))  # Performance summary
