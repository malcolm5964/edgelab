#!/usr/bin/env python3
# Refer to: https://github.com/Uberi/speech_recognition
# This is a demonstration for speech recognition.
# You should speak clearly when prompted: "Say something"

#%% Import required libraries

import speech_recognition as sr  # Core library for speech-to-text
import time                      # For measuring how long recognition takes
import os                        # For clearing the terminal (optional)

#%% Recording from Microphone

# Create a Recognizer instance to handle speech recognition
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    # Adjust the recognizer to ignore ambient noise (e.g., fan, background chatter)
    # You should stay silent for 1-2 seconds during this
    r.adjust_for_ambient_noise(source)

    # Clear the terminal (for Unix-based systems, optional)
    os.system('clear')

    print("Say something!")  # Prompt the user to speak

    # Record audio from the microphone
    audio = r.listen(source)  # Blocks until speech is detected and ends automatically

#%% Recognize Speech using Google Web API (Online)

start_time = time.time()  # Start timer to measure recognition time

try:
    # Perform recognition using Google Web Speech API
    # This is an **online service** and requires an internet connection

    # If you have a Google API key, you can provide it:
    # r.recognize_google(audio, key="YOUR_API_KEY")

    # Convert audio to text
    print("Google Speech Recognition thinks you said: " + r.recognize_google(audio))

except sr.UnknownValueError:
    # Raised when speech is unintelligible (e.g., mumbling, noise)
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    # Raised when there's a problem connecting to the API
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Display how long the Google recognition took
print('Time for Google Speech Recognition = {:.0f} seconds'.format(time.time() - start_time))

#%% Recognize Speech using PocketSphinx (Offline)

start_time = time.time()  # Start timer again for Sphinx

try:
    # Perform recognition using CMU Sphinx (fully offline)
    # You must have PocketSphinx installed: `pip install pocketsphinx`
    print("Sphinx thinks you said: " + r.recognize_sphinx(audio))

except sr.UnknownValueError:
    # Raised when Sphinx cannot interpret the spoken words
    print("Sphinx could not understand audio")

except sr.RequestError as e:
    # Raised if Sphinx backend fails
    print("Sphinx error: {0}".format(e))

# Display how long the Sphinx recognition took
print('Time for Sphinx recognition = {:.0f} seconds'.format(time.time() - start_time))

# =================================================================

'''
    Metrics     Google API  vs. Sphinx
    Internet    Yes             No
    Accuracy    High            Lower
    Speed       Depends         Fast

- Google API good for production demos.
- Sphinx is a great fallback option / offline mode.
'''