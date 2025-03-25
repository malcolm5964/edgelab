#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2  # Import OpenCV for image capture and processing

# Initialize the webcam (device index 0 = default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is successfully opened
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Capture a single frame from the webcam
ret, frame = cap.read()  # `ret` is a boolean indicating success, `frame` is the captured image

# Save the captured frame as an image file
cv2.imwrite('captured_image.jpg', frame)  # Writes the frame to a JPEG file in the current directory

# Release the webcam resource
cap.release()

# Print confirmation message
print("Image captured and saved!")
