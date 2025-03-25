# For the code to work, the open-source Haar Cascade XML model must be downloaded.
# Download from:
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
# Place it in the same folder as this script.

import cv2  # OpenCV library for computer vision and webcam processing

#%% Initialize Haar Cascade Face Detector

# Path to the Haar Cascade classifier file for detecting frontal faces
haarcascade = "haarcascade_frontalface_alt2.xml"

# Load the Haar Cascade model
detector = cv2.CascadeClassifier(haarcascade)

#%% Start Webcam Video Capture

# Open the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Validate webcam connection
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main Frame Processing Loop

# This loop will run until 'q' is pressed
while True:
    try:
        # Capture a single frame from the webcam
        ret, frame = cap.read()

        # OPTIONAL: Resize the image to a smaller size for faster processing
        # This helps speed up face detection
        frame = cv2.resize(frame, (256, 256))  # Try commenting this line to compare speed

        # Convert the color image to grayscale
        # Haar Cascades work on grayscale images for efficiency and simplicity
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        # Returns a list of bounding boxes (x, y, width, height) for each detected face
        faces = detector.detectMultiScale(image_gray)

        # Loop through each detected face
        for face in faces:
            (x, y, w, d) = face

            # Draw a white rectangle around the detected face on the original frame
            # Parameters: top-left corner, bottom-right corner, color (white), thickness
            cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)

        # OPTIONAL: Resize the frame to display a larger window (720x720 pixels)
        frame = cv2.resize(frame, (720, 720))

        # Show the processed frame in a window titled "frame"
        cv2.imshow("frame", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow exit using Ctrl+C
        break

#%% Cleanup After Exit

cap.release()              # Release the webcam
cv2.destroyAllWindows()    # Close any OpenCV GUI windows

'''
detectMultiScale() Parameters

faces = detector.detectMultiScale(
    image_gray,
    scaleFactor=1.1,       # How much the image size is reduced at each scale (lower = slower, better)
    minNeighbors=5,        # How many neighbors each rectangle should have to retain it
    minSize=(30, 30),      # Minimum face size to detect
    flags=cv2.CASCADE_SCALE_IMAGE
)
'''