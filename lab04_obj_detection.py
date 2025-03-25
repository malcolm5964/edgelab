#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi
# Download lightweight TFLite EfficientDet model
# wget -q -O efficientdet.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

import cv2
import mediapipe as mp
import time

# Import MediaPipe's Python task wrappers
from mediapipe.tasks import python  # Python base API for task setup
from mediapipe.tasks.python import vision  # Vision-specific modules (object detection, etc.)

#%% Parameters
maxResults = 5  # Maximum number of objects to detect in a frame
scoreThreshold = 0.25  # Minimum confidence score to consider a detection valid
frameWidth = 640       # Width of captured video frame
frameHeight = 480      # Height of captured video frame
model = 'efficientdet.tflite'  # Path to the downloaded object detection model

# Visualization parameters for drawing bounding boxes and text
MARGIN = 10             # Margin from bounding box for text display
ROW_SIZE = 30           # Vertical spacing of text
FONT_SIZE = 1           # Font size for drawing labels
FONT_THICKNESS = 1      # Thickness of label text
TEXT_COLOR = (0, 0, 0)  # Black color for label text

#%% Initializing results and result callback
detection_frame = None  # Will store the annotated output frame
detection_result_list = []  # A list to collect results from the callback

# Callback function to save model outputs into detection_result_list
def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    detection_result_list.append(result)

#%% Create an object detection model object

# Set the base model options (using the path to the model file)
base_options = python.BaseOptions(model_asset_path=model)

# Configure the object detection task
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Real-time streaming mode
    max_results=maxResults,                       # Limit number of detected objects
    score_threshold=scoreThreshold,               # Confidence threshold for showing results
    result_callback=save_result                   # Callback to collect results from detection
)

# Create an instance of the object detector using the defined options
detector = vision.ObjectDetector.create_from_options(options)

#%% OpenCV Webcam Setup

# Open the default camera (device index 0)
cap = cv2.VideoCapture(0)

# Set the video frame resolution (as expected by the model)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Verify that the webcam is connected and accessible
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Main loop for continuous video processing
while True:
    try:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirror view (more natural interaction)
        frame = cv2.flip(frame, 1)

        # Convert the image from BGR (OpenCV format) to RGB (TFLite format)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the RGB image in a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Keep a copy of the frame to draw on
        current_frame = frame

        # Run the object detector asynchronously (non-blocking)
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # If there are results from the detector
        if detection_result_list:
            for detection in detection_result_list[0].detections:

                # Extract bounding box coordinates
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                # Draw bounding box in orange (BGR: 0,165,255)
                cv2.rectangle(current_frame, start_point, end_point, (0, 165, 255), 3)

                # --- Draw label and score above the bounding box ---
                category = detection.categories[0]  # Top category result
                category_name = category.category_name  # Object name (e.g., "cat")
                probability = round(category.score, 2)  # Detection confidence
                result_text = category_name + ' (' + str(probability) + ')'  # e.g. "cat (0.93)"

                # Compute label position (slightly above bounding box)
                text_location = (
                    MARGIN + bbox.origin_x,
                    MARGIN + ROW_SIZE + bbox.origin_y
                )

                # Render label text on the frame
                cv2.putText(current_frame, result_text, text_location,
                            cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            # Store this frame to display
            detection_frame = current_frame

            # Clear the result list to prepare for next frame
            detection_result_list.clear()

        # Display the annotated frame (if ready)
        if detection_frame is not None:
            cv2.imshow('object_detection', detection_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow safe exit with Ctrl+C
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
