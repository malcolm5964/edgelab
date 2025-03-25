#%% Reference: https://github.com/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/raspberry_pi/
# Download hand gesture detector model
# wget -O gesture_recognizer.task -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

import cv2
import mediapipe as mp
import time

# Import the MediaPipe Python wrapper and vision-specific APIs
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
from mediapipe.framework.formats import landmark_pb2 # For drawing landmark data on the screen

# Predefined modules for drawing hand connections and landmarks
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Parameters

numHands = 2  
# Maximum number of hands to detect at the same time.
# ↑ = more processing, but supports multi-hand recognition.

model = 'gesture_recognizer.task'  
# Path to the downloaded TFLite gesture recognition model.

minHandDetectionConfidence = 0.5  
# Minimum confidence required to detect a hand.
# ↑ = more strict, ↓ = allows detection in lower-confidence scenarios.

minHandPresenceConfidence = 0.5  
# Ensures hand is still present once detected; prevents flickering.

minTrackingConfidence = 0.5  
# Confidence needed to continue tracking a detected hand.

frameWidth = 640
frameHeight = 480
# Frame resolution for webcam capture.

# Visualization parameters
row_size = 50           # Vertical spacing between lines of text
left_margin = 24        # Left margin for text display
text_color = (0, 0, 0)  # Text color: black
font_size = 1
font_thickness = 1

# Label box parameters
label_text_color = (255, 255, 255)  # Text color for labels: white
label_font_size = 1
label_thickness = 2

#%% Initializing results and save result call back for appending results.
recognition_frame = None
recognition_result_list = []

# Callback function that will be called whenever the model produces a result
def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
    recognition_result_list.append(result)

#%% Create an Hand Gesture Control object.

# Initialize base model options with path to gesture model
base_options = python.BaseOptions(model_asset_path=model)

# Set options for gesture recognizer
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,  # Real-time inference
    num_hands=numHands,
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence,
    result_callback=save_result  # Use callback for streaming recognition
)

# Create the recognizer instance from options
recognizer = vision.GestureRecognizer.create_from_options(options)

#%% Open CV Video Capture and frame analysis
cap = cv2.VideoCapture(0)  # Open default camera (webcam)

# Set resolution as defined above
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# The loop will break on pressing the 'q' key
while True:
    try:
        # Capture one frame from webcam
        ret, frame = cap.read() 
        
        # Flip the frame to match natural hand orientation
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR (OpenCV format) to RGB (TFLite model expects RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Wrap the RGB frame into MediaPipe's image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Store current frame for annotation
        current_frame = frame

        # Run the recognizer asynchronously, passing in the timestamp in milliseconds
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        # If results were detected in the callback
        if recognition_result_list:
            # Loop through each detected hand and its landmarks
            for hand_index, hand_landmarks in enumerate(recognition_result_list[0].hand_landmarks):

                # --- Bounding Box Computation ---
                # Used to place the label text near the detected hand
                x_min = min([landmark.x for landmark in hand_landmarks])
                y_min = min([landmark.y for landmark in hand_landmarks])
                y_max = max([landmark.y for landmark in hand_landmarks])

                # Convert normalized coords to pixel coords
                frame_height, frame_width = current_frame.shape[:2]
                x_min_px = int(x_min * frame_width)
                y_min_px = int(y_min * frame_height)
                y_max_px = int(y_max * frame_height)

                # --- Display Gesture Label ---
                # Get gesture category name and confidence score (top result)
                if recognition_result_list[0].gestures:
                    gesture = recognition_result_list[0].gestures[hand_index]
                    category_name = gesture[0].category_name
                    score = round(gesture[0].score, 2)
                    result_text = f'{category_name} ({score})'

                    # Get dimensions of text for placement
                    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size, label_thickness)[0]
                    text_width, text_height = text_size

                    # Position label above hand
                    text_x = x_min_px
                    text_y = y_min_px - 10  # Slightly above the hand

                    # Ensure the label is not drawn off-screen
                    if text_y < 0:
                        text_y = y_max_px + text_height

                    # Draw the gesture text on the screen
                    cv2.putText(current_frame, result_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_DUPLEX,
                                label_font_size,
                                label_text_color,
                                label_thickness,
                                cv2.LINE_AA)

                # --- Draw Landmarks & Connections on the Frame ---
                # Convert to landmark protobuf format for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # Use MediaPipe's utility to draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    current_frame,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Store the processed frame for display
            recognition_frame = current_frame

            # Clear the result list for the next frame
            recognition_result_list.clear()

        # If a processed frame is ready, display it
        if recognition_frame is not None:
            cv2.imshow('gesture_recognition', recognition_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Graceful exit with Ctrl+C
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
