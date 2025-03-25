import cv2
import mediapipe as mp
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === CONFIGURATION ===
model_path = 'efficientdet.tflite'           # Path to the downloaded TFLite object detection model
target_object = 'cell phone'                 # Object name to filter for (case-insensitive)
input_video_path = 'input_video.mp4'         # Path to the input video to summarize
output_video_path = 'summary_output.mp4'     # Path to the summarized output video
save_frames_dir = 'filtered_frames'          # Optional folder to store individual matching frames as images

# === OBJECT DETECTION PARAMETERS ===
score_threshold = 0.3     # Minimum confidence score to count a detection as valid
max_results = 5           # Maximum number of objects to detect per frame

# === PREPARE OUTPUT FOLDER ===
os.makedirs(save_frames_dir, exist_ok=True)  # Create folder to save matched frames (if it doesn't exist)

# === SET UP DETECTOR ===
# Define model and detection options for MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,       # Non-live mode (frame-by-frame from video)
    max_results=max_results,
    score_threshold=score_threshold
)

# Create the object detector from the configured options
detector = vision.ObjectDetector.create_from_options(options)

# === OPEN INPUT VIDEO ===
cap = cv2.VideoCapture(input_video_path)         # Open video file for reading
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

# Retrieve video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))             # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width of video frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of video frames

# === OUTPUT VIDEO WRITER ===
# Create a video writer to store the summary (only matched frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === FRAME LOOP ===
frame_idx = 0       # Counter to track current frame index
matched_count = 0   # Counter to track how many frames matched the target object

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Stop if the video ends or there's an error

    # Convert frame from BGR (OpenCV) to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the RGB image into MediaPipe's Image class
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run the object detection on the current frame
    detection_result = detector.detect(mp_image)

    # Check if any detection matches the target object
    match_found = False
    for detection in detection_result.detections:
        category = detection.categories[0]  # Take the top prediction for each detection
        if category.category_name.lower() == target_object.lower():
            match_found = True
            break

    if match_found:
        matched_count += 1

        # Write this frame to the summary video output
        out.write(frame)

        # Save the frame as a .jpg image (optional)
        filename = os.path.join(save_frames_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(filename, frame)

    frame_idx += 1  # Move to the next frame

# === CLEANUP ===
cap.release()   # Release the video input stream
out.release()   # Finalize the output video file

# Final report
print(f"✅ Summarization complete. {matched_count} matching frames saved.")
print(f"🎞️ Output video saved to: {output_video_path}")
