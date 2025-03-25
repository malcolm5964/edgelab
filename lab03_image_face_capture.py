# Import required libraries
import cv2                      # OpenCV for video capture and image processing
import mediapipe as mp          # MediaPipe for face mesh and drawing utilities

#%% Initialize MediaPipe Face Mesh

# Load the FaceMesh module from MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Create a FaceMesh object with settings:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,           # False = continuous video stream (not static images)
    max_num_faces=1,                   # Detect only one face (change to >1 to track multiple)
    min_detection_confidence=0.5,      # Minimum confidence for face detection
    min_tracking_confidence=0.5        # Minimum confidence for tracking facial landmarks
)

# Drawing utilities from MediaPipe to render mesh and contours
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Initialize the webcam

cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0)

# Check if webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

#%% Main processing loop

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the BGR image (OpenCV default) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(rgb_frame)

    # If facial landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh tessellation (triangular structure across the face)
            mp_drawing.draw_landmarks(
                image=frame,  # Original frame (BGR)
                landmark_list=face_landmarks,  # Detected landmark coordinates
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Mesh triangles
                landmark_drawing_spec=None,  # Don't draw individual dots
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw contours (jawline, lips, eyebrows, etc.)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # Show the annotated frame in a window
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

#%% Clean up

cap.release()              # Release the webcam
cv2.destroyAllWindows()    # Close all OpenCV windows
