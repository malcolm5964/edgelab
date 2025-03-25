# This script subscribes to a topic and captures an image from the webcam when it receives a "capture" command.

import cv2
import paho.mqtt.client as mqtt

# MQTT Configuration
BROKER_ADDRESS = "localhost"  # Replace with your MQTT broker address
TOPIC = "camera/trigger"      # Topic to listen for capture commands

# MQTT callback when a connection is established
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(TOPIC)

# MQTT callback when a message is received on a subscribed topic
def on_message(client, userdata, msg):
    command = msg.payload.decode()
    print(f"Received command: {command}")
    if command.lower() == "capture":
        capture_image()

# Function to capture image from webcam
def capture_image():
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access the webcam.")
        return
    ret, frame = cap.read()
    if ret:
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Image captured and saved as '{filename}'")
    else:
        print("❌ Failed to capture image.")
    cap.release()

# Initialize and run MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_ADDRESS, 1883, 60)

print("📡 Waiting for capture command...")
client.loop_forever()
