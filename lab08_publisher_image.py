# This script publishes a previously captured image (from Script 1) as binary data to an MQTT topic.

import paho.mqtt.client as mqtt
import time

# MQTT Configuration
BROKER_ADDRESS = "localhost"       # Replace with your MQTT broker address
TOPIC = "camera/image"             # Topic to publish image data
IMAGE_PATH = "captured_image.jpg"  # Path to the image to be sent

# Initialize MQTT client and connect
client = mqtt.Client()
client.connect(BROKER_ADDRESS, 1883, 60)

# Read the image file in binary mode
try:
    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()
        # Publish the image data as binary payload
        client.publish(TOPIC, image_data)
        print(f"📤 Published image '{IMAGE_PATH}' to topic '{TOPIC}'")
except FileNotFoundError:
    print(f"❌ Image '{IMAGE_PATH}' not found. Please run the subscriber script to capture an image first.")

client.disconnect()

'''
1. Start an MQTT broker

2. Run subscriber_capture.py 

3. Publish "capture" to topic camera/trigger using any MQTT client (e.g. mosquitto_pub)
    mosquitto_pub -t camera/trigger -m "capture"

4. After image is saved, run publisher_image.py to publish it to camera/image
'''