Subscriber
import paho.mqtt.client as mqtt
import os
import time

BROKER = "192.168.248.148"  # Replace with your broker IP
SUBSCRIBE_TOPIC = "capture/image"
PUBLISH_TOPIC = "image/data"

def on_message(client, userdata, message):
    print("Message received! Capturing image...")

    # Capture image using fswebcam
    os.system("fswebcam -r 1280x720 --no-banner image.jpg")
    time.sleep(2)  # Ensure the file is written

    # Read the image file as binary
    with open("image.jpg", "rb") as img_file:
        img_data = img_file.read()

    # Publish the image as bytes
    print("Publishing image...")
    client.publish(PUBLISH_TOPIC, img_data)

client = mqtt.Client("Subscriber", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect(BROKER, 1883)
client.subscribe(SUBSCRIBE_TOPIC)

print("Waiting for messages to capture image...")
client.loop_forever()



=============================================================================================================


Publisher
import paho.mqtt.client as mqtt
import time

BROKER = "192.168.248.148"  # Replace with your broker IP
REQUEST_TOPIC = "capture/image"
IMAGE_TOPIC = "image/data"

def on_message(client, userdata, message):
    print("Image received! Saving as received_image.jpg")

    # Save the received binary data as an image file
    with open("received_image.jpg", "wb") as img_file:
        img_file.write(message.payload)

    print("Image saved successfully!")

client = mqtt.Client("Publisher", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message
client.connect(BROKER, 1883)

# Subscribe to receive images
client.subscribe(IMAGE_TOPIC)
client.loop_start()  # Start listening in a separate thread

while True:
    print("Requesting image capture...")
    client.publish(REQUEST_TOPIC, "Capture Image")
    time.sleep(10)  # Adjust timing as needed
