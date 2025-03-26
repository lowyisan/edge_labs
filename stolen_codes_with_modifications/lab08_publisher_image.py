# Top-Level Explanation:
# This script publishes a pre-captured image as binary data to an MQTT topic.
# It uses the paho-mqtt client to connect to an MQTT broker, reads an image file from disk,
# and publishes its binary content to a specified topic. This script is typically run after a separate 
# process (e.g., a subscriber script) captures an image. The script also includes basic error handling
# to alert the user if the image file does not exist.
#
# Potential Lab Test Q&A:
# Q: What is MQTT and why is it used in this script?
# A: MQTT is a lightweight messaging protocol used for IoT communication. In this script, it is used to 
#    publish image data so that subscribers can receive and process the image.
#
# Q: Why do we read the image in binary mode?
# A: Reading the image in binary mode ('rb') ensures that the file's raw byte data is preserved when sent.
#
# Q: What does the client.disconnect() call do?
# A: It closes the connection to the MQTT broker, which is important for releasing network resources.

# Import required libraries
import paho.mqtt.client as mqtt  # MQTT client library for connecting and publishing messages
import time                      # Time library for delays and logging

# MQTT Configuration
BROKER_ADDRESS = "localhost"       # MQTT broker address (change if using a remote broker)
TOPIC = "camera/image"             # MQTT topic where the image data will be published
IMAGE_PATH = "captured_image.jpg"  # Path to the image file to be published

# Initialize the MQTT client and connect to the broker
client = mqtt.Client()             # Create a new MQTT client instance
client.connect(BROKER_ADDRESS, 1883, 60)  # Connect to the broker at port 1883 with a keepalive of 60 seconds

# Read the image file in binary mode and publish it to the MQTT topic
try:
    with open(IMAGE_PATH, "rb") as f:   # Open the image file in read-binary mode
        image_data = f.read()           # Read the entire file as binary data
        client.publish(TOPIC, image_data)  # Publish the binary image data to the specified topic
        print(f"📤 Published image '{IMAGE_PATH}' to topic '{TOPIC}'")
except FileNotFoundError:
    # If the file is not found, print an error message indicating that the image must be captured first.
    print(f"❌ Image '{IMAGE_PATH}' not found. Please run the subscriber script to capture an image first.")

# Disconnect the MQTT client from the broker after publishing the message
client.disconnect()

'''
Usage Instructions:
1. Start an MQTT broker (e.g., Mosquitto).
2. Run subscriber_capture.py to capture an image.
3. Publish the message "capture" to the topic "camera/trigger" using any MQTT client:
    mosquitto_pub -t camera/trigger -m "capture"
4. Once the image is saved, run publisher_image.py to publish the image to the topic "camera/image".
'''
