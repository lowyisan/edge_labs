# Top-Level Explanation:
# This script subscribes to an MQTT topic ("camera/trigger") and listens for messages.
# When a message with the command "capture" is received, it activates the webcam to capture an image
# and saves the image as "captured_image.jpg". This allows for remote triggering of image capture via MQTT.
#
# Potential Lab Test Q&A:
# Q: What is MQTT and how is it used in this script?
# A: MQTT is a lightweight messaging protocol commonly used for IoT applications. In this script,
#    it is used to listen for commands on a specific topic and trigger actions (image capture) based on those commands.
#
# Q: How does the script determine when to capture an image?
# A: The script defines an MQTT callback function that checks if the received command (after decoding)
#    is "capture" (case-insensitive). If it matches, the capture_image() function is called.
#
# Q: Why is it important to release the webcam after capturing the image?
# A: Releasing the webcam (via cap.release()) frees the hardware resource for other processes or future captures,
#    preventing resource conflicts or lock-ups.

# Import OpenCV for image capture and processing
import cv2

# Import the paho-mqtt client library for MQTT communications
import paho.mqtt.client as mqtt

# MQTT Configuration
BROKER_ADDRESS = "localhost"  # MQTT broker address (change if your broker is remote)
TOPIC = "camera/trigger"      # MQTT topic to subscribe to for capture commands

# MQTT callback function when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(TOPIC)  # Subscribe to the topic to receive capture commands

# MQTT callback function when a message is received on a subscribed topic
def on_message(client, userdata, msg):
    command = msg.payload.decode()  # Decode the binary payload to string
    print(f"Received command: {command}")
    # If the command (case-insensitive) is "capture", call the capture_image function
    if command.lower() == "capture":
        capture_image()

# Function to capture an image from the webcam
def capture_image():
    print("Capturing image from webcam...")
    # Open the default webcam (device index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access the webcam.")
        return
    # Read a single frame from the webcam
    ret, frame = cap.read()
    if ret:
        filename = "captured_image.jpg"
        # Save the captured frame as a JPEG image
        cv2.imwrite(filename, frame)
        print(f"✅ Image captured and saved as '{filename}'")
    else:
        print("❌ Failed to capture image.")
    cap.release()  # Release the webcam resource

# Initialize the MQTT client
client = mqtt.Client()
client.on_connect = on_connect  # Assign the on_connect callback function
client.on_message = on_message  # Assign the on_message callback function

# Connect to the MQTT broker on the specified address and port (1883 is the default)
client.connect(BROKER_ADDRESS, 1883, 60)

print("📡 Waiting for capture command...")
# Start the MQTT client loop, which processes network traffic, dispatches callbacks, and reconnects if necessary
client.loop_forever()
