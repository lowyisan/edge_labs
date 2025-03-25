# IoT Communications: MQTT

## **Objective**
Learn how to install and configure an MQTT broker, create MQTT publisher and subscriber clients, and test communication between them on a Raspberry Pi 400.

*Potential Q&A:*

- **Q:** What is the purpose of using MQTT in IoT?  
  **A:** MQTT is a lightweight publish/subscribe protocol designed for resource-constrained devices and low-bandwidth networks, making it ideal for IoT communications.

---

## **Prerequisites**

1. Raspberry Pi 400 with an operating system (e.g., Raspbian) already set up.
2. Internet connection for the Raspberry Pi.

---

## **Materials**

- Raspberry Pi 400
- Power supply for Raspberry Pi
- Keyboard, mouse, and monitor for Raspberry Pi (or access via SSH)
- MQTT broker (e.g., Mosquitto)
- MQTT publisher and subscriber clients (e.g., using the Python Paho MQTT library)

---

## **Introduction**

MQTT stands for **Message Queue Telemetry Transport**. It is a lightweight, open protocol that uses a publish/subscribe messaging model to facilitate communication among devices. Here are its main components:

- **MQTT Broker:**  
  Acts as an intermediary that receives messages from publisher clients and then forwards them to subscriber clients based on topics.  
  *Potential Q&A:*  
  - **Q:** What role does the MQTT broker play?  
    **A:** It manages the distribution of messages between publishers and subscribers, ensuring messages are delivered to clients interested in a specific topic.

- **Topic:**  
  A namespace or channel for messages. Publishers send messages to a topic, and subscribers receive messages from topics they are interested in.

- **MQTT Client:**  
  A device (or application) that can publish messages, subscribe to topics, unsubscribe, and disconnect.  
  *Potential Q&A:*  
  - **Q:** How does an MQTT client communicate with the broker?  
    **A:** It first establishes a connection with the broker, then can publish to or subscribe from specific topics.

Below is an example figure that illustrates a typical MQTT implementation:

![MQTT Architecture](https://github.com/drfuzzi/INF2009_MQTT/assets/108112390/26517ab1-d700-48cd-bfbd-4d7511ecfc9a)

*Figure 1: An example of MQTT implementation*

---

## **Lab Exercise**

### **1. Install and Configure the MQTT Broker on a Raspberry Pi 400**

#### a. Update the Package List (Optional)
```bash
sudo apt update  # Update the package list to ensure you have the latest repository information.

Potential Q&A:

Q: Why run sudo apt update?
A: It refreshes the package lists from the repositories to make sure you install the latest versions.
```

b. Install the Mosquitto MQTT Broker
```bash
sudo apt install mosquitto  # Install Mosquitto, a popular open-source MQTT broker.
```
Potential Q&A:

Q: What is Mosquitto?
A: Mosquitto is an open-source MQTT broker that facilitates MQTT messaging for IoT devices.

c. Locate the Mosquitto Configuration File
Open the configuration file using a text editor:
```bash
sudo nano /etc/mosquitto/mosquitto.conf
```
Potential Q&A:

Q: Why edit mosquitto.conf?
A: To customize the broker’s behavior such as port settings and access permissions.

d. Edit the Mosquitto Configuration File
Within the file, add or modify the following lines to enable connections on port 1883 and allow anonymous access:
```bash
listener 1883          # The broker will listen on port 1883 for incoming connections.
allow_anonymous true   # Allow clients to connect without requiring a username and password.
```
Potential Q&A:

Q: Is it safe to allow anonymous access?
A: For testing or controlled environments it is acceptable; however, for production systems, you should secure your broker with authentication.

e. Start the Mosquitto Broker Manually
```bash
sudo mosquitto -c /etc/mosquitto/mosquitto.conf  # Start the broker with the specified configuration file.
```
2. Enable Mosquitto Broker to Run on Boot (Optional)
a. Start and Enable the Broker on Boot
```bash
sudo systemctl start mosquitto   # Start the Mosquitto service.
sudo systemctl enable mosquitto  # Enable Mosquitto to run on boot.
```
Potential Q&A:

Q: What does systemctl enable do?
A: It configures the service to start automatically during system boot.

b. Restart the Mosquitto Broker to Apply Configuration Changes
```bash
sudo systemctl restart mosquitto  # Restart the service to load new configuration settings.
```
c. Verify the Mosquitto Service Status
```bash
systemctl status mosquitto  # Check if Mosquitto is running and view its status.
```
d. Disable and Stop the Broker (if needed)
```bash
sudo systemctl disable mosquitto  # Prevent Mosquitto from starting on boot.
sudo systemctl stop mosquitto     # Stop the running Mosquitto service.
```
3. Install and Configure the MQTT Client (Publisher and/or Subscriber)
a. Activate Your Python Virtual Environment
```bash
source myenv/bin/activate  # Activate the virtual environment where your Python packages are installed.
```
Potential Q&A:

Q: Why use a virtual environment?
A: It isolates dependencies and package versions for your project from the system-wide Python installation.

b. Install the Python Paho MQTT Library
```bash
pip install paho-mqtt  # Install the Paho MQTT library to enable MQTT functionalities in Python.
```
c. Create a Python Script for the MQTT Publisher (mqtt_publisher.py)
Below is an example script with inline comments:
```bash
import paho.mqtt.client as mqtt  # Import the MQTT client library.
import time  # Import time module to add delays between publishes.

# Create an MQTT client instance with a unique client ID.
client = mqtt.Client("Publisher")

# Connect to the MQTT broker. Replace "localhost" with the broker's IP (e.g., "192.168.50.115").
client.connect("localhost", 1883)

while True:
    # Publish a message "Hello, MQTT!" to the topic "test/topic".
    client.publish("test/topic", "Hello, MQTT!")
    # Pause for 5 seconds before sending the next message.
    time.sleep(5)
```
Potential Q&A:

Q: What is the purpose of client.publish()?
A: It sends a message to the specified topic so that any subscribers to that topic can receive it.

d. Create a Python Script for the MQTT Subscriber (mqtt_subscriber.py)
Below is an example script with inline comments:
```bash
import paho.mqtt.client as mqtt  # Import the MQTT client library.

# Callback function that is called when a message is received.
def on_message(client, userdata, message):
    # Print the received message along with its topic.
    print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

# Create an MQTT client instance with a unique client ID.
client = mqtt.Client("Subscriber")

# Assign the on_message callback function to handle incoming messages.
client.on_message = on_message

# Connect to the MQTT broker (replace "localhost" with the broker's IP address if needed).
client.connect("localhost", 1883)

# Subscribe to the topic "test/topic" to receive messages.
client.subscribe("test/topic")

# Enter a loop that waits for messages and processes them as they arrive.
client.loop_forever()
```
Potential Q&A:

Q: How does the subscriber receive messages?
A: It connects to the broker, subscribes to a topic, and continuously listens for incoming messages using client.loop_forever().

4. Testing Your MQTT Communication
Open Two Terminal Windows on the Raspberry Pi.

In the First Terminal, Run the Subscriber Script:
```bash
python3 mqtt_subscriber.py
```
Potential Q&A:

Q: What should you expect in the subscriber terminal?
A: It should display incoming messages published to the topic "test/topic".

In the Second Terminal, Run the Publisher Script:
```bash
python3 mqtt_publisher.py
```
Potential Q&A:

Q: What will the publisher do?
A: It will continuously publish the message "Hello, MQTT!" every 5 seconds.

Remember to Activate the Virtual Environment Before Running the Scripts:
```bash
source myenv/bin/activate
```
Observe the Communication:
The subscriber terminal should display the messages sent by the publisher.

Lab Assignment
Task: Create two Python scripts to extend this lab session by integrating image capture with MQTT communication.

Subscriber Script:

Listen on an MQTT topic for a message indicating that an image capture is requested.

Once the message is received, capture an image from a connected webcam.

Publisher Script:

After capturing the image, transmit the image over MQTT to a designated topic.

Potential Q&A:

Q: How can you capture an image from a webcam in Python?
A: You can use libraries such as OpenCV (cv2) to access the webcam, capture frames, and save or process the images.

Q: What is the benefit of integrating MQTT with image capture?
A: It allows for remote triggering of image capture and can be used in applications such as security systems, remote monitoring, or interactive IoT devices.