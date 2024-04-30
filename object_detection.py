import torch
import cv2
import pyttsx3
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5:v6.0", "yolov5s")
model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

# Initialize variables for motion detection
previous_frame = None
motion_detected = False

# Timer for the first 5 seconds
start_time = cv2.getTickCount()


# Function to convert list to a string with commas
def list_to_string(lst):
    return ", ".join(lst)


# Capture video
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    quit()

# Get the frame rate from the camera
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if the frame rate is zero
if fps == 0:
    print("Error: Unable to retrieve frame rate from the camera.")
    quit()

# Set target FPS
target_fps = 30
frame_skip = int(fps / target_fps)
frame_count = 0

# Variable to store the last frame with detected objects
last_frame_with_objects = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Capture frames for the first 5 seconds
    current_time = cv2.getTickCount()
    elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
    if elapsed_time >= 5:
        break

    # Skip frames to achieve target FPS
    if frame_count % frame_skip == 0:
        # Mirror the frame
        mirrored_frame = cv2.flip(frame, 1)  # 1 for horizontal flip

        # Inference
        results = model(mirrored_frame)

        # Extract detected object names
        detected_object_names_frame = results.names
        detected_object_names_frame = [
            name
            for name in detected_object_names_frame
            if name in results.pandas().xyxy[0].name.tolist()
        ]

        # Perform motion detection
        gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if previous_frame is not None:
            frame_diff = cv2.absdiff(previous_frame, gray_frame)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    motion_detected = True
                    break

        previous_frame = gray_frame

        # Convert detected object names to a string
        detected_objects_str = list_to_string(set(detected_object_names_frame))

        # Speak detected objects and motion detection result
        if motion_detected:
            engine.say("Motion detected.")
            motion_detected = False  # Reset motion detection flag after speaking
        if detected_objects_str:
            engine.say(f"I can see a {detected_objects_str}.")

        engine.runAndWait()

        # Display the frame with detected objects
        cv2.imshow("YOLOv5 Object Detection", results.render()[0])

        # Store the frame if objects were detected
        if detected_objects_str:
            last_frame_with_objects = mirrored_frame

    frame_count += 1

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object
cap.release()

# Display the last frame with detected objects (if any)
if last_frame_with_objects is not None:
    # cv2.imshow("Last Frame with Detected Objects", last_frame_with_objects)
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
