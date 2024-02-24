import pyttsx3
import speech_recognition as sr
import datetime
import os
import subprocess as sp
import openai
import pywhatkit as kit
import requests
import random
import time
from datetime import datetime
import threading
from googleapiclient.discovery import build
import webbrowser
import cv2
import face_recognition
import sys
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
from PIL import Image
import psutil

API_KEY = "AIzaSyBKU7o_xaRSHmYG7x5oWVz1GtnsOwU1sJ0"
# mainclapexe()

USERNAME = "AYAAN"
BOTNAME = "OSKEE"
openai.api_key = "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"


def Speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
    engine.setProperty("voice", id)
    engine.say(text=text)
    engine.runAndWait()


def speechrecognition():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except:
        return ""


def object_detection():
    # Load the YOLO object detection model
    net = cv2.dnn.readNet(
        "C:\\Users\\asus\\Desktop\\IP\\yolov3.weights",
        "C:\\Users\\asus\\Desktop\\IP\\yolov3.cfg",
    )

    # Load the classes file
    with open("C:\\Users\\asus\\Desktop\\OSKI\\od\\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Define the layer names for the output layers of the YOLO model
    layer_names = ["yolo_82", "yolo_94", "yolo_106"]

    # Specify the minimum confidence required to filter out weak predictions
    min_confidence = 0.5

    # Load the video capture object
    video_capture = cv2.VideoCapture(0)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Initialize the speech recognition engine
    recognizer = sr.Recognizer()

    exit_requested = False

    while not exit_requested:
        # Grab a frame from the video capture object
        ret, frame = video_capture.read()

        # Only proceed if the frame was successfully grabbed
        if not ret:
            break

        # Get the height and width of the frame
        height, width = frame.shape[:2]

        # Construct a blob from the frame and perform a forward pass with the YOLO model
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        layer_outputs = net.forward(layer_names)

        # Initialize lists to store the bounding box coordinates, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            # Loop over each of the detections in the layer output
            for detection in output:
                # Extract the class ID and confidence from the detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only consider detections with a confidence greater than the minimum confidence
                if confidence > min_confidence:
                    # Scale the bounding box coordinates back relative to the size of the frame
                    box = detection[0:4] * np.array([width, height, width, height])
                    center_x, center_y, w, h = box.astype("int")

                    # Calculate the top-left corner of the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Append the bounding box, confidence, and class ID to their respective lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Convert the tuple to a list
        indices = list(indices)

        # Loop over the remaining indices after non-maximum suppression
        for i in indices:
            box = boxes[i]
            x, y, w, h = box

            # Draw the bounding box and label on the frame
            color = (0, 255, 0)  # BGR color format
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Speak the name of the detected object
            engine.say(f"it is a {classes[class_ids[i]]}")
            engine.runAndWait()

        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            try:
                command = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                command = None
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
                command = "..."

        print(f"Command: {command}")

        if command and "exit" in command:
            Speak("exiting object detection")
            exit_requested = True

    # Release the video capture object and close the window

    video_capture.release()
    cv2.destroyAllWindows()


def get_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def adjust_volume_by_amount(amount):
    volume = get_volume_control()
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(0.0, min(1.0, current_volume + amount / 100.0))
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    Speak(f"Volume adjusted by {amount} levels. New volume: {new_volume * 100:.2f}%")


def parse_volume_command(command):
    try:
        if "increase the volume by" in command:
            amount = int(command.split("increase the volume by")[-1])
            adjust_volume_by_amount(amount)
        elif "decrease the volume by" in command:
            amount = int(command.split("decrease the volume by")[-1])
            adjust_volume_by_amount(-amount)
        else:
            print(
                "Invalid command. Please specify 'increase volume by' or 'decrease volume by' with a numeric value."
            )
    except ValueError:
        print("Invalid numeric value in the command.")


def adjust_volume_by_voice():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        Speak("adjusting volume")

        while True:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()

                if "exit" in command or "stop" in command:
                    Speak("Exiting volume adjustment.")
                    break

                parse_volume_command(command)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )
                break


def show_image(file_path):
    img = Image.open(file_path)
    img.show()
    time.sleep(1)  # Display the image for 10 seconds
    img.close()


def close_image_viewer(file_path):
    try:
        process_name = "PhotosApp.exe"  # Adjust the process name based on your system
        for process in psutil.process_iter(["pid", "name"]):
            if process.info["name"] == process_name:
                pid = process.info["pid"]
                os.system(f"taskkill /f /pid {pid}")
                break
    except Exception as e:
        print(f"Error closing image viewer: {e}")


def take_screenshot(file_path="C:\\Users\\asus\\Pictures\\Screenshots\\screenshot.jpg"):
    # Capture screenshot
    screenshot = pyautogui.screenshot()

    # Save the screenshot
    screenshot.save(file_path)
    print(f"Screenshot saved at {file_path}")

    # Show the image using Pillow
    show_image(file_path)

    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                print("say...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                user_input = recognizer.recognize_google(audio).lower()

                if user_input == "close":
                    Speak("closing it sir")
                    close_image_viewer(file_path)
                    break

        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            print(f"Speech recognition request failed: {e}")


def search_youtube(query):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    search_response = (
        youtube.search()
        .list(q=query, type="video", part="id,snippet", maxResults=1)
        .execute()
    )

    video_id = search_response["items"][0]["id"]["videoId"]
    return f"https://www.youtube.com/watch?v={video_id}"


def search_on_google(query):
    kit.search(query)


def generate_chat_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return response["choices"][0]["message"]["content"]


def get_current_date_time():
    now = datetime.now()
    current_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_date_time


def write_to_notepad(query):
    notepad_path = r"C:\Windows\System32\notepad.exe"
    sp.Popen([notepad_path, "ass.txt"])

    with open("text.txt", "w") as file:
        file.write(query)


current_date_time = get_current_date_time()


def display_image(file_path):
    # Read the image from the file
    image = cv2.imread(file_path)

    if image is None:
        print(f"Error: Could not read the image from {file_path}")
        return

    # Display the image
    cv2.imshow("PICTURE", image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    close_window = False
    while not close_window:
        if cv2.getWindowProperty("PICTURE", cv2.WND_PROP_VISIBLE) < 1:
            break
        query = speechrecognition().lower()
        if "close" in query:
            close_window = True
    cv2.destroyAllWindows()


def take_picture():
    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture frame.")
        cap.release()
        return

    # Save the frame as an image
    save_directory = "C:\\Users\\asus\\Pictures\\Camera Roll"
    file_name = "yourface.jpg"
    save_path = os.path.join(save_directory, file_name)

    # Save the frame as an image
    cv2.imwrite(save_path, frame)

    # Release the camera
    cap.release()
    display_image(save_path)

    Speak(
        "Picture taken and saved as yourface.jpg and stored in camera roll. By the way , you look very  handsome today"
    )


def recognize_faces():
    Speak("please be in front of the camera sir ")
    known_people = {
        "Ayaan": face_recognition.face_encodings(
            face_recognition.load_image_file(
                "C:\\Users\\asus\\Desktop\\OSKI\\images\\pho.jpg"
            )
        )[0],
        "nikhil": face_recognition.face_encodings(
            face_recognition.load_image_file(
                "C:\\Users\\asus\\Desktop\\OSKI\\images\\pho2.jpg"
            )
        )[0],
        # Add more people as needed
    }

    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 800, 600)
    speech_engine = pyttsx3.init()

    while True:
        # Capture each frame
        _, frame = video_capture.read()

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(
                list(known_people.values()), face_encoding
            )
            name = "Stranger"

            for idx, match in enumerate(matches):
                if match:
                    name = list(known_people.keys())[idx]
                    # Speak "face recognized" when a known face is recognized
                    speech_engine.say("Face recognized")
                    speech_engine.runAndWait()
                    # Turn the box green
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "Approved",
                        (left, bottom + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    cv2.waitKey(3000)
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return

            # Draw a rectangle around the face in red
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                name,
                (left, bottom + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# recognize_faces()


def greet_user():
    """Greets the user according to the time"""

    hour = datetime.now().hour
    if (hour >= 6) and (hour < 12):
        Speak(f"Good Morning {USERNAME}")
    elif (hour >= 12) and (hour < 16):
        Speak(f"Good afternoon {USERNAME}")
    elif (hour >= 16) and (hour < 6):
        Speak(f"Good Evening {USERNAME}")
    else:
        Speak("i hope you're enjoying the night ")
    Speak(f"I am {BOTNAME}. How may I assist you?")


Speak("hello sir , welcome back to your room")
# greet_user()


def execution(query):
    Query = str(query).lower()
    if "hello" in Query:
        Speak("Hello Ayaan, welcome back")
    elif "screenshot" in Query:
        Speak("sure sir")
        take_screenshot()
    elif "VLC" in Query:
        os.startfile(
            "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\VideoLAN\VLC media player.lnk"
        )  # Adjust this path accordingly

        Speak("opening VLC")
    elif (
        "adjust the volume" in Query
        or "increase the volume" in Query
        or "decrease the volume" in Query
    ):
        adjust_volume_by_voice()
    elif "object detection" in Query:
        Speak("sure sir , opening object detection, i might me inaccurate sir.")
        object_detection()
    elif "bye" in Query:
        Speak("Goodbye Ayaan, have a nice life")
        exit()
    elif "time" in Query:
        time = datetime.now().strftime("%H:%M")
        Speak(f"It's {time} sir")
    elif "start writing" in Query:
        print("What would you like to write?")
        content_to_write = speechrecognition()
        if content_to_write:
            write_to_notepad(content_to_write)
            print("done")
            exit()
    elif "take a picture" in Query:
        Speak("sure sir. Make sure to put on a big smile ")
        t = threading.Thread(target=take_picture)
        t.start()

    elif "cmd" in Query:
        Speak("Sure sir, opening command prompt")
        os.system("start cmd")

    elif "what is your name" in Query or "tumhara naam kya hai" in Query:
        Speak("my name is oskee , and i am created by ayaan")
    elif "google" in Query:
        Speak("what do you want to search on google, sir?")
        query = speechrecognition().lower()
        search_on_google(query)

    elif "youtube" in Query:
        Speak("sure sir , opening youtube")
        search_youtube(query)
        webbrowser.open(search_youtube(query))

    elif "after effects" in Query:
        Speak("ok sir, i will hold on")
        os.startfile(
            "C:\\Program Files\\Adobe\\Adobe After Effects 2022\\Support Files\\AfterFX.exe"
        )
    else:
        try:
            assistant_response = generate_chat_response(Query)
            print("OSKI:", assistant_response)
            Speak(assistant_response)
        except:
            return "sorry you've exceeded your limit. Please try again after some time."


while True:
    print("")
    Query = speechrecognition()
    execution(Query)
