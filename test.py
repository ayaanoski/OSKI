import pyttsx3
import speech_recognition as sr
import datetime
import os
import subprocess as sp
import openai
import pywhatkit as kit
import requests
import random
from clap import mainclapexe
import time
from datetime import datetime
import threading
from googleapiclient.discovery import build
import webbrowser
import cv2


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
greet_user()


def execution(query):
    Query = str(query).lower()
    if "hello" in Query:
        Speak("Hello Ayaan, welcome back")

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
