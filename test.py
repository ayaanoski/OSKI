import pyttsx3
import speech_recognition as sr
import datetime
import os
import subprocess as sp
import openai
import pywhatkit as kit
import time
from datetime import datetime
import threading
from googleapiclient.discovery import build
import webbrowser
import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
from PIL import Image
import psutil
import threading
import requests

API_KEY = "AIzaSyBKU7o_xaRSHmYG7x5oWVz1GtnsOwU1sJ0"
NEWS_API_KEY = "44579792a5254931ad30db4faf675139"
OPENWEATHER_APP_ID = "ff8f8b135e0f8ac86cbdd41570b4838b"
USERNAME = "TEAM OSKEE"
BOTNAME = "OSKEE"
openai.api_key = "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"

listening = True

from testface import face_recognition_with_animation

face_recognition_with_animation("C:\\Users\\asus\\Desktop\\OSKI\\images\\pho.jpg")


def Speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty("voice", id)
    engine.say(text=text)
    engine.runAndWait()


import speech_recognition as sr


def speechrecognition():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")

        # Adjust for ambient noise before capturing audio
        r.adjust_for_ambient_noise(source)

        # Set the pause threshold to 1 second
        r.pause_threshold = 1

        # Capture the audio input
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio)
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""


def get_latest_news():
    news_headlines = []
    res = requests.get(
        f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}&category=general"
    ).json()
    articles = res["articles"]
    for article in articles:
        news_headlines.append(article["title"])
    return news_headlines[:5]


def get_weather_report(city):
    # Your code to fetch weather data from the API
    api_key = "ff8f8b135e0f8ac86cbdd41570b4838b"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if (
            "weather" in data
            and isinstance(data["weather"], list)
            and len(data["weather"]) > 0
        ):
            weather = data["weather"][0]["main"]
        else:
            weather = "Unavailable"

        if "main" in data:
            temperature = data["main"].get("temp")
            feels_like = data["main"].get("feels_like")
        else:
            temperature = "Unavailable"
            feels_like = "Unavailable"

        return weather, temperature, feels_like
    else:
        print("Failed to fetch weather data.")
        return None, None, None


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


# main volume func
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


# main screenshot function
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


current_date_time = get_current_date_time()


def display_image(file_path, main_speech_recognition):
    # Read the image from the file
    image = Image.open(file_path)
    image.show()

    # Check for the "close" command continuously
    while True:
        query = main_speech_recognition().lower()
        if "close" in query:
            process_name = (
                "PhotosApp.exe"  # Adjust the process name based on your system
            )
            for process in psutil.process_iter(["pid", "name"]):
                if process.info["name"] == process_name:
                    pid = process.info["pid"]
                    os.system(f"taskkill /f /pid {pid}")
                    Speak(
                        "Picture taken and saved as yourface.jpg and stored in camera roll. By the way, you look very handsome today"
                    )
                    break


# main camera picture fucntion
def take_picture(main_speech_recognition):
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

    # Display the image with the main speech recognition function
    display_image(save_path, main_speech_recognition)

    Speak(
        "Picture taken and saved as yourface.jpg and stored in camera roll. By the way, you look very handsome today"
    )


def record_screen_with_voice_commands(output_path, fps=60):
    def start_recording():
        nonlocal recording
        recording = True
        recording_thread = threading.Thread(
            target=record_screen_thread, args=(output_path, fps)
        )
        recording_thread.start()

    def stop_recording():
        nonlocal recording
        recording = False

    def record_screen_thread(output_path, fps):
        screen_width, screen_height = pyautogui.size()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (screen_width, screen_height))

        while recording:
            frame = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()

    recording = False

    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Command: {command}")

            if "start recording" in command:
                start_recording()
                Speak("Recording started.")
            elif "stop recording" in command:
                stop_recording()
                Speak("Recording stopped.")

                break

        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            print(f"Speech recognition request failed: {e}")

        time.sleep(1)  # Add a delay to avoid high CPU usage


output_path = "screen_recording.mp4"


def greet_user():
    hour = datetime.now().hour
    if (hour >= 6) and (hour < 12):
        Speak(f"Good Morning {USERNAME}")
    elif (hour >= 12) and (hour < 16):
        Speak(f"Good afternoon {USERNAME}")
    elif (hour >= 16) and (hour < 6):
        Speak(f"Good Evening {USERNAME}")
    else:
        Speak("i hope you guys are enjoying the night")
    Speak(f"I am {BOTNAME}. How may I assist you guys?")


greet_user()


def find_my_ip():
    ip_address = requests.get("https://api64.ipify.org?format=json").json()
    return ip_address["ip"]


def execution(query):
    Query = str(query).lower()
    if "hello" in Query:
        Speak("Hello Ayaan, welcome back")
    elif (
        "what are the things you can do" in Query or "menu" in Query or "help" in Query
    ):
        print("Here is what I can do:")
        print("1. YOUTUBE")
        print("2. GOOGLE SEARCH")
        print("3. MOUSE CONTROL")
        print("4. KEYBOARD CONTROL")
        print("5. TAKE YOUR PICTURE")
        print("6. TAKE A SCREENSHOT:")
        print("7. SCREEN RECORDING:")
        print("8. WEATHER REPORT")
        print("9. LATEST NEWS REPORT")
        print("10. TOP TRENDING MOVIE LIST")
        Speak("Here is what I can do:")
        Speak("1. YOUTUBE")
        Speak("2. GOOGLE SEARCH")
        Speak("3. MOUSE CONTROL")
        Speak("4. KEYBOARD CONTROL")
        Speak("5. TAKE YOUR PICTURE")
        Speak("6. TAKE A SCREENSHOT:")
        Speak("7. SCREEN RECORDING:")
        Speak("9. LATEST NEWS REPORT")
        Speak("10. TOP TRENDING MOVIE LIST")
        Speak("You can ask me anything from these options.")

    elif "screenshot" in Query:
        Speak("sure sir")
        take_screenshot()

    elif "mouse control" in Query:
        Speak("you now have full control over your mouse")
        from mouse import run_gesture_controller

        run_gesture_controller()
    elif "keyboard control" in Query:
        Speak("you now have full control over your keyboard")
        from key import mainkey

        mainkey()
    elif (
        "adjust the volume" in Query
        or "increase the volume" in Query
        or "decrease the volume" in Query
    ):
        adjust_volume_by_voice()
    elif "screen recording" in Query:
        Speak("ok sir , tell me when to start and when to stop.")
        record_screen_with_voice_commands(output_path, fps=60)

    elif "bye" in Query:
        Speak("Goodbye Ayaan, have a nice life")
        exit()
    elif "time" in Query:
        time = datetime.now().strftime("%H:%M")
        Speak(f"It's {time} sir")

    elif "take a picture" in Query:
        Speak("sure sir. Make sure to put on a big smile ")
        take_picture(speechrecognition)

    elif "cmd" in Query:
        Speak("Sure sir, opening command prompt")
        os.system("start cmd")
    elif "news" in query:
        Speak(f"I'm reading out the latest news headlines, sir")
        Speak(get_latest_news())
        Speak("For your convenience, I am printing it on the screen sir.")
        print(*get_latest_news(), sep="\n")

    elif "weather" in query:
        ip_address = find_my_ip()
        city = requests.get(f"https://ipapi.co/{ip_address}/city/").text
        Speak(f"Getting weather report for your city {city}")
        weather, temperature, feels_like = get_weather_report(city)
        if weather is not None and temperature is not None and feels_like is not None:
            Speak(
                f"The current temperature is {temperature}, but it feels like {feels_like}"
            )
            Speak(f"Also, the weather report talks about {weather}")
            Speak("For your convenience, I am printing it on the screen sir.")
            print(
                f"Description: {weather}\nTemperature: {temperature}\nFeels like: {feels_like}"
            )
        else:
            Speak("Sorry, unable to fetch weather data at the moment.")

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
