import speech_recognition as sr
import pyautogui
import pyperclip
import os
import time
import subprocess


def type_text(text):
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")


def start_writing():
    # Open Notepad
    os.system("start notepad")
    pyautogui.sleep(1)

    # Listen to the microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    text = ""  # Store the entire conversation

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            type_text(text)  # Type the initial text
            while True:
                print("Listening...")
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                if "stop writing" in text.lower():
                    break
                type_text(text)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Error; {0}".format(e))

    # Save text to clipboard and file
    time.sleep(1)  # Wait for the text to appear in Notepad
    pyautogui.hotkey("ctrl", "a")  # Select all text in Notepad
    pyautogui.hotkey("ctrl", "c")  # Copy selected text to clipboard

    clipboard_text = pyperclip.paste()

    with open("speech_to_text.txt", "w") as file:
        file.write(clipboard_text)

    print("Text saved to clipboard and file.")

    def kill_notepad():
        try:
            subprocess.run(["taskkill", "/f", "/im", "notepad.exe"], check=True)
            print("Notepad terminated successfully.")
        except subprocess.CalledProcessError:
            print("Failed to terminate Notepad.")

    kill_notepad()


if __name__ == "__main__":
    start_writing()
