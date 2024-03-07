import speech_recognition as sr
import pyautogui


def mainkey():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                audio_data = recognizer.listen(source)
                text = recognizer.recognize_google(audio_data).lower()
                print("You said:", text)
                if "exit" in text:
                    print("Exiting program.")
                    break
                elif "space" in text:
                    print("Pressing the space bar...")
                    pyautogui.press("space")
                elif "back" in text:
                    print("Pressing the backspace...")
                    pyautogui.press("backspace")
                elif "select all " in text:
                    print("Pressing the backspace...")
                    pyautogui.hotkey("ctrl", "a")
                else:
                    # Use the text to perform corresponding keyboard actions
                    pyautogui.write(text)

            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


mainkey()
