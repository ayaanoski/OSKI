import pyaudio
import sounddevice as sd
import numpy as np

threshold = 2.0
Clap = False


def detect_clap(indata, frames, time, status):
    global Clap
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > threshold:
        print("clapped!")
        Clap = True


def listen_for_clap():
    with sd.InputStream(callback=detect_clap):
        return sd.sleep(1000)


def mainclapexe():
    while True:
        listen_for_clap()
        if Clap == True:
            break

        else:
            pass
