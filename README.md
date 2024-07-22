
# OSKI: A Personalized Virtual Voice Assistant with Local Desktop Automation Features

OSKI is a versatile voice assistant designed to automate various desktop tasks and provide useful information through voice commands. This documentation provides an overview of the features, setup instructions, and usage of OSKI.

## Features

1. **Voice Interaction:**
   - Responds to voice commands and provides spoken feedback using `pyttsx3`.
   - Recognizes voice input using `speech_recognition`.

2. **Desktop Automation:**
   - Adjust system volume.
   - Control mouse and keyboard.
   - Open and close applications like Notepad, Command Prompt, and browsers.
   - Take screenshots and pictures.
   - Record screen with voice commands.

3. **Information Retrieval:**
   - Fetch latest news headlines.
   - Provide current weather information.
   - Perform Google and YouTube searches.

4. **Object Detection:**
   - Detect objects in real-time using YOLOv5 model.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

### Required Libraries

Install the required libraries using the following command:

```sh
pip install pyttsx3 SpeechRecognition openai pywhatkit google-api-python-client opencv-python numpy comtypes pycaw pyautogui Pillow psutil requests
```

### Configuration

1. **API Keys:**
   - Replace the placeholder API keys in the code with your own API keys for Google, NewsAPI, OpenWeather, and OpenAI.

2. **Voice Properties:**
   - Adjust the voice properties in the `Speak` function if necessary.

### File Structure

Ensure the following file structure for any additional scripts:

```plaintext
OSKI/
├── main_script.py
├── note.py
├── key.py
├── mouse.py
└── testface.py
```

## Usage

### Running OSKI

Execute the main script to start OSKI:

```sh
python main_script.py
```

### Available Commands

- **Greeting:**
  - "Hello"
  - "Goodbye"

- **Desktop Automation:**
  - "Take a screenshot"
  - "Take a picture"
  - "Adjust the volume"
  - "Open Notepad"
  - "Open Command Prompt"
  - "Close Chrome"
  - "Close Brave"

- **Information Retrieval:**
  - "Show weather"
  - "Show news"
  - "Google [search query]"
  - "YouTube [search query]"

- **Object Detection:**
  - "What can you see?"

### Example Usage

1. **Greeting:**
   - User: "Hello"
   - OSKI: "Hello Ayaan, welcome back"

2. **Take a Screenshot:**
   - User: "Take a screenshot"
   - OSKI: "Sure sir"

3. **Google Search:**
   - User: "Google Python tutorials"
   - OSKI: "Searching for Python tutorials"

4. **Weather Information:**
   - User: "Show weather"
   - OSKI: "Please say the name of the city."

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes to the branch.
5. Submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Inspired by various open-source voice assistants and automation tools.
- Utilizes the power of OpenAI for natural language processing.

## Contact

For any inquiries, please contact Ayaan Adil at [ayaanninja2403@gmail.com].
