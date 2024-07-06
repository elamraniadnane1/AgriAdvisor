 
import os
import speech_recognition as sr
from gtts import gTTS

def recognize_speech_from_microphone(language="ar"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "Speech was unintelligible"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"

def text_to_speech(text, language="ar"):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")
