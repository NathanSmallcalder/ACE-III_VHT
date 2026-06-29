import pyttsx3
import time

class TTSEngine:
    def __init__(self, rate=150, volume=1.0):
        """
        Initializes a local, cross-platform Text-to-Speech engine.
        Rate is set to 150 by default (slower than normal 200) for clarity.
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voices
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)

    def speak(self, text: str):
        """Speaks the text aloud and blocks until finished."""
        print(f"[TTS] Saying: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        # Small padding pause to let the audio hardware clear before listening
        time.sleep(0.5)