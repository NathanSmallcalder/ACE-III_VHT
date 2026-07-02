import pyttsx3
import time

class TTSEngine:
    def __init__(self, rate=150, volume=1.0):
        self.rate = rate
        self.volume = volume
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        self._voice_id = next((v.id for v in voices if 'en' in v.id.lower()), None)
        engine.stop()

    def speak(self, text: str):
        print(f"[TTS] Saying: {text}")
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        if self._voice_id:
            engine.setProperty('voice', self._voice_id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.5)