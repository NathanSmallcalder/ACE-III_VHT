import sounddevice as sd
from kokoro import KPipeline

class TTSEngine:
    def __init__(self, voice="af_heart", speed=1.0, lang_code="a"):
        self.voice = voice
        self.speed = speed
        self._pipeline = KPipeline(lang_code=lang_code)

    def speak(self, text: str):
        print(f"[TTS] Saying: {text}")
        for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed):
            sd.play(audio.numpy(), samplerate=24000)
            sd.wait()
