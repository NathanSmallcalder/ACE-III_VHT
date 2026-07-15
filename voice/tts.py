class TTSEngine:
    def __init__(self, furhat):
        self.furhat = furhat

    def speak(self, text: str):
        print(f"[TTS] Saying: {text}")
        self.furhat.say(text=text, blocking=True)  # returns when speech ends
