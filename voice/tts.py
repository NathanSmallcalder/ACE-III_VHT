
class TTSEngine:
    def __init__(self, furhat):
        self.furhat = furhat
        furhat.set_voice(name="JennyNeural")

    def speak(self, text: str, rate="slow"):
        print(f"[TTS] Saying: {text}")
        natural_text = text.replace(". ", '. <break time="350ms" /> ') # Break time on full stops
        natural_text = natural_text.replace(", ", ', <break time="250ms" /> ') # Break time on commas
        ssml = f'<prosody rate="{rate}">{natural_text}</prosody>'
        self.furhat.say(text=ssml, blocking=True)
