import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from voice.config import NO_SPEECH_TIMEOUT, WHISPER_RATE, AUDIO_CHUNK, SILENCE_THRESHOLD, SILENCE_DURATION, MAX_RESPONSE_DURATION

class AudioCapture:
    def __init__(self, model_size="medium", silence_timeout=SILENCE_DURATION, model=None):
        self.sample_rate = WHISPER_RATE
        self.silence_timeout = silence_timeout

        if model is not None:
            self.model = model
        else:
            print("[Audio] Loading Whisper model... (")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def capture_response(self) -> str:
        print("[Audio] Listening for response...")

        recording_buffer = []
        silent_chunks_limit = int((self.silence_timeout * self.sample_rate) / AUDIO_CHUNK)
        silent_chunks_count = 0
        has_spoken = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while True:
                chunk, overflowed = stream.read(AUDIO_CHUNK)
                recording_buffer.append(chunk)

                energy = np.sqrt(np.mean(chunk**2))

                if energy > SILENCE_THRESHOLD:
                    has_spoken = True
                    silent_chunks_count = 0
                else:
                    if has_spoken:
                        silent_chunks_count += 1

                if has_spoken and silent_chunks_count > silent_chunks_limit:
                    print("[Audio] Silence detected. Processing speech...")
                    break

                if len(recording_buffer) * AUDIO_CHUNK > self.sample_rate * MAX_RESPONSE_DURATION:
                    print("[Audio] Max time limit reached. Processing...")
                    break

        audio_data = np.concatenate(recording_buffer, axis=0).flatten()

        segments, _ = self.model.transcribe(audio_data, beam_size=5,language="en", word_timestamps=False, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
        return " ".join([segment.text for segment in segments]).strip()