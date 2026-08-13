import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

silence_threshold = 0.05
silence_duration = 2
max_response = 60
fluency_task_silence = 15

class AudioCapture:
    def __init__(self, model_size="large", silence_timeout=silence_duration, model=None):
        self.sample_rate = 16000
        self.silence_timeout = silence_timeout
        
        if model is not None:
            self.model = model
        else:
            print("[Audio] Loading Whisper model... (")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def capture_response(self, on_tick=None, question_key=None, max_duration=max_response) -> str:
        """"""
        print("[Audio] Listening for response...")

        recording_buffer = []
        silent_chunks_limit = int((self.silence_timeout * self.sample_rate) / 1024)
        silent_chunks_count = 0
        has_spoken = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while True:
                chunk, _overflowed = stream.read(1024)
                recording_buffer.append(chunk)
                if on_tick is not None:
                    on_tick()

                energy = np.sqrt(np.mean(chunk**2))

                if energy > silence_duration:
                    has_spoken = True
                    silent_chunks_count = 0
                else:
                    if has_spoken:
                        silent_chunks_count += 1

                if has_spoken and silent_chunks_count > silent_chunks_limit:
                    print("[Audio] Silence detected. Processing speech...")
                    break

                if len(recording_buffer) * 1024 > self.sample_rate * max_duration:
                    print("[Audio] Max time limit reached. Processing...")
                    break

        audio_data = np.concatenate(recording_buffer, axis=0).flatten()

        segments, _ = self.model.transcribe(audio_data, beam_size=5,language="en", word_timestamps=False, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500})
        return " ".join([segment.text for segment in segments]).strip()