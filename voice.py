import queue
import threading
import time
import wave

import numpy as np
import pyaudio
import pyttsx3
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

AUDIO_RATE = 44100
WHISPER_RATE = 16000
AUDIO_CHUNK = 1024
SILENCE_THRESHOLD = 0.05
SILENCE_DURATION = 1.5
NO_SPEECH_TIMEOUT = 15
MAX_RESPONSE_DURATION = 30
FLUENCY_SILENCE_DURATION = 10
FLUENCY_MAX_RESPONSE_DURATION = 65


def speak(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def drain_queue(q, settle_time=1.0):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
    time.sleep(settle_time)
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def listen_for_response(transcribe_q, whisper_model,
                        silence_duration=SILENCE_DURATION,
                        max_duration=MAX_RESPONSE_DURATION) -> str:
    print("Listening...")
    buffer = []
    speech_started = False
    silence_chunks = 0
    waiting_chunks = 0
    silence_chunk_limit = int(silence_duration * AUDIO_RATE / AUDIO_CHUNK)
    no_speech_chunk_limit = int(NO_SPEECH_TIMEOUT * AUDIO_RATE / AUDIO_CHUNK)
    max_chunks = int(max_duration * AUDIO_RATE / AUDIO_CHUNK)

    while True:
        data = transcribe_q.get()
        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        peak = np.max(np.abs(chunk))

        if peak > SILENCE_THRESHOLD:
            speech_started = True
            silence_chunks = 0
            buffer.append(chunk)
        elif speech_started:
            silence_chunks += 1
            buffer.append(chunk)
            if silence_chunks >= silence_chunk_limit:
                break
        else:
            waiting_chunks += 1
            if waiting_chunks >= no_speech_chunk_limit:
                return ""

        if speech_started and len(buffer) >= max_chunks:
            break

    audio = np.concatenate(buffer)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    gcd = np.gcd(WHISPER_RATE, AUDIO_RATE)
    audio = resample_poly(audio, WHISPER_RATE // gcd, AUDIO_RATE // gcd).astype(np.float32)

    segments, _ = whisper_model.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        language="en",
        condition_on_previous_text=False,
        no_speech_threshold=0.7,
    )
    return "".join(s.text for s in segments).strip()


def _capture_loop(stream, record_q, transcribe_q):
    while True:
        data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
        record_q.put(data)
        transcribe_q.put(data)


def _recorder_loop(record_q, wf):
    while True:
        data = record_q.get()
        if data is None:
            break
        wf.writeframes(data)
    wf.close()


def setup_audio():
    """Opens the microphone, starts the Whisper model, and launches capture/recorder threads.

    Returns (p, stream, record_q, transcribe_q, whisper_model).
    Caller is responsible for cleanup: record_q.put(None), stream.stop_stream(),
    stream.close(), p.terminate().
    """
    p = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']} (default rate: {int(info['defaultSampleRate'])} Hz)")

    default_index = p.get_default_input_device_info()["index"]
    choice = input(f"Select input device index (default {default_index}): ").strip()
    try:
        device_index = int(choice) if choice else default_index
    except ValueError:
        device_index = default_index

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=AUDIO_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=AUDIO_CHUNK,
    )

    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

    record_q = queue.Queue()
    transcribe_q = queue.Queue()

    wf = wave.open("session.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(AUDIO_RATE)

    threading.Thread(target=_capture_loop, args=(stream, record_q, transcribe_q), daemon=True).start()
    threading.Thread(target=_recorder_loop, args=(record_q, wf), daemon=True).start()

    return p, stream, record_q, transcribe_q, whisper_model

