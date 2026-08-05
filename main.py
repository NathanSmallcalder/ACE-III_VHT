from graph import graph, configure, ACE_DATA
from data_loader import get_session_config
from virtual_avatar.avatar import furhat_connect
from voice.tts import TTSEngine
from voice.capture import AudioCapture, FLUENCY_SILENCE_DURATION
from ui.session_window import SessionWindow

DOMAIN_ORDER = list(ACE_DATA.keys())
START_DOMAIN = "Memory"   # DOMAIN_ORDER[0]

initial_state = {
    "messages": [],
    "current_domain": START_DOMAIN,
    "question_index": 0,
    "sub_question_index": 0,
    "question_score": 0,
    "scores": {domain: 0 for domain in DOMAIN_ORDER},
    "domain_queue": [d for d in DOMAIN_ORDER if d not in (START_DOMAIN, "Fluency")],  # Fluency is reached via advance_node's mid-Memory detour
    "complete": False,
    "needs_repeat": False,
    "repeat_count": 0,
    "reprompt_kind": None,  # None | "season" | "name" | "leader" | "trial" — which handler's reprompt is active
    "turn_progress": 0,     # per-question counter, meaning depends on the active handler
    "recall_matches": {},   # recall_key -> per-answer bool list, for recognition-task skip logic
    "question_log": [],
    "question_turn_start": 0,
    "previous_task_signature": None,
}

if __name__ == "__main__":
    session_config = get_session_config()
    furhat = furhat_connect()
    tts = TTSEngine(furhat)
    audio = AudioCapture()
    audio_fluency = AudioCapture(silence_timeout=FLUENCY_SILENCE_DURATION, model=audio.model)
    gui = SessionWindow()

    configure(session_config, tts, audio, audio_fluency, gui)
    graph.invoke(initial_state)
