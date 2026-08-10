import json

from langchain_core.messages import AIMessage, HumanMessage

from graph import graph, configure, ACE_DATA
from data_loader import get_session_config
from virtual_avatar.avatar import furhat_connect
from voice.tts import TTSEngine
from voice.capture import AudioCapture, FLUENCY_SILENCE_DURATION
from ui.session_window import SessionWindow

DOMAIN_ORDER = list(ACE_DATA.keys())

initial_state = {
    "messages": [],
    "current_domain": DOMAIN_ORDER[0],
    "question_index": 0,
    "sub_question_index": 0,
    "question_score": 0,
    "scores": {domain: 0 for domain in DOMAIN_ORDER},
    "domain_queue": [d for d in DOMAIN_ORDER if d not in (DOMAIN_ORDER[0], "Fluency")],  # Fluency is reached via advance_node's mid-Memory detour
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

def load_file(path):
    """Rebuild an ACEState from a results/progress/progress_*.json checkpoint
    written by graph.save_progress, so an interrupted session can resume
    exactly where it left off."""
    with open(path, "r") as f:
        data = json.load(f)

    messages = [
        (AIMessage if m["role"] == "ai" else HumanMessage)(content=m["content"])
        for m in data.get("messages", [])
    ]

    return {
        "messages": messages,
        "current_domain": data["current_domain"],
        "question_index": data["question_index"],
        "sub_question_index": data.get("sub_question_index", 0),
        "question_score": data.get("question_score", 0),
        "scores": data["scores"],
        "domain_queue": data["domain_queue"],
        "complete": data.get("complete", False),
        "needs_repeat": data.get("needs_repeat", False),
        "repeat_count": data.get("repeat_count", 0),
        "reprompt_kind": data.get("reprompt_kind"),
        "turn_progress": data.get("turn_progress", 0),
        "recall_matches": data.get("recall_matches", {}),
        "question_log": data.get("question_log", []),
        "question_turn_start": data.get("question_turn_start", 0),
        "previous_task_signature": data.get("previous_task_signature"),
    }

if __name__ == "__main__":
    gui = SessionWindow()
    gui.wait_for_start()

    if gui.load_path:
        initial_state = load_file(gui.load_path)

    gui.set_loading_status("Loading session configuration...")
    session_config = get_session_config()

    gui.set_loading_status("Connecting to Furhat...")
    furhat = furhat_connect()
    tts = TTSEngine(furhat)

    gui.set_loading_status("Initializing audio capture...")
    audio = AudioCapture()
    audio_fluency = AudioCapture(silence_timeout=FLUENCY_SILENCE_DURATION, model=audio.model)

    gui.show_session_layout()

    configure(session_config, tts, audio, audio_fluency, gui)
    graph.invoke(initial_state)
