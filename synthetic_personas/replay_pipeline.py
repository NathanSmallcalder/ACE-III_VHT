"""
python -m synthetic_personas.replay_pipeline synthetic_transcripts/NSHD100000_healthy.json
"""
import argparse
import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
from data_loader import ace_json, get_session_config
from visual_tasks.visual import is_click_point_question


class ReplayAudio:
    """Feeds pre-generated answers to capture_response() in order. Only advances to
    the next answer when question_key changes from the previous call -- so a reprompt
    (registration/name-address trials, a non-answer retry, person_name follow-ups,
    season, ...) correctly repeats the same answer instead of consuming the next
    question's answer. graph.py passes question_key=(domain, q_index, sub_index),
    which only changes once a question is actually finalized."""

    def __init__(self, answers):
        self._answers = list(answers)
        self._index = -1
        self._last_key = object()  # sentinel: never equals a real question_key

    def capture_response(self, on_tick=None, question_key=None):
        if question_key != self._last_key:
            self._last_key = question_key
            self._index += 1
        answer = self._answers[self._index] if self._index < len(self._answers) else ""
        print("Patient:", answer)
        return answer

class NullTTS:
    furhat = None  # no robot to gesture with during a replay

    def speak(self, text):
        print("Assessor:", text)

class NullGUI:
    def add_message(self, role, text):
        pass

    def get_stage_frame(self):
        return None

    def stage_size(self):
        return (800, 600)

    def show_stimulus_image(self, path):
        pass

    def pump(self):
        pass

    def close(self):
        pass

    def set_on_close(self, callback):
        pass


def _visual_and_click_question_texts():
    """question_text of every real question routed through run_visual_task/run_click_task
    -- those never call capture_response, so their transcript entries must be excluded
    from the audio queue or they'd just sit there unconsumed forever."""
    return {
        question["question_text"]
        for domain in ace_json.values()
        for question in domain["questions"]
        if question.get("match_type") in ("clock", "pen_paper", "cube", "infinity", "sentances") or is_click_point_question(question)
    }

def _flatten(transcript, domains, skip_questions):
    """All 'Answered' values across the given domains, in transcript order, excluding
    visual/click items (see _visual_and_click_question_texts)."""
    return [entry["Answered"] for domain in domains for entry in transcript.get(domain, [])
            if entry["Question"] not in skip_questions]

def _by_question_text(transcript):
    """question_text -> Answered, across every domain in the transcript."""
    return {
        entry["Question"]: entry["Answered"]
        for items in transcript.values() if isinstance(items, list)
        for entry in items
    }

def _make_visual_stubs(transcript):
    """Replacements for graph.run_visual_task / graph.run_click_task -- feed the
    transcript's answer for that question instead of driving a real camera, video
    recorder, or Tkinter click canvas."""
    by_question = _by_question_text(transcript)

    def stub_run_visual_task(state, question, tts, audio, session_config, gui):
        text = question["question_text"]
        answer = by_question.get(text, "")
        print("Assessor:", text)
        print("Patient :", answer)
        return {"messages": [AIMessage(content=text), HumanMessage(content=answer)]}

    def stub_run_click_task(state, question, tts, session_config, next_question, gui):
        text = question["question_text"]
        answer = by_question.get(text, "")
        print("Assessor:", text)
        print("Patient:", answer)
        return {"messages": [AIMessage(content=text), HumanMessage(content=answer)]}

    return stub_run_visual_task, stub_run_click_task


def run_replay(transcript_path):
    with open(transcript_path) as f:
        transcript = json.load(f)

    session_config = get_session_config()
    session_config['patient']['name'] = f"{transcript.get('participant_id')}_{transcript.get('cognitive_status')}"

    skip_questions = _visual_and_click_question_texts()
    audio = ReplayAudio(_flatten(transcript, ["Attention", "Memory", "Language", "Visuospatial"], skip_questions))
    audio_fluency = ReplayAudio(_flatten(transcript, ["Fluency"], skip_questions))
    tts = NullTTS()
    gui = NullGUI()

    graph.configure(session_config, tts, audio, audio_fluency, gui)

    stub_visual, stub_click = _make_visual_stubs(transcript)
    graph.run_visual_task = stub_visual
    graph.run_click_task = stub_click

    domain_order = list(ace_json.keys())
    initial_state = {
        "messages": [],
        "current_domain": domain_order[0],
        "question_index": 0,
        "sub_question_index": 0,
        "question_score": 0,
        "scores": {domain: 0 for domain in domain_order},
        "domain_queue": [d for d in domain_order[1:] if d != "Fluency"],  # Fluency is reached via advance_node's mid-Memory detour
        "complete": False,
        "needs_repeat": False, 
        "repeat_count": 0,
        "reprompt_kind": None,
        "turn_progress": 0,
        "recall_matches": {},
        "question_log": [],
        "question_turn_start": 0,
        "previous_task_signature": None,
    }

    print(f"Replaying {transcript.get('participant_id')} ({transcript.get('cognitive_status')})...")
    graph.graph.invoke(initial_state, config={"recursion_limit": 1000})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_path")
    args = parser.parse_args()
    run_replay(args.transcript_path)
