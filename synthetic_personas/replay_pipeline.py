"""
Replays a pre-generated synthetic-persona transcript (from evaluate_synthetic_personas.py)
through the REAL graph.py state machine, injecting the transcript's answers in place of
live Whisper audio capture / camera / click UI. This exercises the actual routing and
scoring logic (classify_turn, _HANDLERS, reprompt handling, VISUAL_SCORERS, ...) end to
end against a canned transcript instead of a live session.

Every item should score for real: spoken items and the 4 click items go through the
normal audio/click path; Infinity/Cube/Clock have real reference image paths already in
the transcript; Writing and pencil/paper Comprehension reply with real sentence text /
the real scorer's own JSON shape respectively, so their scorers are patched below
(_patch_visual_task_scoring) to score that directly instead of transcribing a photo or
video that doesn't exist.

Usage: python -m synthetic_personas.replay_pipeline synthetic_transcripts/NSHD100000_healthy.json
"""
import argparse
import json

from langchain_core.messages import AIMessage, HumanMessage

import graph
from data_loader import get_session_config
from visual_tasks import pen_paper_scorer, writing as writing_scorer
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


def _visual_and_click_question_texts():
    """question_text of every real question routed through run_visual_task/run_click_task
    -- those never call capture_response, so their transcript entries must be excluded
    from the audio queue or they'd just sit there unconsumed forever."""
    return {
        question["question_text"]
        for domain in graph.ACE_DATA.values()
        for question in domain["questions"]
        if question.get("match_type") == "visual" or is_click_point_question(question)
    }

def _flatten(transcript, domains, skip_questions):
    """All 'Awnsered' values across the given domains, in transcript order, excluding
    visual/click items (see _visual_and_click_question_texts)."""
    return [entry["Awnsered"] for domain in domains for entry in transcript.get(domain, [])
            if entry["Question"] not in skip_questions]

def _by_question_text(transcript):
    """question_text -> Awnsered, across every domain in the transcript."""
    return {
        entry["Question"]: entry["Awnsered"]
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
        print("Assessor (visual):", text)
        print("Patient (canned):", answer)
        return {"messages": [AIMessage(content=text), HumanMessage(content=answer)]}

    def stub_run_click_task(state, question, tts, session_config, next_question, gui):
        text = question["question_text"]
        answer = by_question.get(text, "")
        print("Assessor (click):", text)
        print("Patient (canned):", answer)
        return {"messages": [AIMessage(content=text), HumanMessage(content=answer)]}

    return stub_run_visual_task, stub_run_click_task


def _patch_visual_task_scoring():
    """The real Writing/pencil-paper scorers expect a photo/video to analyze via VLM.
    The replay's answers for those two are already real text/JSON (persona_scorer.py
    asks the LLM to reply with actual sentences / the real scorer's own JSON shape), so
    patch the underlying scorers to score that directly instead of trying to open a
    nonexistent file. marking.py imports both lazily (inside the function body) at call
    time, so patching the module attribute here is picked up automatically."""
    def score_writing_image(response_text):
        total = writing_scorer.score_sentence_writing(response_text) if response_text else 0
        return {"total": total}

    def score_pen_paper_video(response_text):
        try:
            data = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            data = {}
        return pen_paper_scorer.score_pen_paper(data)

    writing_scorer.score_writing_image = score_writing_image
    pen_paper_scorer.score_pen_paper_video = score_pen_paper_video


def run_replay(transcript_path):
    with open(transcript_path) as f:
        transcript = json.load(f)

    _patch_visual_task_scoring()
    session_config = get_session_config()

    skip_questions = _visual_and_click_question_texts()
    audio = ReplayAudio(_flatten(transcript, ["Attention", "Memory", "Language", "Visuospatial"], skip_questions))
    audio_fluency = ReplayAudio(_flatten(transcript, ["Fluency"], skip_questions))
    tts = NullTTS()
    gui = NullGUI()

    graph.configure(session_config, tts, audio, audio_fluency, gui)

    stub_visual, stub_click = _make_visual_stubs(transcript)
    graph.run_visual_task = stub_visual
    graph.run_click_task = stub_click

    domain_order = list(graph.ACE_DATA.keys())
    initial_state = {
        "messages": [],
        "current_domain": domain_order[0],
        "question_index": 0,
        "sub_question_index": 0,
        "question_score": 0,
        "scores": {domain: 0 for domain in domain_order},
        "domain_queue": domain_order[1:],
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
