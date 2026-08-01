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
    """Feeds pre-generated answers to capture_response() in order. If asked more times
    than there are answers (a reprompt/retry), repeats the last answer rather than
    crashing -- a consistent persona just says the same thing again."""

    def __init__(self, answers):
        self._answers = list(answers)
        self._index = 0

    def capture_response(self, on_tick=None):
        if self._index < len(self._answers):
            answer = self._answers[self._index]
            self._index += 1
        else:
            answer = self._answers[-1] if self._answers else ""
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
    (match_type=='visual' or a click-point question) -- those never consume from _audio,
    so their transcript entries must be excluded when building the audio queue, or they'd
    just sit there unconsumed and permanently shift every later question out of position."""
    texts = set()
    for domain in graph.ACE_DATA.values():
        for question in domain["questions"]:
            if question.get("match_type") == "visual" or is_click_point_question(question):
                texts.add(question["question_text"])
    return texts


def _repeat_count(entry):
    """How many times the real state machine will call capture_response() for this
    question, regardless of how good the model is -- both are deterministic, not
    LLM-compliance noise: _handle_name_address_trials always repeats 3 times, and
    _handle_registration repeats up to 3 times whenever the ground truth isn't a
    perfect score (it only stops early on an all-correct trial)."""
    question = entry["Question"]
    if question.startswith("Name and address learning:"):
        return 3
    if question.startswith("Registration:"):
        return 1 if entry["Ground_Truth"] == entry["Score_Cap"] else 3
    return 1


def _flatten(transcript, domains, skip_questions):
    """All 'Awnsered' values across the given domains, in transcript order, excluding
    visual/click items (see _visual_and_click_question_texts) and repeating trial-based
    questions the same number of times the real state machine will ask them (see
    _repeat_count) so the queue doesn't desync on the very first one."""
    answers = []
    for domain in domains:
        for entry in transcript.get(domain, []):
            if entry["Question"] in skip_questions:
                continue
            answers.extend([entry["Awnsered"]] * _repeat_count(entry))
    return answers


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
