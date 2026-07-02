from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from rapidfuzz import fuzz
import json
from LLM.dialogue import *
from marking.marking import *
from voice.capture import AudioCapture
from voice.tts import TTSEngine
from voice.config import FLUENCY_SILENCE_DURATION, FLUENCY_MAX_RESPONSE_DURATION
from data_loader import get_session_config, resolve_dynamic_answers
from visual_tasks.visual import run_visual_task

class ACEState(MessagesState):
    current_domain: str
    question_index: int
    sub_question_index: int
    question_score: int
    scores: dict
    domain_queue: list
    complete: bool
    needs_repeat: bool
    repeat_count: int

_session_config = get_session_config()
_tts = TTSEngine()
_audio = AudioCapture()
_audio_fluency = AudioCapture(silence_timeout=FLUENCY_SILENCE_DURATION, model=_audio.model)

with open("json/ACE-III.json", "r") as f:
    ACE_DATA = json.load(f)["Domains"]

for _domain in ACE_DATA.values():
    for _question in _domain["questions"]:
        _question["answers"] = resolve_dynamic_answers(_question["answers"], _session_config)

def conversation_node(state: ACEState) -> dict:
    domain = state["current_domain"]
    q_index = state["question_index"]
    sub_index = state.get("sub_question_index", 0)
    question = ACE_DATA[domain]["questions"][q_index]

    if question.get("match_type") == "visual":
        return run_visual_task(state, question, _tts, _audio, _session_config)

    prompts = get_sub_prompts(question)
    if prompts and sub_index < len(prompts):
        text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]

    if state.get("needs_repeat"):
        wrapper = rephrase_question(text)
    elif not state["messages"]:
        wrapper = introduce(_session_config["patient"]["name"])
    elif sub_index == 0:
        last_patient = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None
        )
        wrapper = acknowledge(last_patient) if last_patient else ""
    else:
        wrapper = ""

    spoken_text = f"{wrapper} {text}".strip() if wrapper else text

    print("Assessor:", spoken_text)
    _tts.speak(spoken_text)

    for _ in range(5):
        user_input = _audio_fluency.capture_response() if domain == "Fluency" else _audio.capture_response()

        if not user_input:
            print("[no response detected]")
            continue

        if fuzz.partial_ratio(user_input.lower(), spoken_text.lower()) > 75:
            print("[echo detected, ignoring]")
            continue

        break
    else:
        user_input = ""

    if not user_input:
        return {"needs_repeat": True}

    print("Patient:", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}


def scoring_node(state: ACEState) -> dict:
    domain = state['current_domain']
    q_index = state['question_index']
    sub_index = state.get("sub_question_index", 0)
    question = ACE_DATA[domain]['questions'][q_index]

    last_message = state['messages'][-1].content

    expected_answers = question['answers']
    prompts = get_sub_prompts(question)
    is_multi = bool(prompts)

    if prompts and sub_index < len(prompts):
        asked_text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        asked_text = spoken[0] if spoken else question["question_text"]

    # Stochastic admission gate: decides whether the deterministic scorer runs.
    # Non-answer + under the cap -> re-prompt without scoring.
    # Non-answer + cap hit -> fall through and score whatever was given.
    repeats = state.get("repeat_count", 0)
    max_repeats = question.get("max_attempts", 1)
    match_type = question.get("match_type", "")
    is_fluency = match_type in ("fluency_letter", "fluency_animal")
    if not is_fluency and classify_turn(last_message, asked_text) != "answer" and repeats < max_repeats:
        return {"needs_repeat": True, "repeat_count": repeats + 1}

    # --- deterministic marking from here down ---

    if is_multi:
        score = score_question(last_message, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
        new_sub = sub_index + 1
    else:
        score = score_question(last_message, question)
        new_sub = 0

    if score is None:
        return {"needs_repeat": False, "repeat_count": 0, "sub_question_index": new_sub}

    question_cap = question["score_cap"]
    question_score_so_far = state.get("question_score", 0)
    score = min(score, question_cap - question_score_so_far)

    domain_cap = ACE_DATA[domain]["score_cap"]
    current_scores = dict(state["scores"])
    domain_score_so_far = current_scores.get(domain, 0)
    score = min(score, domain_cap - domain_score_so_far)

    current_scores[domain] = domain_score_so_far + score

    return {
        "scores": current_scores,
        "sub_question_index": new_sub,
        "question_score": question_score_so_far + score,
        "needs_repeat": False,
        "repeat_count": 0,
    }

def advance_node(state: ACEState) -> dict:
    domain = state["current_domain"]
    q_index = state["question_index"]
    total_questions = len(ACE_DATA[domain]["questions"])

    if q_index + 1 < total_questions:
        return {"question_index": q_index + 1, "sub_question_index": 0, "question_score": 0}
    else:
        q = state["domain_queue"].copy()
        next_domain = q.pop(0)
        return {
            "current_domain": next_domain,
            "question_index": 0,
            "sub_question_index": 0,
            "question_score": 0,
            "domain_queue": q
        }

def report_node(state: ACEState) -> dict:
    print("\n--- ACE-III Complete ---")
    for domain, score in state["scores"].items():
        print(f"{domain}: {score}")
    return {"complete": True}

# Routes Flow of StateMachine
def router(state: ACEState) -> str:
    if state.get("needs_repeat"):
        return "repeat_question"
    # next_sub_question | next_question | next_domain | report
    domain = state['current_domain']
    q_index = state['question_index']
    sub_index = state.get('sub_question_index', 0)
    question = ACE_DATA[domain]["questions"][q_index]
    prompts = get_sub_prompts(question)

    if prompts and sub_index < len(prompts):
        return "next_sub_question"
    
    total_questions = len(ACE_DATA[domain]["questions"])
    if q_index + 1 < total_questions:
        return "next_question"
    elif state['domain_queue']:
        return "next_domain"
    else:
        return "report"

builder = StateGraph(ACEState)
builder.add_node("conversation", conversation_node)
builder.add_node("scoring", scoring_node)
builder.add_node("advance", advance_node)
builder.add_node("report", report_node)

builder.add_edge(START, "conversation")
builder.add_edge("conversation", "scoring")
builder.add_conditional_edges("scoring", router, {
    "repeat_question": "conversation",
    "next_sub_question": "conversation",
    "next_question": "advance",
    "next_domain": "advance",
    "report": "report",
})
builder.add_edge("advance", "conversation")
builder.add_edge("report", END)

graph = builder.compile()