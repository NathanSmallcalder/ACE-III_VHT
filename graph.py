from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from rapidfuzz import fuzz
from PIL import Image
import json
import os
from LLM.dialogue import *
from marking.marking import *
from voice.tts import TTSEngine
from voice.capture import AudioCapture
from voice.config import FLUENCY_SILENCE_DURATION
from datetime import datetime
from data_loader import get_session_config, resolve_dynamic_answers, get_season_transition
from visual_tasks.visual import run_visual_task, run_click_task, is_click_point_question
from virtual_avatar.avatar import furhat_connect

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
    reprompt_kind: str   # None | "season" | "name" | "leader" | "trial"
    turn_progress: int   # generic per-question counter; meaning depends on the active handler
    recall_matches: dict  # recall_key -> per-answer bool list, for recognition-task skip logic

_session_config = get_session_config()
_furhat = furhat_connect()
_tts = TTSEngine(_furhat)
_audio = AudioCapture()
_audio_fluency = AudioCapture(silence_timeout=FLUENCY_SILENCE_DURATION, model=_audio.model)

with open("json/ACE-III.json", "r") as f:
    ACE_DATA = json.load(f)["Domains"]

_SEASON_TRUE, _SEASON_ADJACENT = get_season_transition(datetime.now())

for _domain in ACE_DATA.values():
    for _question in _domain["questions"]:
        _original_answers = _question["answers"]
        if "DYNAMIC:season" in _original_answers:
            _question["season_sub_index"] = _original_answers.index("DYNAMIC:season")
        # Only set when session_config records a recent leadership change —
        # the outgoing-leader probe is a no-op otherwise.
        if "DYNAMIC:uk_prime_minister" in _original_answers and _session_config.get("previous_uk_pm"):
            _question["outgoing_leader"] = _session_config["previous_uk_pm"]
        if "DYNAMIC:us_president" in _original_answers and _session_config.get("previous_us_president"):
            _question["outgoing_leader"] = _session_config["previous_us_president"]
        _question["answers"] = resolve_dynamic_answers(_original_answers, _session_config)


def _finalize_score(state, question, domain, q_index, score, new_sub) -> dict:
    """Apply question/domain caps, add to the running score, and clear all
    reprompt/progress state. Every handler and the default scoring path end
    here, so there's exactly one place that knows how to close out a turn."""
    question_cap = question["score_cap"]
    question_score_so_far = state.get("question_score", 0)
    score = min(score, question_cap - question_score_so_far)

    domain_cap = ACE_DATA[domain]["score_cap"]
    current_scores = dict(state["scores"])
    domain_score_so_far = current_scores.get(domain, 0)
    score = min(score, domain_cap - domain_score_so_far)

    current_scores[domain] = domain_score_so_far + score
    print(f"Scored {score} points for question {q_index + 1} in domain '{domain}' (total: {current_scores[domain]})")
    return {
        "scores": current_scores,
        "sub_question_index": new_sub,
        "question_score": question_score_so_far + score,
        "needs_repeat": False,
        "repeat_count": 0,
        "reprompt_kind": None,
        "turn_progress": 0,
    }


def _reprompt(kind: str) -> dict:
    """Trigger a handler-specific reprompt turn (no score yet)."""
    return {"needs_repeat": True, "repeat_count": 0, "reprompt_kind": kind}


# --- person_name: surname-only reprompt + outgoing-leader probe ---

def _handle_person_name(state, question, response, domain, q_index, sub_index):
    outgoing = question.get("outgoing_leader")

    if state.get("reprompt_kind") == "leader":
        score = score_person_name(response, [outgoing, outgoing.split()[-1]]) if outgoing else 0
        return _finalize_score(state, question, domain, q_index, score, 0)

    if state.get("reprompt_kind") == "name":
        return _finalize_score(state, question, domain, q_index, score_question(response, question), 0)

    score = score_question(response, question)
    if score:
        return _finalize_score(state, question, domain, q_index, score, 0)

    # A single bare word that isn't the surname reads as "first name only"
    # (e.g. "Maggie") — ask for the surname before giving up on the question.
    if len(clean_response(response).split()) == 1:
        return _reprompt("name")

    # Wrong (and not just incomplete) — if there's been a recent change of
    # leader, probe for the outgoing politician's name as an alternate credit.
    if outgoing:
        return _reprompt("leader")

    return _finalize_score(state, question, domain, q_index, 0, 0)


def _reprompt_text_person_name(state, question, text):
    if state.get("reprompt_kind") == "name":
        return "And what was their surname?"
    if state.get("reprompt_kind") == "leader":
        return "Who was the previous one, before them?"
    return None


# --- word-learning trials (Registration): repeat up to max_attempts times to
# help the patient learn, but only the *first* attempt counts for score ---

def _handle_registration(state, question, response, domain, q_index, sub_index):
    max_trials = question.get("max_attempts", 3)
    trial = state.get("turn_progress", 0)
    attempt_score = score_question(response, question)
    first_attempt_score = attempt_score if trial == 0 else state.get("question_score", 0)
    all_correct = attempt_score == question["score_cap"]
    trial += 1
    print(f"Registration trial {trial}/{max_trials}: {attempt_score}/{question['score_cap']} this trial "
          f"(first-attempt score: {first_attempt_score})")

    if not all_correct and trial < max_trials:
        return {
            "needs_repeat": True, "repeat_count": 0,
            "turn_progress": trial, "reprompt_kind": "registration",
            "question_score": first_attempt_score,
        }

    # first_attempt_score is already fully determined — don't let
    # _finalize_score's question_score_so_far cap-subtraction (meant for
    # incremental sub-answer sums) double-count the value we stashed there
    # between trials; present a fresh view for that calculation.
    return _finalize_score({**state, "question_score": 0}, question, domain, q_index, first_attempt_score, 0)


def _reprompt_text_registration(state, question, text):
    return text if state.get("reprompt_kind") == "registration" else None


# --- season transition leniency (Orientation to Time, season sub-question only) ---
def _handle_season(state, question, response, domain, q_index, sub_index):
    expected_answers = question["answers"]
    # Near a season boundary: if the patient names the *upcoming* season
    # instead of the current one, ask them to confirm rather than marking it
    # wrong outright — only score correct if they then name the true season.
    if (
        state.get("reprompt_kind") != "season" and _SEASON_ADJACENT
        and score_fuzzy(response, [_SEASON_ADJACENT]) and not score_fuzzy(response, [_SEASON_TRUE])
    ):
        return _reprompt("season")

    score = score_question(response, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
    return _finalize_score(state, question, domain, q_index, score, sub_index + 1)


def _reprompt_text_season(state, question, text):
    return "Could it be another season?" if state.get("reprompt_kind") == "season" else None


# --- discontinuous banding over sub-answer count (e.g. word repetition) ---
def _handle_sub_score_bands(state, question, response, domain, q_index, sub_index):
    expected_answers = question["answers"]
    word_score = score_question(response, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
    correct_so_far = state.get("turn_progress", 0) + (1 if word_score else 0)
    new_sub = sub_index + 1

    if new_sub < len(expected_answers):
        return {
            "needs_repeat": False, "repeat_count": 0,
            "sub_question_index": new_sub, "turn_progress": correct_so_far,
        }

    score = scaled_count(correct_so_far, question["sub_score_bands"])
    return _finalize_score(state, question, domain, q_index, score, new_sub)


# mid-sequence turns use default phrasing, not a reprompt
def _reprompt_text_word_rep(state, question, text):
    return None


# --- fixed learning trials, only the final trial's response is scored ---
def _handle_name_address_trials(state, question, response, domain, q_index, sub_index):
    total_trials = question.get("trials", 3)
    trial = state.get("turn_progress", 0) + 1
    attempt_score = score_question(response, question)
    print(f"Name & address trial {trial}/{total_trials}: {attempt_score}/{question['score_cap']} this trial")

    if trial < total_trials:
        return {"needs_repeat": True, "repeat_count": 0, "turn_progress": trial, "reprompt_kind": "trial"}

    return _finalize_score(state, question, domain, q_index, attempt_score, 0)


# Learning trials restate the name and address verbatim, not an LLM
# paraphrase — same words every trial is the point of the repetition.
def _reprompt_text_name_address(state, question, text):
    return text if state.get("reprompt_kind") == "trial" else None


# --- recognition: skip elements already credited in a linked recall question ---

def _recognition_recalled(question, sub_index, state):
    """True if this recognition element's tokens were already fully credited
    in the linked delayed-recall question, so it doesn't need to be re-asked."""
    element_indices = question.get("element_recall_indices")
    if not element_indices or sub_index >= len(element_indices):
        return False
    recalled = state.get("recall_matches", {}).get(question.get("recall_key"), [])
    return all(idx < len(recalled) and recalled[idx] for idx in element_indices[sub_index])


def _handle_recognition(state, question, response, domain, q_index, sub_index):
    new_sub = sub_index + 1
    if _recognition_recalled(question, sub_index, state):
        return _finalize_score(state, question, domain, q_index, 1, new_sub)
    score = score_question(response, question, sub_index=sub_index)
    return _finalize_score(state, question, domain, q_index, score, new_sub)


def _reprompt_text_recognition(state, question, text):
    return None


# kind -> (handle_fn, reprompt_text_fn). skip_turn_gate is True for every kind
# except "season" (see _match_kind for what a partial/short answer means per kind).
_HANDLERS = {
    "person_name": (_handle_person_name, _reprompt_text_person_name),
    "registration": (_handle_registration, _reprompt_text_registration),
    "season": (_handle_season, _reprompt_text_season),
    "word_rep": (_handle_sub_score_bands, _reprompt_text_word_rep),
    "name_address": (_handle_name_address_trials, _reprompt_text_name_address),
    "recognition": (_handle_recognition, _reprompt_text_recognition),
}


def _match_kind(question, sub_index):
    if question.get("match_type") == "person_name":
        return "person_name"
    if question.get("score_first_attempt_only"):
        # a short/partial recall (e.g. "lemon") is a complete, valid attempt
        return "registration"
    if question.get("season_sub_index") is not None and sub_index == question["season_sub_index"]:
        return "season"
    if question.get("sub_score_bands"):
        return "word_rep"
    if question.get("score_final_trial_only"):
        # a partial recall mid-trial (e.g. only 4 of 7 elements) is a genuine,
        # complete turn — classify_turn tends to read it as "incomplete" since
        # it isn't the full phrase, which would wrongly trigger a generic
        # re-ask instead of just tallying this trial's score.
        return "name_address"
    if question.get("element_recall_indices"):
        return "recognition"
    return None

def conversation_node(state: ACEState) -> dict:
    domain = state["current_domain"]
    q_index = state["question_index"]
    sub_index = state.get("sub_question_index", 0)
    question = ACE_DATA[domain]["questions"][q_index]

    if _match_kind(question, sub_index) == "recognition" and _recognition_recalled(question, sub_index, state):
        # Already credited via the linked recall question — no need to ask.
        return {"messages": [AIMessage(content=""), HumanMessage(content="")]}

    if question.get("match_type") == "visual":
        return run_visual_task(state, question, _tts, _audio, _session_config)

    if is_click_point_question(question):
        total_questions = len(ACE_DATA[domain]["questions"])
        next_question = ACE_DATA[domain]["questions"][q_index + 1] if q_index + 1 < total_questions else None
        return run_click_task(state, question, _tts, _session_config, next_question)

    # Show once, on the first presentation of the question — not on every
    # repeat/reprompt turn, which would otherwise reopen a new viewer window.
    if question.get("image") and not state.get("needs_repeat"):
        Image.open(question["image"]).show()

    prompts = get_sub_prompts(question)
    if prompts and sub_index < len(prompts):
        text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]

    kind = _match_kind(question, sub_index)
    reprompt = _HANDLERS[kind][1](state, question, text) if kind else None

    if reprompt:
        spoken_text = reprompt
    elif state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        if not state["messages"]:
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
            print("echo detected, ignoring")
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

    # Non-answer + under the retry cap -> re-prompt without scoring.
    # Non-answer + cap hit -> fall through and score whatever was given.
    repeats = state.get("repeat_count", 0)
    max_repeats = question.get("max_attempts", 1)
    match_type = question.get("match_type", "")
    is_fluency = match_type in ("fluency_letter", "fluency_animal")

    kind = _match_kind(question, sub_index)

    skip_turn_gate = (
        is_fluency or match_type == "visual" or is_click_point_question(question)
        or (kind is not None and kind != "season")
    )
    if not skip_turn_gate and classify_turn(last_message, asked_text) != "answer" and repeats < max_repeats:
        return {"needs_repeat": True, "repeat_count": repeats + 1}

    if kind:
        return _HANDLERS[kind][0](state, question, last_message, domain, q_index, sub_index)

    if is_multi:
        score = score_question(last_message, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
        new_sub = sub_index + 1
    else:
        score = score_question(last_message, question)
        new_sub = 0

    if score is None:
        return {
            "needs_repeat": False, "repeat_count": 0,
            "reprompt_kind": None, "turn_progress": 0, "sub_question_index": new_sub,
        }

    result = _finalize_score(state, question, domain, q_index, score, new_sub)
    recall_key = question.get("recall_key")
    if recall_key and not is_multi:
        # Stash per-element results so a later recognition question can skip
        # elements already credited here.
        matches = score_mixed_list_detailed(last_message, expected_answers)
        result["recall_matches"] = {**state.get("recall_matches", {}), recall_key: matches}
    return result

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _interpret_ace_total(total: int) -> str:
    # Cutoffs and sensitivity/specificity from the ACE-III administration guide.
    if total >= 88:
        return "At or above 88 — not indicative of dementia by either ACE-III cutoff."
    if total >= 82:
        return "Below 88 but at/above 82 — flagged by the more sensitive (88) cutoff only."
    return "Below both 88 and 82 — flagged by both ACE-III cutoffs."


def report_node(state: ACEState) -> dict:
    scores = state["scores"]
    total = sum(scores.values())
    interpretation = _interpret_ace_total(total)

    print("\n--- ACE-III Complete ---")
    for domain, score in scores.items():
        print(f"{domain}: {score}/{ACE_DATA[domain]['score_cap']}")
    print(f"Total: {total}/100")
    print(interpretation)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    patient_name = _session_config.get("patient", {}).get("name", "unknown")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(patient_name))
    out_path = os.path.join(RESULTS_DIR, f"ACE-III_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({
            "patient": _session_config.get("patient"),
            "assessor": _session_config.get("assessor"),
            "date": datetime.now().isoformat(),
            "domain_scores": scores,
            "domain_caps": {d: ACE_DATA[d]["score_cap"] for d in ACE_DATA},
            "total_score": total,
            "interpretation": interpretation,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

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
