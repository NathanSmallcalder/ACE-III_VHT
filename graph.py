from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, HumanMessage
from rapidfuzz import fuzz
from PIL import Image
import json
import os
import re
import time
from LLM.dialogue import *
from marking.marking import *
from datetime import datetime
from data_loader import resolve_dynamic_answers, get_season_transition
from visual_tasks.visual import run_visual_task, run_click_task, is_click_point_question, task_modality

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
    question_log: list   # one record per finished question, for the end-of-test review JSON
    question_turn_start: int  # index into messages where the in-progress question's turns began
    previous_task_signature: tuple  # (domain, modality) of the most recently finished question, or None


_session_config = None
tts = None
_audio = None
_audio_fluency = None
_gui = None

with open("json/ACE-III.json", "r") as f:
    ACE_DATA = json.load(f)["Domains"]

SEASON_TRUE, SEASON_ADJACENT = get_season_transition(datetime.now())

def configure(session_config, tts_engine, audio, audio_fluency, gui):
    """Wire up the runtime dependencies the graph's nodes read from module
    globals, and resolve every question's DYNAMIC answers against
    session_config. Must be called once before graph.invoke(...)."""
    global _session_config, tts, _audio, _audio_fluency, _gui
    _session_config = session_config
    tts = tts_engine
    _audio = audio
    _audio_fluency = audio_fluency
    _gui = gui

    for _domain in ACE_DATA.values():
        for _question in _domain["questions"]:
            _original_answers = _question["answers"]
            if "DYNAMIC:season" in _original_answers:
                _question["season_sub_index"] = _original_answers.index("DYNAMIC:season")

            if "DYNAMIC:uk_prime_minister" in _original_answers and session_config.get("previous_uk_pm"):
                _question["outgoing_leader"] = session_config["previous_uk_pm"]
            if "DYNAMIC:us_president" in _original_answers and session_config.get("previous_us_president"):
                _question["outgoing_leader"] = session_config["previous_us_president"]
            _question["answers"] = resolve_dynamic_answers(_original_answers, session_config)

"""Apply question/domain caps, add to the running score, and clear all
reprompt/progress state. Every handler and the default scoring path end
here"""
def _finalize_score(state, question, domain, q_index, score, new_sub) -> dict:
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
    """Trigger a handler-specific reprompt turn (no score yet).
       Called in Attention Season question and Memory personal names questions.
       for specific  "what was their last name" / "who was the previous one". , "could it be another season instances"
    """
    return {"needs_repeat": True, "repeat_count": 0, "reprompt_kind": kind}


def _handle_person_name(state, question, response, domain, q_index, sub_index):
    """
    reprompt_kind carries state across turns: "leader" = 3rd attempt (checks outgoing
    politician's name/surname), "name" = 2nd attempt (re-scores after asking for surname).
    First attempt below: score normally, else reprompt for surname or outgoing leader as fallback.
    """
    outgoing = question.get("outgoing_leader")
    if state.get("reprompt_kind") == "leader":
        score = score_person_name(response, [outgoing, outgoing.split()[-1]]) if outgoing else 0
        return _finalize_score(state, question, domain, q_index, score, 0)
    if state.get("reprompt_kind") == "name":
        return _finalize_score(state, question, domain, q_index, score_question(response, question), 0)
    score = score_question(response, question)
    if score:
        return _finalize_score(state, question, domain, q_index, score, 0)

    # ask for the surname before giving up on the question.
    if len(clean_response(response).split()) == 1:
        return _reprompt("name")

    # Wrong (and not just incomplete) — if there's been a recent change of
    # leader, probe for the outgoing politician's name as an alternate credit.
    if outgoing:
        return _reprompt("leader")

    return _finalize_score(state, question, domain, q_index, 0, 0)


def _reprompt_text_person_name(state, question, text):
    """
    Picks the spoken follow-up text based on which reprompt_kind was just set
    by _handle_person_name: "name" asks for the surname, "leader" asks for
    the outgoing politician. Returns None on a normal (non-reprompt) turn.
    """
    if state.get("reprompt_kind") == "name":
        return "And what was their surname?"
    if state.get("reprompt_kind") == "leader":
        return "Who was the previous one, before them?"
    return None


def _handle_registration(state, question, response, domain, q_index, sub_index):
    """
    Repeats the word list up to max_trials times so the patient can learn it,
    but only trial 1's score counts (question_score carries it across turns).
    Stops early once all_correct; final call scores with first_attempt_score, ignoring later trials.
    """
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

    return _finalize_score({**state, "question_score": 0}, question, domain, q_index, first_attempt_score, 0)


def _reprompt_text_registration(state, question, text):
    """
    Repeat the same question for registration question.
    """
    return text if state.get("reprompt_kind") == "registration" else None


def _handle_season(state, question, response, domain, q_index, sub_index):
    """
    Adjacent-season check: e.g. "spring" said during a summer/spring boundary
    window — reprompt once ("could it be another season?") instead of scoring
    wrong outright. sub_index scoring only fires for this one sub-answer slot.
    """
    expected_answers = question["answers"]
    if (
        state.get("reprompt_kind") != "season" and SEASON_ADJACENT
        and score_fuzzy(response, [SEASON_ADJACENT]) and not score_fuzzy(response, [SEASON_TRUE])
    ):
        return _reprompt("season")

    score = score_question(response, question, sub_index=sub_index) if sub_index < len(expected_answers) else 0
    return _finalize_score(state, question, domain, q_index, score, sub_index + 1)

def _reprompt_text_season(state, question, text):
    """
    Reprompt for season slightly off by patient.
    """
    return "Could it be another season?" if state.get("reprompt_kind") == "season" else None

def _handle_sub_score_bands(state, question, response, domain, q_index, sub_index):
    """
    Called for Language's word repetition question (caterpillar, eccentricity,
    unintelligible, statistician) 
    """
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

def _handle_name_address_trials(state, question, response, domain, q_index, sub_index):
    """
    Repeats the name/address up to 3 times for patient learning only the
    final trial's score is kept, earlier attempts are discarded, not accumulated.
    """
    total_trials = question.get("trials", 3)
    trial = state.get("turn_progress", 0) + 1
    attempt_score = score_question(response, question)
    print(f"Name & address trial {trial}/{total_trials}: {attempt_score}/{question['score_cap']} this trial")

    if trial < total_trials:
        return {"needs_repeat": True, "repeat_count": 0, "turn_progress": trial, "reprompt_kind": "trial"}

    return _finalize_score(state, question, domain, q_index, attempt_score, 0)

def _reprompt_text_name_address(state, question, text):
    """
    Re-speaks just the name/address stimulus on trial 2 and 3 (same words
    each time, by design — not a paraphrase), skipping the trial-1-only
    preamble ("We will do this three times..."). Returns None on the
    final/scoring turn.
    """
    if state.get("reprompt_kind") != "trial":
        return None
    quotes = re.findall(r"Speak:\s*'(.*?)'", question.get("instructions", ""))
    stimulus = quotes[-1] if quotes else text
    return f"Let's do that again. {stimulus}"

def _recognition_recalled(question, sub_index, state):
    """True if this recognition element's tokens were already fully credited
    in the linked delayed-recall question, so it doesn't need to be re-asked."""
    element_indices = question.get("element_recall_indices")
    if not element_indices or sub_index >= len(element_indices):
        return False
    recalled = state.get("recall_matches", {}).get(question.get("recall_key"), [])
    return all(idx < len(recalled) and recalled[idx] for idx in element_indices[sub_index])


def _handle_recognition(state, question, response, domain, q_index, sub_index):
    """
    Auto-award 1pt if this element's were already credited in delayed
    recall (skip re-asking, see _recognition_recalled); otherwise score the
    multiple-choice response normally against this sub-answer's expected value.  
    """
    new_sub = sub_index + 1
    if _recognition_recalled(question, sub_index, state):
        return _finalize_score(state, question, domain, q_index, 1, new_sub)
    score = score_question(response, question, sub_index=sub_index)
    return _finalize_score(state, question, domain, q_index, score, new_sub)


"""
Handlers for unique questions that dont fit generic structure. 
like non linear scoring for repetition, names, recall address/name trial.
"""
_HANDLERS = {
    "person_name": (_handle_person_name, _reprompt_text_person_name),
    "registration": (_handle_registration, _reprompt_text_registration),
    "season": (_handle_season, _reprompt_text_season),
    "word_rep": (_handle_sub_score_bands, None),
    "name_address": (_handle_name_address_trials, _reprompt_text_name_address),
    "recognition": (_handle_recognition, None),
}


def _match_kind(question, sub_index):
    """
    Reads JSON fields for specific question HANDLERS
    """
    # Memory Q3-Q6: current/former UK PM, current US president, JFK
    if question.get("match_type") == "person_name":
        return "person_name"
    # Memory Q1: 3-word learning trial (lemon, key, ball)
    if question.get("score_first_attempt_only"):
        # a short/partial recall (e.g. "lemon") is a complete, valid attempt
        return "registration"
    # Attention Q1: the season sub-answer within day/date/month/year/season
    if question.get("season_sub_index") is not None and sub_index == question["season_sub_index"]:
        return "season"
    # Language Q3: word repetition (caterpillar, eccentricity, unintelligible, statistician)
    if question.get("sub_score_bands"):
        return "word_rep"
    # Memory Q2 (name/address learning, 3 trials) and Visuospatial's delayed
    # recall question (same name/address, asked again later)
    if question.get("score_final_trial_only"):
        # a partial recall mid-trial (e.g. only 4 of 7 elements) is a genuine,
        # complete turn — classify_turn tends to read it as "incomplete" since
        # it isn't the full phrase, which would wrongly trigger a generic
        # re-ask instead of just tallying this trial's score.
        return "name_address"
    # Visuospatial recognition question: name/number/street/town/county multiple-choice
    if question.get("element_recall_indices"):
        return "recognition"
    return None

def conversation_node(state: ACEState) -> dict:
    """
    Speaks the current question and captures the patient's response.
    Handles skip logic, visual/click task routing, and _HANDLERS-driven
    reprompt phrasing.
    """
    domain = state["current_domain"]
    q_index = state["question_index"]
    sub_index = state.get("sub_question_index", 0)
    question = ACE_DATA[domain]["questions"][q_index]

    if _match_kind(question, sub_index) == "recognition" and _recognition_recalled(question, sub_index, state):
        return {"messages": [AIMessage(content=""), HumanMessage(content="")]}

    if question.get("match_type") == "visual":
        return run_visual_task(state, question, tts, _audio, _session_config, _gui)

    if is_click_point_question(question):
        total_questions = len(ACE_DATA[domain]["questions"])
        next_question = ACE_DATA[domain]["questions"][q_index + 1] if q_index + 1 < total_questions else None
        return run_click_task(state, question, tts, _session_config, next_question, _gui)

    # Show once, on the first presentation of the question — not on every
    # repeat/reprompt turn, which would otherwise reopen a new viewer window.
    if question.get("image") and not state.get("needs_repeat"):
        _gui.show_stimulus_image(question["image"])

    prompts = get_sub_prompts(question)
    if prompts and sub_index < len(prompts):
        text = prompts[sub_index]
    else:
        spoken = parse_spoken_prompts(question)
        text = spoken[0] if spoken else question["question_text"]

    kind = _match_kind(question, sub_index)
    reprompt_fn = _HANDLERS[kind][1] if kind else None
    reprompt = reprompt_fn(state, question, text) if reprompt_fn else None

    if reprompt:
        spoken_text = reprompt
    elif state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        wrapper = resolve_wrapper(
            state, _session_config["patient"]["name"], domain, task_modality(question), sub_index,
        )
        spoken_text = f"{wrapper} {text}".strip() if wrapper else text

    print("Assessor:", spoken_text)
    
    _gui.add_message("assessor", spoken_text)
    tts.speak(spoken_text)

    question_key = (domain, q_index, sub_index)
    for _ in range(5):
        user_input = (
            _audio_fluency.capture_response(on_tick=_gui.pump, question_key=question_key) if domain == "Fluency"
            else _audio.capture_response(on_tick=_gui.pump, question_key=question_key)
        )
        if not user_input:
            print("[no response detected]")
            continue
        break
    else:
        user_input = ""

    if not user_input:
        return {"needs_repeat": True}

    print("Patient:", user_input)
    _gui.add_message("patient", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}


def scoring_node(state: ACEState) -> dict:
    """
    Scores the patient's last response: generic questions run classify_turn
    then score_question directly.
    """
    domain = state['current_domain']
    q_index = state['question_index']
    sub_index = state.get("sub_question_index", 0)
    question = ACE_DATA[domain]['questions'][q_index]
    score_domain = question.get("score_domain", domain)

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

    uses_score_fuzzy = kind in ("season", "recognition") or (kind is None and (is_multi or match_type == "fuzzy"))
    if uses_score_fuzzy:
        last_message = extract_final_answer(last_message, asked_text)

    if kind:
        return _HANDLERS[kind][0](state, question, last_message, score_domain, q_index, sub_index)

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

    result = _finalize_score(state, question, score_domain, q_index, score, new_sub)
    recall_key = question.get("recall_key")
    print(f"Scoring node: domain={domain}, question={q_index + 1}, sub={sub_index + 1}, "
          f"score={score}, new_sub={new_sub}, recall_key={recall_key}, is_multi={is_multi}")
    if recall_key and not is_multi:
        # Stash per-element results so a later recognition question can skip
        # elements already credited here.
        matches = score_mixed_list_detailed(last_message, expected_answers)
        result["recall_matches"] = {**state.get("recall_matches", {}), recall_key: matches}
    return result

def _question_record(state: ACEState) -> dict:
    """One review-log entry for the question just finished: every patient
    response captured since question_turn_start, plus its final score."""
    domain = state["current_domain"]
    q_index = state["question_index"]
    question = ACE_DATA[domain]["questions"][q_index]
    start = state.get("question_turn_start", 0)
    responses = [m.content for m in state["messages"][start:] if isinstance(m, HumanMessage)]
    return {
        "domain": domain,
        "question_index": q_index,
        "question_text": question.get("question_text"),
        "match_type": question.get("match_type", "fuzzy_list"),
        "response": responses,
        "score": state.get("question_score", 0),
        "score_cap": question["score_cap"],
    }

def advance_node(state: ACEState) -> dict:
    """
    Logs the finished question, then moves to the next question in the
    domain (resetting question_score); if the domain's questions are
    exhausted, pops the next domain off domain_queue.
    """
    domain = state["current_domain"]
    q_index = state["question_index"]
    question = ACE_DATA[domain]["questions"][q_index]
    total_questions = len(ACE_DATA[domain]["questions"])
    question_log = state.get("question_log", []) + [_question_record(state)]
    question_turn_start = len(state["messages"])
    previous_task_signature = (domain, task_modality(question))

    if domain == "Memory" and q_index == 1:
        # I accidently left memory in the json so now i have to detour to fluency before going back to memory for the retrograde questions.
        result = {
            "current_domain": "Fluency", "question_index": 0, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Memory" and q_index == 5:
        # Delayed recall/recognition (Memory Q6-7) need real interference
        # before them, so skip ahead to the next domain now instead of
        # asking them right after the retrograde questions; we detour back
        # once Visuospatial (the last domain) finishes.
        q = state["domain_queue"].copy()
        next_domain = q.pop(0)
        result = {
            "current_domain": next_domain, "question_index": 0, "sub_question_index": 0, "question_score": 0,
            "domain_queue": q,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Visuospatial" and q_index + 1 >= total_questions:
        # End of that detour: back to Memory for delayed recall/recognition.
        result = {
            "current_domain": "Memory", "question_index": 6, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif domain == "Fluency" and q_index + 1 >= total_questions:
        # End of the detour:
        result = {
            "current_domain": "Memory", "question_index": 2, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    elif q_index + 1 < total_questions:
        result = {
            "question_index": q_index + 1, "sub_question_index": 0, "question_score": 0,
            "question_log": question_log, "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }
    else:
        q = state["domain_queue"].copy()
        next_domain = q.pop(0)
        result = {
            "current_domain": next_domain,
            "question_index": 0,
            "sub_question_index": 0,
            "question_score": 0,
            "domain_queue": q,
            "question_log": question_log,
            "question_turn_start": question_turn_start,
            "previous_task_signature": previous_task_signature,
        }

    # Checkpoint reflects where the assessment will resume from (the question
    # about to be asked next), not the one that just finished.
    save_progress({**state, **result})
    return result

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _interpret_ace_total(total: int) -> str:
    # Cutoffs and sensitivity/specificity from the ACE-III administration guide.
    if total >= 88:
        return "At or above 88 — not indicative of dementia by either ACE-III cutoff."
    if total >= 82:
        return "Below 88 but at/above 82 — flagged by the more sensitive (88) cutoff only."
    return "Below both 88 and 82 — flagged by both ACE-III cutoffs."


def report_node(state: ACEState) -> dict:
    """
    Finishes the ACE-III test reports the scores. Generates a JSON file and ends the session
    """
    scores = state["scores"]
    total = sum(scores.values())
    interpretation = _interpret_ace_total(total)
    # advance_node handles every question but the very last one (router sends
    # scoring straight to report for it), so record it here before dumping.
    question_log = state.get("question_log", []) + [_question_record(state)]

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
            "questions": question_log,
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # A finished assessment has no in-progress state left to resume from —
    # remove the checkpoint so it can't be mistaken for an unfinished session.
    progress_path = os.path.join(PROGRESS_DIR, f"progress_{safe_name}.json")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    _gui.add_message("assessor", "Thank you — that concludes the assessment.")
    time.sleep(3)
    _gui.close()

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
    elif domain == "Visuospatial":
        # Always detours back to Memory for delayed recall/recognition, even
        # though domain_queue is already empty by this point.
        return "next_domain"
    elif state['domain_queue']:
        return "next_domain"
    else:
        return "report"

PROGRESS_DIR = os.path.join(RESULTS_DIR, "progress")


def save_progress(state: ACEState) -> str:
    """Writes the full in-progress state to disk so an interrupted session
    can be resumed later. Overwrites the same file each call — this is a
    checkpoint of the latest position, not a history of every save."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    patient_name = _session_config.get("patient", {}).get("name", "unknown")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(patient_name))
    out_path = os.path.join(PROGRESS_DIR, f"progress_{safe_name}.json")

    with open(out_path, "w") as f:
        json.dump({
            "current_domain": state["current_domain"],
            "question_index": state["question_index"],
            "sub_question_index": state.get("sub_question_index", 0),
            "question_score": state.get("question_score", 0),
            "scores": state["scores"],
            "domain_queue": state["domain_queue"],
            "complete": state.get("complete", False),
            "needs_repeat": state.get("needs_repeat", False),
            "repeat_count": state.get("repeat_count", 0),
            "reprompt_kind": state.get("reprompt_kind"),
            "turn_progress": state.get("turn_progress", 0),
            "recall_matches": state.get("recall_matches", {}),
            "question_log": state.get("question_log", []),
            "question_turn_start": state.get("question_turn_start", 0),
            "previous_task_signature": state.get("previous_task_signature"),
            "messages": [{"role": m.type, "content": m.content} for m in state["messages"]],
        }, f, indent=2)

    return out_path

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
