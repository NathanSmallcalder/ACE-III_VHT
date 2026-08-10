import os
import time

from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage

from LLM.dialogue import resolve_wrapper, rephrase_question, check_in
from marking.marking import parse_spoken_prompts

draw_tasks = {"Clock", "Infinity Diagram", "Wire Cube", "Writing"}
draw_timer = 180

GRID_COLS, GRID_ROWS = 3, 4
GRID_ITEMS = [
    "spoon", "book", "kangaroo",
    "penguin", "anchor", "camel",
    "harp", "rhinoceros", "barrel",
    "crown", "crocodile", "accordion",
]

def _is_draw_task(question: dict) -> bool:
    text = question.get("question_text", "")
    return any(text.startswith(prefix + ":") for prefix in draw_tasks)

def _is_video_task(question: dict) -> bool:
    return question.get("question_text", "").startswith("Comprehension: Follow three-stage commands")

def is_click_point_question(question: dict) -> bool:
    return question.get("question_text", "").startswith("Comprehension: Which picture")

def task_type(question: dict) -> str:
    """Coarse task-type tag, independent of clinical domain, used to detect when
    the assessment moves to a different kind of interaction."""
    if _is_draw_task(question):
        return "draw"
    if _is_video_task(question):
        return "video"
    if is_click_point_question(question):
        return "click"
    return "spoken"


def run_click_task(state, question: dict, tts, session_config: dict, next_question: dict | None, gui) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    elif state.get("previous_task_signature") != (state["current_domain"], "click"):
        spoken_text = f"You need to click on the screen now. {text}"
    else:
        spoken_text = text
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)
    tts.speak(spoken_text)

    image_path = question.get("image")
    keep_open = bool(
        next_question and is_click_point_question(next_question) and next_question.get("image") == image_path
    )
    clicked_index = gui.launch_click_canvas(image_path, keep_open, GRID_COLS, GRID_ROWS) if image_path else None

    if clicked_index is None:
        return {"needs_repeat": True}

    clicked_item = GRID_ITEMS[clicked_index]
    print("Patient (pointed to):", clicked_item)
    gui.add_message("patient", clicked_item)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=clicked_item)]}


def run_visual_task(state, question: dict, tts, audio, session_config: dict, gui) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    # Draw tasks with a reference image show it inside gui.launch_camera_capture,
    # alongside the status panel — showing it here first would just get
    # replaced the moment that panel builds its own stage frame.
    if question.get("image") and not _is_draw_task(question):
        gui.show_stimulus_image(question["image"])

    wrapper = ""
    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        wrapper = resolve_wrapper(
            state, session_config["patient"]["name"], state["current_domain"],
            task_type(question), state.get("sub_question_index", 0),
        )
        spoken_text = f"{wrapper} {text}".strip() if wrapper else text
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)

    if _is_draw_task(question):
        if wrapper:
            tts.speak(wrapper)
        for prompt in spoken:
            tts.speak(prompt)
        time.sleep(0.45)

        tts.speak(check_in())
        task_name = question["question_text"].split(":")[0].lower().replace(" ", "_")
        output_path = os.path.join(os.path.dirname(__file__), f"{task_name}.png")
        gui.launch_camera_capture(output_path, audio, tts, duration=draw_timer,
                                   reference_image_path=question.get("image"))
        return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=output_path)]}

    if _is_video_task(question):
        if wrapper:
            tts.speak(wrapper)
        practice, scored_prompts = (spoken[0], spoken[1:]) if len(spoken) > 1 else (None, spoken)
        if practice:
            tts.speak(practice)
            audio.capture_response(on_tick=gui.pump)  # practice trial, not scored
        output_path = os.path.join(os.path.dirname(__file__), "..", "data", "videos", "pen_paper.mp4")
        gui.launch_video_capture(output_path, scored_prompts, tts)
        return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=output_path)]}

    tts.speak(spoken_text)

    for _ in range(5):
        user_input = audio.capture_response(on_tick=gui.pump)
        if not user_input:
            print("[no response detected]")
            continue
        if fuzz.partial_ratio(user_input.lower(), spoken_text.lower()) > 75:
            print("[echo detected, ignoring]")
            continue
        break
    else:
        return {"needs_repeat": True}

    print("Patient:", user_input)
    gui.add_message("patient", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}
