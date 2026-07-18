import os
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage

from camera.capture import capture_drawing, record_video
from LLM.dialogue import introduce, acknowledge, rephrase_question, classify_turn
from marking.marking import parse_spoken_prompts

DRAW_TASKS = {"Clock", "Infinity Diagram", "Wire Cube", "Writing"}
DRAW_DURATION = 120

# Follow-three-stage-commands: recorded (not spoken-response) so the VLM scorer
# can judge the pencil/paper actions directly, rather than relying on the
# patient describing what they did.
VIDEO_TASK_PREFIX = "Comprehension: Follow three-stage commands"
VIDEO_RESPONSE_PAUSE = 6   # seconds given to act on each spoken command
VIDEO_SETTLE_SECONDS = 3   # extra recording time after the last command

# Picture-comprehension grid (page3_img3.png): 3 columns x 4 rows, read
# top-left to bottom-right — same order as the Naming question's answers.
CLICK_POINT_PREFIX = "Comprehension: Which picture"
GRID_COLS, GRID_ROWS = 3, 4
GRID_ITEMS = [
    "spoon", "book", "kangaroo",
    "penguin", "anchor", "camel",
    "harp", "rhinoceros", "barrel",
    "crown", "crocodile", "accordion",
]


def _is_draw_task(question: dict) -> bool:
    text = question.get("question_text", "")
    return any(text.startswith(prefix + ":") for prefix in DRAW_TASKS)


def _is_video_task(question: dict) -> bool:
    return question.get("question_text", "").startswith(VIDEO_TASK_PREFIX)


def is_click_point_question(question: dict) -> bool:
    return question.get("question_text", "").startswith(CLICK_POINT_PREFIX)


def _launch_camera_capture(gui, output_path: str, audio, question_text: str,
                            duration: int = DRAW_DURATION, reference_image_path: str | None = None):
    """Give the patient `duration` seconds to draw on a real sheet of paper in
    front of the webcam, then photograph and rectify it via camera/capture.py.
    Ends early if the patient says they're finished (voice), same as the
    timeout path otherwise."""
    root = gui.root
    frame = gui.get_stage_frame()

    # Copy-from-reference tasks (Wire Cube, Infinity Diagram) need the
    # reference visible the whole time the patient is drawing — show it
    # alongside the status panel in the same frame, not as a separate
    # stage_frame that the panel would immediately replace.
    panel = tk.Frame(frame, bg="black")
    if reference_image_path:
        stage_w, stage_h = gui.stage_size()
        ref_max_w = max(stage_w - 420, 150)  # leave room for the status panel + padding
        pil_ref = Image.open(reference_image_path)
        scale = min(ref_max_w / pil_ref.width, stage_h / pil_ref.height, 1.0)
        if scale < 1.0:
            pil_ref = pil_ref.resize(
                (int(pil_ref.width * scale), int(pil_ref.height * scale)), Image.LANCZOS
            )
        ref_tk_img = ImageTk.PhotoImage(pil_ref)
        ref_label = tk.Label(frame, image=ref_tk_img, bg="black")
        ref_label.image = ref_tk_img  # keep a reference alive — Tk won't otherwise
        ref_label.pack(side=tk.LEFT, padx=10, pady=10)
        panel.pack(side=tk.LEFT, expand=True)
    else:
        panel.pack(expand=True)

    timer_label = tk.Label(panel, text=f"{duration // 60}:{duration % 60:02d}",
                           font=("Arial", 16), fg="black")
    timer_label.pack()

    status_label = tk.Label(panel, text="Please draw on the paper in front of you.",
                            font=("Arial", 14), fg="white", bg="black", wraplength=360, justify="center")
    status_label.pack(pady=20)

    remaining = {"s": duration}
    done = {"v": False}
    voice_finished = {"v": False}
    stop_listening = threading.Event()
    timer_ids = {"tick": None, "poll": None}

    def finish(label_text):
        if done["v"]:
            return
        done["v"] = True
        stop_listening.set()
        for tid in (timer_ids["tick"], timer_ids["poll"]):
            if tid is not None:
                try:
                    root.after_cancel(tid)
                except tk.TclError:
                    pass
        timer_label.config(text=label_text, fg="red")
        status_label.config(text="Please hold your drawing steady facing the camera...")
        root.update()

        def do_capture():
            capture_drawing(output_path, preview=False)
            root.after(0, root.quit)  # back onto the main thread, doesn't tear down the shared window

        threading.Thread(target=do_capture, daemon=True).start()

    def tick():
        if not timer_label.winfo_exists() or done["v"]:
            return
        if voice_finished["v"]:
            finish("Finished!")
            return
        s = remaining["s"]
        mins, secs = divmod(s, 60)
        timer_label.config(text=f"{mins}:{secs:02d}", fg="red" if s <= 10 else "black")
        if s > 0:
            remaining["s"] -= 1
            timer_ids["tick"] = root.after(1000, tick)
        else:
            finish("Time's up!")

    def poll_voice():
        if done["v"] or not timer_label.winfo_exists():
            return
        if voice_finished["v"]:
            finish("Finished!")
            return
        timer_ids["poll"] = root.after(200, poll_voice)

    def listen_for_finish():
        # Reuse the same LLM turn-classifier the rest of the assessment uses
        # ("answer" == the patient is done responding to the current prompt)
        # instead of hand-rolled keyword matching.
        while not stop_listening.is_set():
            text = audio.capture_response()
            if stop_listening.is_set():
                break
            if text and classify_turn(text, question_text) == "answer":
                voice_finished["v"] = True
                break

    listener = threading.Thread(target=listen_for_finish, daemon=True)
    listener.start()

    tick()
    poll_voice()
    root.mainloop()

    stop_listening.set()
    listener.join(timeout=1)


def _launch_video_capture(gui, output_path: str, prompts: list[str], tts):
    """Record a single clip spanning `prompts`, spoken one at a time with a
    pause after each for the patient to act, so the VLM scorer can judge the
    pencil/paper actions from the footage rather than a spoken description.
    Recording runs on a background thread for the whole sequence."""
    frame = gui.get_stage_frame()

    status_label = tk.Label(frame, text="Recording...", font=("Arial", 14),
                            fg="white", bg="black", wraplength=360, justify="center")
    status_label.pack(expand=True, pady=20)

    stop_event = threading.Event()
    recorder = threading.Thread(target=record_video, args=(output_path, stop_event), daemon=True)
    recorder.start()

    def wait(seconds):
        end = time.time() + seconds
        while time.time() < end:
            gui.pump()
            time.sleep(0.05)

    for prompt in prompts:
        status_label.config(text=prompt)
        tts.speak(prompt)
        wait(VIDEO_RESPONSE_PAUSE)

    wait(VIDEO_SETTLE_SECONDS)
    stop_event.set()
    recorder.join(timeout=5)


# Tracks the one click-grid stage content that may be reused across
# consecutive pointing questions sharing the same image, so it doesn't
# flicker closed/reopen between each of the four "Comprehension: Which
# picture..." questions on the same page.
_active_click_window = {"image_path": None, "canvas": None,
                         "cell_w": None, "cell_h": None, "highlight": None}


def _close_click_window():
    _active_click_window.update(image_path=None, canvas=None,
                                 cell_w=None, cell_h=None, highlight=None)


def _open_click_window(gui, image_path: str):
    frame = gui.get_stage_frame()

    pil_img = Image.open(image_path)
    max_w, max_h = gui.stage_size()
    scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas = tk.Canvas(frame, width=pil_img.width, height=pil_img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.image = tk_img  # keep a reference alive — Tk won't otherwise

    cell_w = pil_img.width / GRID_COLS
    cell_h = pil_img.height / GRID_ROWS
    _active_click_window.update(image_path=image_path, canvas=canvas,
                                 cell_w=cell_w, cell_h=cell_h, highlight=None)


def _launch_click_canvas(gui, image_path: str, keep_open: bool, initial_timeout: int = 30, settle_seconds: int = 3):
    """Show `image_path` as a GRID_COLS x GRID_ROWS grid and let the patient
    click a cell. Self-corrections are allowed: each click restarts the
    settle timer, so only the last click before `settle_seconds` of no
    further clicks is finalized. Returns the 0-based grid index clicked
    (row-major), or None if nothing was clicked before `initial_timeout`.

    Reuses the previous stage content when it's already showing this same
    image (`keep_open` from the caller), instead of closing and reopening it
    for every question that points at the same picture."""
    if _active_click_window["image_path"] != image_path:
        _open_click_window(gui, image_path)
    elif _active_click_window["highlight"] is not None:
        _active_click_window["canvas"].delete(_active_click_window["highlight"])
        _active_click_window["highlight"] = None

    root = gui.root
    canvas = _active_click_window["canvas"]
    cell_w = _active_click_window["cell_w"]
    cell_h = _active_click_window["cell_h"]

    selected = {"index": None}
    result = {"index": None}
    timers = {"settle": None, "timeout": None}
    done = {"v": False}

    def finish():
        if done["v"]:
            return
        done["v"] = True
        for tid in (timers["settle"], timers["timeout"]):
            if tid is not None:
                try:
                    root.after_cancel(tid)
                except tk.TclError:
                    pass
        result["index"] = selected["index"]
        if keep_open:
            root.quit()
        else:
            _close_click_window()
            root.quit()

    def on_click(e):
        col = min(max(int(e.x // cell_w), 0), GRID_COLS - 1)
        row = min(max(int(e.y // cell_h), 0), GRID_ROWS - 1)
        selected["index"] = row * GRID_COLS + col

        if _active_click_window["highlight"] is not None:
            canvas.delete(_active_click_window["highlight"])
        x0, y0 = col * cell_w, row * cell_h
        _active_click_window["highlight"] = canvas.create_rectangle(
            x0, y0, x0 + cell_w, y0 + cell_h, outline="red", width=4
        )

        if timers["settle"] is not None:
            root.after_cancel(timers["settle"])
        timers["settle"] = root.after(settle_seconds * 1000, finish)

    canvas.bind("<Button-1>", on_click)
    timers["timeout"] = root.after(initial_timeout * 1000, finish)
    root.mainloop()

    return result["index"]


def run_click_task(state, question: dict, tts, session_config: dict, next_question: dict | None, gui) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        if not state["messages"]:
            wrapper = introduce(session_config["patient"]["name"])
        else:
            last_patient = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                None
            )
            wrapper = acknowledge(last_patient) if last_patient else ""

        spoken_text = f"{wrapper} {text}".strip() if wrapper else text
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)
    tts.speak(spoken_text)

    image_path = question.get("image")
    keep_open = bool(
        next_question and is_click_point_question(next_question) and next_question.get("image") == image_path
    )
    clicked_index = _launch_click_canvas(gui, image_path, keep_open) if image_path else None

    if clicked_index is None:
        return {"needs_repeat": True}

    clicked_item = GRID_ITEMS[clicked_index]
    print("Patient (pointed to):", clicked_item)
    gui.add_message("patient", clicked_item)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=clicked_item)]}


def run_visual_task(state, question: dict, tts, audio, session_config: dict, gui) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    # Draw tasks with a reference image show it inside _launch_camera_capture,
    # alongside the status panel — showing it here first would just get
    # replaced the moment that panel builds its own stage frame.
    if question.get("image") and not _is_draw_task(question):
        gui.show_stimulus_image(question["image"])

    wrapper = ""
    if state.get("needs_repeat"):
        spoken_text = rephrase_question(text)
    else:
        if not state["messages"]:
            wrapper = introduce(session_config["patient"]["name"])
        else:
            last_patient = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                None
            )
            wrapper = acknowledge(last_patient) if last_patient else ""

        spoken_text = f"{wrapper} {text}".strip() if wrapper else text
    print("Assessor:", spoken_text)
    gui.add_message("assessor", spoken_text)

    if _is_draw_task(question):
        if wrapper:
            tts.speak(wrapper)
        for prompt in spoken:
            tts.speak(prompt)
        task_name = question["question_text"].split(":")[0].lower().replace(" ", "_")
        output_path = os.path.join(os.path.dirname(__file__), f"{task_name}.png")
        _launch_camera_capture(gui, output_path, audio, question["question_text"],
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
        _launch_video_capture(gui, output_path, scored_prompts, tts)
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
