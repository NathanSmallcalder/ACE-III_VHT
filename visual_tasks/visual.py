import os
import threading
import tkinter as tk
from PIL import Image, ImageGrab, ImageTk
from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage

from LLM.dialogue import introduce, acknowledge, rephrase_question, classify_turn
from marking.marking import parse_spoken_prompts

DRAW_TASKS = {"Clock", "Infinity Diagram", "Wire Cube", "Writing"}
DRAW_DURATION = 120

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


def is_click_point_question(question: dict) -> bool:
    return question.get("question_text", "").startswith(CLICK_POINT_PREFIX)


def _launch_draw_canvas(output_path: str, audio, question_text: str, duration: int = DRAW_DURATION):
    root = tk.Tk()
    root.title("Draw")

    timer_label = tk.Label(root, text=f"{duration // 60}:{duration % 60:02d}",
                           font=("Arial", 16), fg="black")
    timer_label.pack()

    canvas = tk.Canvas(root, width=500, height=500, bg="white")
    canvas.pack()

    last = {"x": None, "y": None}
    remaining = {"s": duration}
    running = {"v": False}
    done = {"v": False}
    voice_finished = {"v": False}
    stop_listening = threading.Event()

    def start(e):
        last["x"], last["y"] = e.x, e.y
        if not running["v"]:
            running["v"] = True
            tick()

    def draw(e):
        canvas.create_line(last["x"], last["y"], e.x, e.y,
                           width=3, fill="navy", capstyle=tk.ROUND)
        last["x"], last["y"] = e.x, e.y

    def finish(label_text):
        if done["v"]:
            return
        done["v"] = True
        stop_listening.set()
        timer_label.config(text=label_text, fg="red")
        canvas.unbind("<Button-1>")
        canvas.unbind("<B1-Motion>")
        root.update()
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        ImageGrab.grab(bbox=(x, y, x + canvas.winfo_width(), y + canvas.winfo_height())).save(output_path)
        root.destroy()

    def tick():
        if voice_finished["v"]:
            finish("Finished!")
            return
        s = remaining["s"]
        mins, secs = divmod(s, 60)
        timer_label.config(text=f"{mins}:{secs:02d}", fg="red" if s <= 10 else "black")
        if s > 0:
            remaining["s"] -= 1
            root.after(1000, tick)
        else:
            finish("Time's up!")

    def poll_voice():
        if done["v"]:
            return
        if voice_finished["v"]:
            finish("Finished!")
            return
        root.after(200, poll_voice)

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

    canvas.bind("<Button-1>", start)
    canvas.bind("<B1-Motion>", draw)
    root.protocol("WM_DELETE_WINDOW", lambda: finish("Closed"))
    poll_voice()
    root.mainloop()

    stop_listening.set()
    listener.join(timeout=1)


# Tracks the one click-grid window that may be reused across consecutive
# pointing questions sharing the same image, so it doesn't flicker
# closed/reopen between each of the four "Comprehension: Which picture..."
# questions on the same page.
_active_click_window = {"image_path": None, "root": None, "canvas": None,
                         "cell_w": None, "cell_h": None, "highlight": None}


def _close_click_window():
    root = _active_click_window["root"]
    if root is not None:
        try:
            root.destroy()
        except tk.TclError:
            pass
    _active_click_window.update(image_path=None, root=None, canvas=None,
                                 cell_w=None, cell_h=None, highlight=None)


def _open_click_window(image_path: str):
    root = tk.Tk()
    root.title("Point")

    pil_img = Image.open(image_path)
    max_w = int(root.winfo_screenwidth() * 0.8)
    max_h = int(root.winfo_screenheight() * 0.8)
    scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas = tk.Canvas(root, width=pil_img.width, height=pil_img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.image = tk_img  # keep a reference alive — Tk won't otherwise

    cell_w = pil_img.width / GRID_COLS
    cell_h = pil_img.height / GRID_ROWS
    _active_click_window.update(image_path=image_path, root=root, canvas=canvas,
                                 cell_w=cell_w, cell_h=cell_h, highlight=None)


def _launch_click_canvas(image_path: str, keep_open: bool, initial_timeout: int = 30, settle_seconds: int = 3):
    """Show `image_path` as a GRID_COLS x GRID_ROWS grid and let the patient
    click a cell. Self-corrections are allowed: each click restarts the
    settle timer, so only the last click before `settle_seconds` of no
    further clicks is finalized. Returns the 0-based grid index clicked
    (row-major), or None if nothing was clicked before `initial_timeout`.

    Reuses the previous window when it's already showing this same image
    (`keep_open` from the caller), instead of closing and reopening it for
    every question that points at the same picture."""
    if _active_click_window["image_path"] != image_path:
        _close_click_window()
        _open_click_window(image_path)
    elif _active_click_window["highlight"] is not None:
        _active_click_window["canvas"].delete(_active_click_window["highlight"])
        _active_click_window["highlight"] = None

    root = _active_click_window["root"]
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

    def on_close():
        # Manual close always tears the window down for real, regardless
        # of keep_open — there's nothing left to reuse next time.
        done["v"] = True
        _close_click_window()

    canvas.bind("<Button-1>", on_click)
    timers["timeout"] = root.after(initial_timeout * 1000, finish)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    return result["index"]


def run_click_task(state, question: dict, tts, session_config: dict, next_question: dict | None = None) -> dict:
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
    tts.speak(spoken_text)

    image_path = question.get("image")
    keep_open = bool(
        next_question and is_click_point_question(next_question) and next_question.get("image") == image_path
    )
    clicked_index = _launch_click_canvas(image_path, keep_open) if image_path else None

    if clicked_index is None:
        return {"needs_repeat": True}

    clicked_item = GRID_ITEMS[clicked_index]
    print("Patient (pointed to):", clicked_item)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=clicked_item)]}


def run_visual_task(state, question: dict, tts, audio, session_config: dict) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    if question.get("image"):
        Image.open(question["image"]).show()

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

    if _is_draw_task(question):
        if wrapper:
            tts.speak(wrapper)
        for prompt in spoken:
            tts.speak(prompt)
        task_name = question["question_text"].split(":")[0].lower().replace(" ", "_")
        output_path = os.path.join(os.path.dirname(__file__), f"{task_name}.png")
        _launch_draw_canvas(output_path, audio, question["question_text"])
        return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=output_path)]}

    tts.speak(spoken_text)

    for _ in range(5):
        user_input = audio.capture_response()

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
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}
