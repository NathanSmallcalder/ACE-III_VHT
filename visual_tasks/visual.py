import os
import threading
import tkinter as tk
from PIL import Image, ImageGrab
from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage

from LLM.dialogue import introduce, acknowledge, rephrase_question, classify_turn
from marking.marking import parse_spoken_prompts

DRAW_TASKS = {"Clock", "Infinity Diagram", "Wire Cube"}
DRAW_DURATION = 120


def _is_draw_task(question: dict) -> bool:
    text = question.get("question_text", "")
    return any(text.startswith(prefix + ":") for prefix in DRAW_TASKS)


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
