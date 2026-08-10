import os
import threading
import time
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from camera.capture import capture_drawing, record_video
from LLM.dialogue import is_finished_drawing

PROGRESS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "progress")

DRAW_CHECKIN_PROMPTS = ["Are you ready to show me?", "Tell me when you're done."]
DRAW_SHOW_ME_PROMPT = "Okay, now show me."


class SessionWindow:
    """
    on construction the window shows a Start screen only; call
    wait_for_start() to block until the assessor clicks Start (fresh
    session) or Load Session (picks a saved progress JSON, exposed
    afterwards as self.load_path), then show_session_layout() once
    hardware/model init is done"""

    def __init__(self, title: str = "ACE-III Assessment"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1100x700")

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.bind("<Escape>", lambda _e: self.close())

        self._started = False
        self.load_path = None  # set to a progress JSON path if Load Session was used
        self._on_close = None
        self._click_window = {"image_path": None, "canvas": None,
                               "cell_w": None, "cell_h": None, "highlight": None}
        self._build_start_screen()
        self.pump()

    def set_on_close(self, callback) -> None:
        self._on_close = callback

    def _build_start_screen(self) -> None:
        self._start_frame = tk.Frame(self.root, bg="black")
        self._start_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self._start_frame, text="ACE-III Assessment", font=("Segoe UI", 28, "bold"),
            bg="black", fg="white",
        ).pack(expand=True, pady=(0, 10))

        self._status_label = tk.Label(
            self._start_frame, text="", font=("Segoe UI", 12), bg="black", fg="#aaaaaa"
        )
        self._status_label.pack(pady=(0, 20))

        button_row = tk.Frame(self._start_frame, bg="black")
        button_row.pack(pady=(0, 60))

        self._start_button = tk.Button(
            button_row, text="Start", font=("Segoe UI", 16), width=12,
            command=self._on_start_clicked,
        )
        self._start_button.pack(side=tk.LEFT, padx=10)

        self._load_button = tk.Button(
            button_row, text="Load Session", font=("Segoe UI", 16), width=12,
            command=self._on_load_clicked,
        )
        self._load_button.pack(side=tk.LEFT, padx=10)

    def _on_start_clicked(self) -> None:
        self._started = True
        self._start_button.configure(state="disabled", text="Loading...")
        self._load_button.configure(state="disabled")

    def _on_load_clicked(self) -> None:
        initialdir = PROGRESS_DIR if os.path.isdir(PROGRESS_DIR) else os.path.dirname(PROGRESS_DIR)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a saved session to resume",
            initialdir=initialdir,
            filetypes=[("Session progress JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return  # user cancelled — stay on the start screen
        self.load_path = path
        self._started = True
        self._start_button.configure(state="disabled")
        self._load_button.configure(state="disabled", text="Loading...")

    def wait_for_start(self) -> None:
        """Block the caller (main thread) until Start or Load Session is
        used, servicing the Tk event queue in the meantime so the window
        stays responsive. Check self.load_path afterwards to see which."""
        while not self._started:
            self.pump()
            time.sleep(0.03)

    def set_loading_status(self, text: str) -> None:
        """Update the status line shown under the title while init runs
        after Start has been clicked."""
        self._status_label.configure(text=text)
        self.pump()

    def show_session_layout(self) -> None:
        self._start_frame.destroy()

        self._stage_container = tk.Frame(self.root, bg="black")
        self._stage_container.pack(fill=tk.BOTH, expand=True)
        self._stage_frame = tk.Frame(self._stage_container, bg="black")
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        self._stage_image_ref = None

        self._question_label = tk.Label(
            self.root, text="", font=("Segoe UI", 20), wraplength=1000,
            bg="black", fg="white", justify="center",
        )
        self._image_showing = False
        self._position_question_label()

        self.pump()

    def _position_question_label(self) -> None:
        if self._image_showing:
            self._question_label.place(relx=0.5, rely=0.05, anchor="n")
        else:
            self._question_label.place(relx=0.5, rely=0.5, anchor="center")

    def add_message(self, role: str, text: str) -> None:
        if role != "assessor" or not text:
            return
        self._question_label.configure(text=text)
        self._position_question_label()
        self.pump()

    def get_stage_frame(self) -> tk.Frame:
        """Clear whatever is on stage and return a fresh, empty Frame for
        the caller to build task-specific widgets into (canvas, labels)."""
        self._stage_frame.destroy()
        self._stage_image_ref = None
        self._stage_frame = tk.Frame(self._stage_container, bg="black")
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        self._image_showing = False
        return self._stage_frame

    def stage_size(self) -> tuple[int, int]:
        self.root.update_idletasks()
        w = self._stage_container.winfo_width()
        h = self._stage_container.winfo_height()
        return (w if w > 50 else 700, h if h > 50 else 650)

    def show_stimulus_image(self, path: str) -> None:
        frame = self.get_stage_frame()
        self._image_showing = True
        self._position_question_label()
        max_w, max_h = self.stage_size()
        pil_img = Image.open(path)
        scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize(
                (int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS
            )
        tk_img = ImageTk.PhotoImage(pil_img)
        label = tk.Label(frame, image=tk_img, bg="black")
        label.pack(expand=True)
        self._stage_image_ref = tk_img
        self.pump()

    def pump(self) -> None:
        """Service the Tk event queue without blocking. Main-thread only —
        never call from a background thread (Tkinter is not thread-safe)."""
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def close(self) -> None:
        if self._on_close:
            self._on_close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def launch_camera_capture(self, output_path: str, audio, tts,
                               duration: int = 180, reference_image_path: str | None = None) -> None:
        """Give the patient `duration` seconds to draw on a real sheet of paper in
        front of the webcam, then photograph and rectify it via camera/capture.py.
        Ends early if the patient says they're finished (voice), same as the
        timeout path otherwise.

        Listening runs on a background thread because a single audio capture call
        can block for up to a minute (see voice/capture.py) and the on-screen
        timer needs to keep counting down regardless. Everything else happens on
        the main thread in a plain poll loop, serviced via self.pump()."""
        frame = self.get_stage_frame()

        # Copy-from-reference tasks (Wire Cube, Infinity Diagram) need the
        # reference visible the whole time the patient is drawing
        panel = tk.Frame(frame, bg="black")
        if reference_image_path:
            stage_w, stage_h = self.stage_size()
            ref_max_w = max(stage_w - 420, 150)  # leave room for the status panel + padding
            pil_ref = Image.open(reference_image_path)
            scale = min(ref_max_w / pil_ref.width, stage_h / pil_ref.height, 1.0)
            if scale < 1.0:
                pil_ref = pil_ref.resize(
                    (int(pil_ref.width * scale), int(pil_ref.height * scale)), Image.LANCZOS
                )
            ref_tk_img = ImageTk.PhotoImage(pil_ref)
            ref_label = tk.Label(frame, image=ref_tk_img, bg="black")
            ref_label.image = ref_tk_img
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

        voice_finished = {"v": False}
        stop_listening = threading.Event()

        def listen_for_finish():
            while not stop_listening.is_set():
                text = audio.capture_response()
                if stop_listening.is_set():
                    break
                if text and is_finished_drawing(text):
                    voice_finished["v"] = True
                    break

        listener = threading.Thread(target=listen_for_finish, daemon=True)
        listener.start()

        start_time = time.time()
        checkin_count = 0
        next_checkin_at = 40
        try:
            while True:
                remaining = duration - int(time.time() - start_time)
                if remaining <= 0 or voice_finished["v"]:
                    break
                mins, secs = divmod(remaining, 60)
                timer_label.config(text=f"{mins}:{secs:02d}", fg="red" if remaining <= 10 else "black")

                elapsed = duration - remaining
                if elapsed >= next_checkin_at and remaining > 15:  # dont check in 15 seconds before the end
                    tts.speak(DRAW_CHECKIN_PROMPTS[checkin_count % len(DRAW_CHECKIN_PROMPTS)])
                    checkin_count += 1
                    next_checkin_at += 40

                self.pump()
                time.sleep(0.2)

            stop_listening.set()
            timer_label.config(text="Finished!" if voice_finished["v"] else "Time's up!", fg="red")
            status_label.config(text="Please hold your drawing steady facing the camera...")
            self.pump()
        except tk.TclError:
            stop_listening.set()

        tts.speak(DRAW_SHOW_ME_PROMPT)
        capture_drawing(output_path)
        listener.join(timeout=1)

    def launch_video_capture(self, output_path: str, prompts: list[str], tts) -> None:
        """Record a single clip spanning `prompts`, spoken one at a time with a
        pause after each for the patient to act, so the VLM scorer can judge the
        pencil/paper actions from the footage rather than a spoken description.
        Recording runs on a background thread for the whole sequence."""
        frame = self.get_stage_frame()

        status_label = tk.Label(frame, text="Recording...", font=("Arial", 14),
                                fg="white", bg="black", wraplength=360, justify="center")
        status_label.pack(expand=True, pady=20)

        stop_event = threading.Event()
        recorder = threading.Thread(target=record_video, args=(output_path, stop_event), daemon=True)
        recorder.start()

        def wait(seconds):
            end = time.time() + seconds
            while time.time() < end:
                self.pump()
                time.sleep(0.05)

        for prompt in prompts:
            status_label.config(text=prompt)
            tts.speak(prompt)
            wait(6)

        wait(3)
        stop_event.set()
        recorder.join(timeout=5)

    def _close_click_window(self) -> None:
        self._click_window.update(image_path=None, canvas=None,
                                   cell_w=None, cell_h=None, highlight=None)

    def _open_click_window(self, image_path: str, grid_cols: int, grid_rows: int) -> None:
        frame = self.get_stage_frame()
        pil_img = Image.open(image_path)
        max_w, max_h = self.stage_size()
        scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas = tk.Canvas(frame, width=pil_img.width, height=pil_img.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img
        cell_w = pil_img.width / grid_cols
        cell_h = pil_img.height / grid_rows
        self._click_window.update(image_path=image_path, canvas=canvas,
                                   cell_w=cell_w, cell_h=cell_h, highlight=None)

    def launch_click_canvas(self, image_path: str, keep_open: bool, grid_cols: int, grid_rows: int,
                             initial_timeout: int = 30, settle_seconds: int = 3) -> int | None:
        """Show `image_path` as a grid_cols x grid_rows grid and let the patient
        click a cell. Self-corrections are allowed: each click restarts the
        settle timer, so only the last click before `settle_seconds` of no
        further clicks is finalized. Returns the 0-based grid index clicked
        (row-major), or None if nothing was clicked before `initial_timeout`.

        Reuses the previous stage content when it's already showing this same
        image (`keep_open` from the caller), instead of closing and reopening it
        for every question that points at the same picture."""
        if self._click_window["image_path"] != image_path:
            self._open_click_window(image_path, grid_cols, grid_rows)
        elif self._click_window["highlight"] is not None:
            self._click_window["canvas"].delete(self._click_window["highlight"])
            self._click_window["highlight"] = None

        root = self.root
        canvas = self._click_window["canvas"]
        cell_w = self._click_window["cell_w"]
        cell_h = self._click_window["cell_h"]

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
                self.get_stage_frame()
                self._close_click_window()
                root.quit()

        def on_click(e):
            col = min(max(int(e.x // cell_w), 0), grid_cols - 1)
            row = min(max(int(e.y // cell_h), 0), grid_rows - 1)
            selected["index"] = row * grid_cols + col

            if self._click_window["highlight"] is not None:
                canvas.delete(self._click_window["highlight"])
            x0, y0 = col * cell_w, row * cell_h
            self._click_window["highlight"] = canvas.create_rectangle(
                x0, y0, x0 + cell_w, y0 + cell_h, outline="red", width=4
            )

            if timers["settle"] is not None:
                root.after_cancel(timers["settle"])
            timers["settle"] = root.after(settle_seconds * 1000, finish)

        canvas.bind("<Button-1>", on_click)
        timers["timeout"] = root.after(initial_timeout * 1000, finish)
        root.mainloop()

        return result["index"]
