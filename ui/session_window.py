import os
import time
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

# Base path for saving/loading assessment progress
progress_dir = os.path.join(os.path.dirname(__file__), "..", "results", "progress")
bg_col = "white"
text_col = "black"


class SessionWindow:
    """
    Main GUI window for the assessment.
    
    Starts on a simple launcher screen (Start / Load Session) and transitions 
    into the active session layout once hardware/models finish loading.
    """

    def __init__(self, title: str = "ACE-III Assessment"):
        # Setup main Tk window
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1100x700")
        self.root.configure(bg=bg_col)

        # Session state tracking
        self._started = False
        self.load_path = None  # Holds progress JSON path if resuming a session
        self._on_close = None
        
        # State containers for interactive components
        self._click_window = {
            "image_path": None, 
            "canvas": None,
            "cell_w": None, 
            "cell_h": None, 
            "highlight": None
        }
        self._draw_panel = None
        self._video_status = None
        
        # Kick off the launcher UI
        self.build_start_screen()
        self.pump()

    def set_on_close(self, callback) -> None:
        """Register a custom cleanup callback for when the window closes."""
        self._on_close = callback

    def build_start_screen(self) -> None:
        """Construct the initial Start / Load screen."""
        self._start_frame = tk.Frame(self.root, bg=bg_col)
        self._start_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self._start_frame, text="ACE-III Assessment", font=("Segoe UI", 28, "bold"),
            bg=bg_col, fg=text_col,
        ).pack(expand=True, pady=(0, 10))

        # Status indicator (useful for showing loading messages)
        self._status_label = tk.Label(
            self._start_frame, text="", font=("Segoe UI", 12), bg=bg_col, fg="grey"
        )
        self._status_label.pack(pady=(0, 20))

        button_row = tk.Frame(self._start_frame, bg=bg_col)
        button_row.pack(pady=(0, 60))

        self._start_button = tk.Button(
            button_row, text="Start", font=("Segoe UI", 16), width=12,
            command=self.on_start_clicked,
        )
        self._start_button.pack(side=tk.LEFT, padx=10)

        self._load_button = tk.Button(
            button_row, text="Load Session", font=("Segoe UI", 16), width=12,
            command=self.on_load_clicked,
        )
        self._load_button.pack(side=tk.LEFT, padx=10)

    def on_start_clicked(self) -> None:
        """Handler for starting a fresh session."""
        self._started = True
        self._start_button.configure(state="disabled", text="Loading...")
        self._load_button.configure(state="disabled")

    def on_load_clicked(self) -> None:
        """Handler for browsing and loading an existing JSON session file."""
        initialdir = progress_dir if os.path.isdir(progress_dir) else os.path.dirname(progress_dir)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a saved session to resume",
            initialdir=initialdir,
            filetypes=[("Session progress JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return  # User canceled the file picker

        self.load_path = path
        self._started = True
        self._start_button.configure(state="disabled")
        self._load_button.configure(state="disabled", text="Loading...")

    def wait_for_start(self) -> None:
        """Pumps events and blocks until the assessor clicks Start or Load."""
        while not self._started:
            self.pump()
            time.sleep(0.03)

    def set_loading_status(self, text: str) -> None:
        """Update the loading text under the main title."""
        self._status_label.configure(text=text)
        self.pump()

    def show_session_layout(self) -> None:
        """Tear down the launcher frame and build the active test layout."""
        self._start_frame.destroy()

        self._stage_container = tk.Frame(self.root, bg=bg_col)
        self._stage_container.pack(fill=tk.BOTH, expand=True)
        self._stage_frame = tk.Frame(self._stage_container, bg=bg_col)
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        self._stage_image_ref = None

        self.question_label = tk.Label(
            self.root, text="", font=("Segoe UI", 20), wraplength=1000,
            bg=bg_col, fg=text_col, justify="center",
        )
        self._image_showing = False
        self.position_question_label()

        self.pump()

    def position_question_label(self) -> None:
        """Move question text to top if an image is on screen, otherwise center it."""
        if self._image_showing:
            self.question_label.place(relx=0.5, rely=0.05, anchor="n")
        else:
            self.question_label.place(relx=0.5, rely=0.5, anchor="center")

    def add_message(self, role: str, text: str) -> None:
        """Display prompt text meant for the assessor/patient."""
        if role != "assessor" or not text:
            return
        self.question_label.configure(text=text)
        self.position_question_label()
        self.pump()

    def get_stage_frame(self) -> tk.Frame:
        """Clear current stage components and return a fresh empty Frame."""
        self._stage_frame.destroy()
        self._stage_image_ref = None
        self._stage_frame = tk.Frame(self._stage_container, bg=bg_col)
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        self._image_showing = False
        return self._stage_frame

    def stage_size(self) -> tuple[int, int]:
        """Get dimensions of the current stage area (with sensible fallbacks)."""
        self.root.update_idletasks()
        w = self._stage_container.winfo_width()
        h = self._stage_container.winfo_height()
        return (w if w > 50 else 700, h if h > 50 else 650)

    def fit_image(self, path: str) -> Image.Image:
        """Scale an image down so it takes up max 75% of the stage area left free by
        the question text, which is banded across the top whenever an image is showing."""
        stage_w, stage_h = self.stage_size()
        reserved = self.question_label.winfo_reqheight() + 30 if self._image_showing else 0
        max_w, max_h = stage_w * 0.75, max(stage_h - reserved, 120) * 0.75
        pil_img = Image.open(path)
        scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
        if scale < 1.0:
            pil_img = pil_img.resize(
                (int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS
            )
        return pil_img

    def show_stimulus_image(self, path: str) -> None:
        """Center and display a stimulus image on stage."""
        frame = self.get_stage_frame()
        self._image_showing = True
        self.position_question_label()
        
        pil_img = self.fit_image(path)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        label = tk.Label(frame, image=tk_img, bg=bg_col)
        # Center in the area below the question text rather than the full stage
        stage_h = self.stage_size()[1]
        reserved = self.question_label.winfo_reqheight() + 30
        label.place(relx=0.5, rely=(reserved + (stage_h - reserved) / 2) / stage_h, anchor="center")

        self._stage_image_ref = tk_img
        self.pump()

    def pump(self) -> None:
        """Process pending Tk events without blocking execution."""
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def wait(self, seconds: float) -> None:
        """Non-blocking delay helper that keeps the UI responsive."""
        end = time.time() + seconds
        while time.time() < end:
            self.pump()
            time.sleep(0.05)

    def close(self) -> None:
        """Safely fire close callback and destroy Tk instance."""
        if self._on_close:
            self._on_close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def start_draw_task(self, duration: int, reference_image_path: str | None = None) -> None:
        """Set up stage for paper-drawing tasks with a live countdown timer."""
        frame = self.get_stage_frame()

        panel = tk.Frame(frame, bg=bg_col)
        if reference_image_path:
            # Show visual reference on left if task requires copying (e.g. cube)
            pil_ref = self.fit_image(reference_image_path)
            ref_tk_img = ImageTk.PhotoImage(pil_ref)
            ref_label = tk.Label(frame, image=ref_tk_img, bg=bg_col)
            ref_label.image = ref_tk_img
            ref_label.pack(side=tk.LEFT, padx=10, pady=10)
            panel.pack(side=tk.LEFT, expand=True)
        else:
            panel.pack(expand=True)

        mins, secs = divmod(max(duration, 0), 60)
        timer_label = tk.Label(panel, text=f"{mins}:{secs:02d}",
                               font=("Arial", 16), bg=bg_col, fg=text_col)
        timer_label.pack()
        
        status_label = tk.Label(panel, text="Please draw on the paper in front of you.",
                                font=("Arial", 14), bg=bg_col, fg=text_col,
                                wraplength=360, justify="center")
        status_label.pack(pady=20)

        self._draw_panel = {
            "deadline": time.time() + duration,
            "timer": timer_label,
            "status": status_label,
            "shown": None,
        }
        self.pump()

    def tick_draw_panel(self) -> None:
        """Update drawing task timer display (turns red when <= 10s)."""
        panel = self._draw_panel
        if panel is None:
            return
        
        remaining = max(int(panel["deadline"] - time.time()), 0)
        if remaining != panel["shown"]:
            panel["shown"] = remaining
            mins, secs = divmod(remaining, 60)
            try:
                panel["timer"].config(text=f"{mins}:{secs:02d}",
                                      fg="red" if remaining <= 10 else text_col)
            except tk.TclError:
                self._draw_panel = None
                return
        self.pump()

    def end_draw_panel(self, finished: bool) -> None:
        """Stop drawing timer and prompt patient to show paper to camera."""
        panel = self._draw_panel
        self._draw_panel = None
        if panel is None:
            return
        try:
            panel["timer"].config(text="")
            panel["status"].config(text="Please hold your drawing steady facing the camera...")
        except tk.TclError:
            return
        self.pump()

    def start_video_panel(self, text: str = "Recording...") -> None:
        """Display single status label for video-monitored prompts."""
        frame = self.get_stage_frame()
        self._video_status = tk.Label(frame, text=text, font=("Arial", 14),
                                      bg=bg_col, fg=text_col,
                                      wraplength=360, justify="center")
        self._video_status.pack(expand=True, pady=20)
        self.pump()

    def set_video_prompt(self, text: str) -> None:
        """Update prompt string on active video stage."""
        if self._video_status is None:
            return
        try:
            self._video_status.config(text=text)
        except tk.TclError:
            self._video_status = None
            return
        self.pump()

    def end_video_panel(self) -> None:
        """Clean up active video status panel."""
        self._video_status = None
        self.get_stage_frame()
        self.pump()

    def _close_click_window(self) -> None:
        """Reset internal click-selection state."""
        self._click_window.update(image_path=None, canvas=None,
                                   cell_w=None, cell_h=None, highlight=None)

    def _open_click_window(self, image_path: str, grid_cols: int, grid_rows: int) -> None:
        """Build a clickable Canvas mapped across an image in a virtual grid."""
        frame = self.get_stage_frame()
        self._image_showing = True
        self.position_question_label()
        
        pil_img = self.fit_image(image_path)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas = tk.Canvas(frame, width=pil_img.width, height=pil_img.height,
                            bg=bg_col, highlightthickness=0)
        canvas.place(relx=0.5, rely=0.6, anchor="center")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img
        
        cell_w = pil_img.width / grid_cols
        cell_h = pil_img.height / grid_rows
        
        self._click_window.update(image_path=image_path, canvas=canvas,
                                   cell_w=cell_w, cell_h=cell_h, highlight=None)

    def launch_click_canvas(self, image_path: str, keep_open: bool, grid_cols: int, grid_rows: int,
                             initial_timeout: int = 15, settle_seconds: int = 3) -> int | None:
        """
        Interactively captures a click inside a virtual grid on an image.
        
        Allows self-corrections: restarts a settlement timer on every click so
        only the final selection is returned after `settle_seconds` of inactivity.
        Returns grid cell index (0-based, row-major) or None on timeout.
        """
        # Re-use active stage if showing the same image, otherwise rebuild
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
            """Clean up active timers and complete selection."""
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
            """Handle click event, draw highlight rectangle, and reset settle timer."""
            col = min(max(int(e.x // cell_w), 0), grid_cols - 1)
            row = min(max(int(e.y // cell_h), 0), grid_rows - 1)
            selected["index"] = row * grid_cols + col

            # Redraw selection box around clicked grid cell
            if self._click_window["highlight"] is not None:
                canvas.delete(self._click_window["highlight"])
            x0, y0 = col * cell_w, row * cell_h
            self._click_window["highlight"] = canvas.create_rectangle(
                x0, y0, x0 + cell_w, y0 + cell_h, outline="red", width=4
            )

            # Reset delay timer on each click to allow corrections
            if timers["settle"] is not None:
                root.after_cancel(timers["settle"])
            timers["settle"] = root.after(settle_seconds * 1000, finish)

        canvas.bind("<Button-1>", on_click)
        timers["timeout"] = root.after(initial_timeout * 1000, finish)
        
        # Block until finish() calls root.quit()
        root.mainloop()

        return result["index"]