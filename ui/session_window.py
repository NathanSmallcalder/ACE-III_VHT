import tkinter as tk
from tkinter import scrolledtext

from PIL import Image, ImageTk


class SessionWindow:
    """Single persistent, patient-facing window for the whole session: a
    read-only chat transcript plus an image/interaction "stage". Draw and
    click tasks in visual_tasks/visual.py build their widgets into the stage
    Frame this hands out, instead of each spinning up its own tk.Tk() root."""

    def __init__(self, title: str = "ACE-III Assessment"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1100x700")

        # Patient-facing: the only window in the session must not be
        # closable by an accidental click — there is no "reopen" path once
        # it's gone. Escape is the assessor's deliberate escape hatch.
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.root.bind("<Escape>", lambda _e: self.close())

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)

        self._stage_container = tk.Frame(paned, bg="black")
        paned.add(self._stage_container, stretch="always", width=700)
        self._stage_frame = tk.Frame(self._stage_container, bg="black")
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        self._stage_image_ref = None  # keep PhotoImage alive against GC

        transcript_container = tk.Frame(paned)
        paned.add(transcript_container, stretch="never", width=380)
        self._transcript = scrolledtext.ScrolledText(
            transcript_container, state="disabled", wrap="word", font=("Segoe UI", 11)
        )
        self._transcript.pack(fill=tk.BOTH, expand=True)
        self._transcript.tag_config("assessor", foreground="#1a4fa0")
        self._transcript.tag_config("patient", foreground="#333333")

        self.pump()  # realize geometry so stage_size() is meaningful immediately

    def add_message(self, role: str, text: str) -> None:
        """role is 'assessor' or 'patient'. Only call this from the spots
        that already print("Assessor:"/"Patient:", ...) today — never pass
        score/correctness info here, the transcript is prompts/responses
        only (per the ACE-III guide's anti-anxiety administration rule)."""
        if not text:
            return
        label = "Assessor" if role == "assessor" else "Patient"
        self._transcript.configure(state="normal")
        self._transcript.insert(tk.END, f"{label}: {text}\n\n", role)
        self._transcript.configure(state="disabled")
        self._transcript.see(tk.END)
        self.pump()

    def get_stage_frame(self) -> tk.Frame:
        """Clear whatever is on stage and return a fresh, empty Frame for
        the caller to build task-specific widgets into (canvas, labels)."""
        self._stage_frame.destroy()
        self._stage_image_ref = None
        self._stage_frame = tk.Frame(self._stage_container, bg="black")
        self._stage_frame.pack(fill=tk.BOTH, expand=True)
        return self._stage_frame

    def stage_size(self) -> tuple[int, int]:
        self.root.update_idletasks()
        w = self._stage_container.winfo_width()
        h = self._stage_container.winfo_height()
        return (w if w > 50 else 700, h if h > 50 else 650)

    def show_stimulus_image(self, path: str) -> None:
        frame = self.get_stage_frame()
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

    def clear_stage(self) -> None:
        self.get_stage_frame()
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
        try:
            self.root.destroy()
        except tk.TclError:
            pass
