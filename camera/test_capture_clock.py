"""
camera/test_capture_clock.py
-----------------------------
Manual integration test wiring camera/capture.py into visual_tasks/clock_scorer.py:
capture a clock drawing off the webcam (or reuse an existing photo), run it through
the document-detect/deskew/enhance pipeline, then score it with the real VLM scorer.

Usage (run from the project root so `camera`/`visual_tasks` resolve as packages):
    python -m camera.test_capture_clock                    # webcam -> pipeline -> score
    python -m camera.test_capture_clock --no-preview        # capture without the live window
    python -m camera.test_capture_clock --single-shot       # grab one frame, crop to the green box
    python -m camera.test_capture_clock path\\to\\photo.jpg   # skip capture, score an existing image
"""
import os
import sys

from camera.capture import capture_drawing, capture_green_box
from visual_tasks.clock_scorer import score_clock_image

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results", "captures", "clock_test.png")


def main():
    args = sys.argv[1:]
    preview = "--no-preview" not in args
    single_shot = "--single-shot" in args
    args = [a for a in args if a not in ("--no-preview", "--single-shot")]

    if args:
        image_path = args[0]
        print(f"Using existing image: {image_path}")
    elif single_shot:
        output_path = os.path.abspath(DEFAULT_OUTPUT)
        print("Grabbing a single frame and cropping to the detected green box...")
        cropped = capture_green_box(output_path)
        if cropped is None:
            print("No document outline detected in that frame — try again with better framing/lighting.")
            return
        image_path = output_path
        print(f"Saved cropped capture to: {image_path}")
    else:
        output_path = os.path.abspath(DEFAULT_OUTPUT)
        print("Capturing clock drawing from webcam..." + ("" if preview else " (no preview)"))
        image_path = capture_drawing(output_path, preview=preview)
        print(f"Saved rectified capture to: {image_path}")

    print("Scoring with score_clock_image()...")
    result = score_clock_image(image_path)

    print("\n── ACE-III Clock Score ──────────────────────────────────────")
    print(f"  Circle:  {result['circle']} / 1")
    print(f"  Numbers: {result['numbers']} / 2")
    print(f"  Hands:   {result['hands']} / 2")
    print(f"  Total:   {result['total']} / 5")


if __name__ == "__main__":
    main()
