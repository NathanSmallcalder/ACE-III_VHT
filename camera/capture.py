import os
import time

import cv2
import numpy as np
from imutils.perspective import four_point_transform

DEFAULT_CAMERA_INDEX = 0
STABLE_FRAMES_REQUIRED = 8      # consecutive frames the outline must hold steady before auto-capture
STABILITY_TOLERANCE = 15        # max corner drift (px) between frames to still count as "stable"
AUTO_CAPTURE_TIMEOUT = 15       # seconds to wait for a stable outline before giving up (headless mode)
RECORD_FPS = 30

""" Shrints an Image down to a width of width """
def resizer(image, width=500):
    h, w = image.shape[:2]
    height = int((h / w) * width)
    size = (width, height)
    return cv2.resize(image, size), size

""" Apply brightness and contrast to an image """
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def _find_document_contour(frame):
    """Locate the 4-point contour of a sheet of paper in `frame`, in the frame's own
    coordinate space. Returns None if no quadrilateral is found."""
    img_re, size = resizer(frame)
    detail = cv2.detailEnhance(img_re, sigma_s=20, sigma_r=0.15)
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_image = cv2.Canny(blur, 75, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel, iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    min_area = 0.1 * size[0] * size[1]
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break  # sorted descending, so nothing bigger remains
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            multiplier = frame.shape[1] / size[0]
            return np.squeeze(approx, axis=1).astype(float) * multiplier

    return None


def _rectify(frame, points=None):
    """Warp `frame` to a top-down view of the detected document. Falls back to the
    raw frame if no quadrilateral could be found."""
    if points is None:
        points = _find_document_contour(frame)
    if points is None:
        return frame
    return four_point_transform(frame, points.astype(int))


def document_scanner(image):
    """Detect + rectify a document in a still image. Returns None if no quadrilateral
    document outline was found."""
    points = _find_document_contour(image)
    if points is None:
        return None
    return four_point_transform(image, points.astype(int))


def capture_green_box(output_path=None, camera_index=DEFAULT_CAMERA_INDEX):
    """Grab a single frame from the webcam and crop it to the detected document
    outline (the green box drawn in the preview window). Returns None if no
    quadrilateral outline was found in that frame.

    If `output_path` is given, the cropped image is also saved there."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")
    finally:
        cap.release()

    cropped = document_scanner(frame)

    if cropped is not None and output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, cropped)

    return cropped


def record_video(output_path, stop_event, max_duration=None, camera_index=DEFAULT_CAMERA_INDEX):
    """Record a headless webcam clip to `output_path` until `stop_event` is set
    (from another thread) or `max_duration` seconds elapse, whichever comes first.

    Meant to run on a background thread while the main thread drives TTS/timers,
    the same split `capture_drawing` uses for photo capture."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, RECORD_FPS, (width, height))

    start_time = time.time()
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            if max_duration is not None and time.time() - start_time > max_duration:
                break
    finally:
        cap.release()
        writer.release()

    return output_path


def _corners_stable(prev, curr, tolerance=STABILITY_TOLERANCE):
    if prev is None or curr is None or prev.shape != curr.shape:
        return False
    return np.max(np.linalg.norm(prev - curr, axis=1)) < tolerance


def capture_drawing(output_path, preview=True, camera_index=DEFAULT_CAMERA_INDEX):
    """Read a live webcam feed and capture a rectified photo of a hand-drawn sheet.

    preview=True  -> shows the live feed with the detected outline overlaid; press
                      SPACE/ENTER to capture, ESC/q to cancel.
    preview=False -> runs headless: auto-captures once the outline holds steady for
                      STABLE_FRAMES_REQUIRED frames, or after AUTO_CAPTURE_TIMEOUT
                      seconds (using the last frame, rectified if an outline was found).

    Returns the path the rectified image was saved to.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    window_name = "ACE-III Capture - SPACE/ENTER to capture, ESC to cancel"
    prev_points = None
    stable_count = 0
    start_time = time.time()
    captured_frame = None
    captured_points = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera")

            points = _find_document_contour(frame)

            if preview:
                display = frame.copy()
                if points is not None:
                    cv2.drawContours(display, [points.astype(int)], -1, (0, 255, 0), 3)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32):  # ENTER or SPACE
                    captured_frame, captured_points = frame, points
                    break
                if key in (27, ord("q")):  # ESC or q
                    raise RuntimeError("Capture cancelled by user")
            else:
                stable_count = stable_count + 1 if _corners_stable(prev_points, points) else 0
                prev_points = points
                if points is not None and stable_count >= STABLE_FRAMES_REQUIRED:
                    captured_frame, captured_points = frame, points
                    break
                if time.time() - start_time > AUTO_CAPTURE_TIMEOUT:
                    print("[Camera] No stable document outline found in time; capturing raw frame.")
                    captured_frame, captured_points = frame, points
                    break
    finally:
        cap.release()
        if preview:
            cv2.destroyWindow(window_name)

    result = _rectify(captured_frame, captured_points)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path
