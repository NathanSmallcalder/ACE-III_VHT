import os
import time
import cv2
import numpy as np
from imutils.perspective import four_point_transform

"""
Detects paper edges and records for the pen_paper task
"""

def resizer(image, width=500):
    # OpenCV image.shape returns (height, width, channels)
    h, w = image.shape[:2]
    height = int((h / w) * width)
    size = (width, height)
    return cv2.resize(image, size), size

def find_document_contour(frame):
    # Downsample frame to accelerate processing and standardize noise scale
    img_re, size = resizer(frame)

    detail = cv2.detailEnhance(img_re, sigma_s=20, sigma_r=0.15)
    gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_image = cv2.Canny(blur, 75, 200)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edge_image, kernel, iterations=1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = 0.1 * size[0] * size[1]
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            break  # Sorted descending; remaining contours are guaranteed too small
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            multiplier = frame.shape[1] / size[0]
            return np.squeeze(approx, axis=1).astype(float) * multiplier
    return None


def rectify(frame, points=None):
    """Crops and flattens paper from the frame using a 4-point perspective transform.
        Falls back to raw frame if no corner points are provided.
    """
    points = find_document_contour(frame)
    if points is None:
        return frame
    return four_point_transform(frame, points.astype(int))

def record_video(output_path, stop_event, max_duration=None, camera_index=0):
    """Records a video """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    # Fetch camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    start_time = time.time()
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            # If max_duration is set and elapsed time exceeds it, break loop to end thread
            if max_duration is not None and time.time() - start_time > max_duration:
                break
    finally:
        cap.release()
        writer.release()

    return output_path

def capture_drawing(output_path, camera_index=0):
    """Read a live webcam feed and capture a rectified photo of a hand-drawn sheet.
    Auto-captures as soon as a document outline is found, or after
    AUTO_CAPTURE_TIMEOUT seconds (using the last frame, rectified if an outline
    was found). Returns the path the rectified image was saved to.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    start_time = time.time()
    captured_frame = None
    captured_points = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera")
            # Check frame for paper boundary
            points = find_document_contour(frame)
            # Immediately capture upon detecting document corners
            if points is not None:
                captured_frame, captured_points = frame, points
                break
            if time.time() - start_time > 15: # before giving up after not finding paper
                print("[Camera] No document outline found in time; capturing raw frame.")
                captured_frame, captured_points = frame, points
                break
    finally:
        cap.release()

    # Rectify (crop/flatten) captured frame if points exist, then save to disk
    result = rectify(captured_frame, captured_points)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


