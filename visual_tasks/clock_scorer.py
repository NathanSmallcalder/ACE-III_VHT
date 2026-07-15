import base64
import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
# LLM/LLM.py
from langchain_openai import ChatOpenAI

# Single source of truth for the backend connection
_BASE_URL = "http://localhost:1234/v1"
_API_KEY  = "lm-studio"
_MODEL    = "qwen/qwen3-vl-4b"

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────
def _build(temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=_BASE_URL,
        api_key=_API_KEY,
        model=_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

# ── VLM prompt ────────────────────────────────────────────────────────────────
CLOCK_PROMPT = """You are analysing a hand-drawn clock image for clinical scoring purposes.

Describe only what is visible in the drawing. Where a field allows "unclear", use it rather than guessing. Where a field does not apply
(e.g. hand fields when no hands are drawn), use "none".
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "circle_present": "<yes/no>",
  "circle_closed": "<yes/no/unclear>",

  "all_12_present": "<yes/no>",
  "missing_numbers": "<comma-separated list of integers, or none>",
  "duplicated_numbers": "<comma-separated list of integers, or none>",
  "numbers_in_correct_order": "<yes/no — 1 to 12 in clockwise sequence>",
  "numbers_outside_circle": "<yes/no — any numbers clearly outside the circle boundary, with visible space between number and circle edge>",
  "numbers_evenly_spaced": "<yes/approximately/no — evenly distributed around the full circle, not bunched into one region>",

  "hand_count": "<0/1/2/more>",
  "hands_originate_from_centre": "<yes/no/unclear/none — do the hands start from roughly the centre of the clock face?>",
  "shorter_hand_points_to": "<integer 1-12, or unclear, or none>",
  "longer_hand_points_to": "<integer 1-12, or unclear, or none>",
  "hands_same_length": "<yes/no/unclear/none — if yes, report the two hands under shorter/longer in any order>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}

Rules for hands: a hand's direction is the number its TIP points toward from the centre. If a hand has an arrowhead, follow the arrow. Do not report the number nearest the hand's shaft.
""" 

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
llm = _build(0.0, 20000)

def _describe_clock(image_path: str) -> dict:
    """Send the clock drawing to the VLM and parse its structured description."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = MIME_TYPES.get(ext, "image/png")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": CLOCK_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ])
    ])

    raw = response.content
    if isinstance(raw, list):
        raw = raw[0]["text"]

    # Strip markdown fences if model ignores instructions
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def score_clock(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    circle_score = 1 if get("circle_present") == "yes" and get("circle_closed") == "yes" else 0

    all_12          = get("all_12_present") == "yes"
    numbers_outside = get("numbers_outside_circle") == "yes"
    spacing         = get("numbers_evenly_spaced")

    if all_12 and not numbers_outside and spacing in ("yes", "approximately"):
        numbers_score = 2
    elif all_12:
        numbers_score = 1
    else:
        numbers_score = 0

    TARGET = {2, 5}  # ten past five: minute→2, hour→5

    hand_count    = get("hand_count")
    same_length   = get("hands_same_length")
    lengths_differ = same_length == "no"  # hands_same_length: no → they differ

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    h1 = to_int(get("shorter_hand_points_to"))
    h2 = to_int(get("longer_hand_points_to"))
    hand_positions = {x for x in [h1, h2] if x is not None}
    both_correct = hand_positions == TARGET
    one_correct  = len(hand_positions & TARGET) == 1

    if hand_count in ("0", "1", ""):
        hands_score = 0
    elif hand_count == "2":
        if both_correct and lengths_differ:
            hands_score = 2
        elif both_correct:
            hands_score = 1
        elif one_correct and lengths_differ:
            hands_score = 1
        else:
            hands_score = 0
    else:
        hands_score = 0  # more than 2 hands

    return {
        "circle":  circle_score,
        "numbers": numbers_score,
        "hands":   hands_score,
        "total":   circle_score + numbers_score + hands_score,
    }


def score_clock_image(image_path: str) -> dict:
    """Describe a hand-drawn clock image via VLM and score it against the ACE-III clock criteria."""
    data = _describe_clock(image_path)
    return score_clock(data)


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "clock.png")
    print(f"Scoring: {os.path.basename(image_path)}")

    data = _describe_clock(image_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "circle_present", "circle_closed", "all_12_present",
        "missing_numbers", "numbers_outside_circle", "numbers_evenly_spaced",
        "hand_count", "shorter_hand_points_to", "longer_hand_points_to", "hands_same_length",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_clock(data)

    print("\n── ACE-III Clock Score ──────────────────────────────────────")
    print(f"  Circle:  {scores['circle']} / 1")
    print(f"  Numbers: {scores['numbers']} / 2")
    print(f"  Hands:   {scores['hands']} / 2")
    print(f"  Total:   {scores['total']} / 5")
