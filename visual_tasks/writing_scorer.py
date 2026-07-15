import re
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from visual_tasks.vlm_client import build_client, encode_image

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
WRITING_PROMPT = """You are transcribing and judging a photo of handwritten sentences, for clinical scoring purposes (ACE-III cognitive assessment sentence-writing task).

Read the handwriting in the photo carefully. Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "transcription": "<your best-effort transcription of everything handwritten in the photo>",
  "sentence_count": "<0/1/2/more — number of distinct complete sentences written; a sentence needs a subject and a verb, a phrase/place-name/person's-name alone does not count>",
  "has_grammar_or_spelling_errors": "<yes/no/unclear — true if ANY of the written sentences contains a grammar or spelling error>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _describe_writing(image_path: str) -> dict:
    """Send the handwriting photo to the VLM and parse its structured description.
    Returns {} on any connection failure or malformed/unexpected response, so callers always get a dict."""
    try:
        response = llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": WRITING_PROMPT},
                {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
            ])
        ])
    except Exception:
        return {}

    raw = response.content
    if isinstance(raw, list):
        raw = raw[0]["text"]

    # Strip markdown fences if model ignores instructions
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def score_writing(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    count = to_int(get("sentence_count"))
    has_errors = get("has_grammar_or_spelling_errors") == "yes"

    if count is not None and count >= 2 and not has_errors:
        total = 2
    elif count is not None and count >= 2:
        total = 1
    elif count == 1 and not has_errors:
        total = 1
    else:
        total = 0

    return {"total": total, "transcription": data.get("transcription", "")}


def score_writing_image(image_path: str) -> dict:
    """Read a photo of handwritten sentences via VLM and score it against the ACE-III writing criteria."""
    data = _describe_writing(image_path)
    return score_writing(data)


if __name__ == "__main__":
    import os
    image_path = os.path.join(os.path.dirname(__file__), "writing.png")
    print(f"Scoring: {os.path.basename(image_path)}")

    data = _describe_writing(image_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in ["transcription", "sentence_count", "has_grammar_or_spelling_errors", "notes"]:
        print(f"  {field}: {str(data.get(field, '')).strip()}")

    scores = score_writing(data)

    print("\n── ACE-III Writing Score ────────────────────────────────────")
    print(f"  Total: {scores['total']} / 2")
