from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
INFINITY_PROMPT = """You are analysing a hand-drawn attempt at copying an interlocking infinity-loop diagram, for clinical scoring purposes.

The target diagram (not shown) is two separate figure-eight (infinity symbol) loops, placed side by side so
they overlap in the middle: this creates 4 visible rounded loop bumps in total (leftmost loop, two
overlapping loops in the centre, and the rightmost loop), with the two central loops crossing over/under
each other rather than merely touching.

Describe only what is visible in the drawing. Where a field allows "unclear", use it rather than guessing.
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "loop_count": "<integer, or unclear — total number of distinct rounded loop bumps in the drawing>",
  "each_loop_closed": "<yes/no — is every loop in the drawing a fully closed curve, with no open gaps>",
  "central_loops_cross": "<yes/no/unclear — do the two central loops visibly cross over/under each other, rather than just touching or merging>",
  "drawn_as_continuous_curve": "<yes/no — is the drawing done as one continuous line, without obvious large breaks>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _describe_infinity(drawn_path: str) -> dict:
    """Send the patient's infinity-diagram drawing to the VLM and parse its structured description."""
    return describe_images(llm, INFINITY_PROMPT, [drawn_path])


def score_infinity(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    checks = [
        to_int(get("loop_count")) == 4,
        get("each_loop_closed") == "yes",
        get("central_loops_cross") == "yes",
        get("drawn_as_continuous_curve") == "yes",
    ]

    total = 1 if all(checks) else 0

    return {"total": total}


def score_infinity_image(drawn_path: str) -> dict:
    """Describe a hand-drawn infinity diagram copy via VLM and score it against the ACE-III criteria."""
    data = _describe_infinity(drawn_path)
    result = score_infinity(data)
    save_vlm_response("infinity_diagram", drawn_path, data, result)
    return result


if __name__ == "__main__":
    import os
    drawn_path = os.path.join(os.path.dirname(__file__), "infinity_diagram.png")
    print(f"Scoring: {os.path.basename(drawn_path)}")

    data = _describe_infinity(drawn_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "loop_count", "each_loop_closed",
        "central_loops_cross", "drawn_as_continuous_curve", "notes",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_infinity(data)

    print("\n── ACE-III Infinity Diagram Score ───────────────────────────")
    print(f"  Total: {scores['total']} / 1")