from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
INFINITY_PROMPT = """You are comparing a hand-drawn copy of an interlocking infinity-loop diagram against the reference diagram it was copied from, for clinical scoring purposes.

The first image is the REFERENCE the patient was shown. The second image is the patient's COPY.

Describe only what is visible in the copy, judged against the reference. Where a field allows "unclear", use it rather than guessing.
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "loop_count_matches_reference": "<yes/no/unclear — does the copy have the same number of loops as the reference>",
  "each_loop_closed": "<yes/no — is every loop in the copy a fully closed curve, with no open gaps>",
  "crossing_pattern_matches_reference": "<yes/no/unclear — do the loops cross over/under each other in the same places and arrangement as the reference>",
  "drawn_as_continuous_curve": "<yes/no — is the copy drawn as one continuous line, without obvious large breaks>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _describe_infinity(reference_path: str, drawn_path: str) -> dict:
    """Send the reference infinity diagram and the patient's copy to the VLM and parse its structured comparison."""
    return describe_images(llm, INFINITY_PROMPT, [reference_path, drawn_path])


def score_infinity(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    checks = [
        get("loop_count_matches_reference") == "yes",
        get("each_loop_closed") == "yes",
        get("crossing_pattern_matches_reference") == "yes",
        get("drawn_as_continuous_curve") == "yes",
    ]

    total = 1 if all(checks) else 0

    return {"total": total}


def score_infinity_image(reference_path: str, drawn_path: str) -> dict:
    """Compare a hand-drawn infinity diagram copy against its reference via VLM and score it against the ACE-III criteria."""
    data = _describe_infinity(reference_path, drawn_path)
    result = score_infinity(data)
    save_vlm_response("infinity_diagram", drawn_path, data, result)
    return result


if __name__ == "__main__":
    import os
    reference_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "page4_img3.png")
    drawn_path = os.path.join(os.path.dirname(__file__), "infinity_diagram.png")
    print(f"Scoring: {os.path.basename(drawn_path)} against {os.path.basename(reference_path)}")

    data = _describe_infinity(reference_path, drawn_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "loop_count_matches_reference", "each_loop_closed",
        "crossing_pattern_matches_reference", "drawn_as_continuous_curve", "notes",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_infinity(data)

    print("\n── ACE-III Infinity Diagram Score ───────────────────────────")
    print(f"  Total: {scores['total']} / 1")