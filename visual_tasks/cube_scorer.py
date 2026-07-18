from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
CUBE_PROMPT = """You are comparing a hand-drawn copy of a wire-frame cube against the reference diagram it was copied from, for clinical scoring purposes.

The first image is the REFERENCE the patient was shown. The second image is the patient's COPY.

Describe only what is visible in the copy, judged against the reference. Where a field allows "unclear", use it rather than guessing.
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "two_quadrilaterals_present": "<yes/no — are there two distinct four-sided shapes (front face and back face of the cube)>",
  "quadrilaterals_closed": "<yes/no — are both four-sided shapes fully closed, with no open corners>",
  "offset_direction_matches_reference": "<yes/no/unclear — is the back face offset diagonally from the front face in the same direction as the reference, giving a 3D appearance>",
  "connecting_lines_count": "<0/1/2/3/4/more — number of diagonal lines connecting corresponding corners of the two quadrilaterals>",
  "connecting_lines_roughly_parallel": "<yes/no/unclear — are the connecting lines roughly parallel to each other, as in the reference>",
  "reads_as_three_dimensional": "<yes/no — overall, does the copy read as a 3D cube rather than a flat 2D shape>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _describe_cube(reference_path: str, drawn_path: str) -> dict:
    """Send the reference cube and the patient's copy to the VLM and parse its structured comparison."""
    return describe_images(llm, CUBE_PROMPT, [reference_path, drawn_path])


def score_cube(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    quads_present = get("two_quadrilaterals_present") == "yes"
    quads_closed  = get("quadrilaterals_closed") == "yes"
    offset_ok     = get("offset_direction_matches_reference") == "yes"
    parallel_ok   = get("connecting_lines_roughly_parallel") == "yes"
    three_d       = get("reads_as_three_dimensional") == "yes"

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    line_count = to_int(get("connecting_lines_count"))
    lines_ok   = line_count == 4
    some_connecting_lines = line_count is not None and line_count >= 1

    if quads_present and quads_closed and three_d and offset_ok and lines_ok and parallel_ok:
        total = 2
    elif quads_present and quads_closed and three_d and some_connecting_lines:
        total = 1
    else:
        total = 0

    return {"total": total}


def score_cube_image(reference_path: str, drawn_path: str) -> dict:
    """Compare a hand-drawn cube copy against its reference via VLM and score it against the ACE-III cube criteria."""
    data = _describe_cube(reference_path, drawn_path)
    result = score_cube(data)
    save_vlm_response("wire_cube", drawn_path, data, result)
    return result


if __name__ == "__main__":
    import os
    reference_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "page4_img4.png")
    drawn_path = os.path.join(os.path.dirname(__file__), "wire_cube.png")
    print(f"Scoring: {os.path.basename(drawn_path)} against {os.path.basename(reference_path)}")

    data = _describe_cube(reference_path, drawn_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "two_quadrilaterals_present", "quadrilaterals_closed",
        "offset_direction_matches_reference", "connecting_lines_count",
        "connecting_lines_roughly_parallel", "reads_as_three_dimensional", "notes",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_cube(data)

    print("\n── ACE-III Wire Cube Score ─────────────────────────────────")
    print(f"  Total: {scores['total']} / 2")