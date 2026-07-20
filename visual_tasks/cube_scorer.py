from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
CUBE_PROMPT = """You are analysing a hand-drawn attempt at copying a wire-frame cube, for clinical scoring purposes.

The target is a 3D wire-frame cube, drawn in any of the usual conventions (e.g. two offset squares connected
corner-to-corner, or an isometric box with a front/top/side face) — a complete wire-frame cube has 12 edges
(straight line segments) in total. Proportions do not need to be accurate.

Describe only what is visible in the drawing. Where a field allows "unclear", use it rather than guessing.
Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "edges_present": "<integer 0-12 — how many of the cube's 12 canonical edges (4 on the front face, 4 on the back face, 4 connecting the two) are represented by a line in the drawing, ignoring any extra stray marks>",
  "general_cube_shape": "<yes/no — does the drawing read as a recognisable 3D cube overall, regardless of exact style or proportions>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}
"""

llm = build_client(0.0, 20000)


def _describe_cube(drawn_path: str) -> dict:
    """Send the patient's cube drawing to the VLM and parse its structured description."""
    return describe_images(llm, CUBE_PROMPT, [drawn_path])


def score_cube(data: dict) -> dict:
    def get(field):
        return str(data.get(field, "")).strip().lower()

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    edges_present = to_int(get("edges_present"))
    cube_shape = get("general_cube_shape") == "yes"

    if edges_present is not None and edges_present >= 12:
        # "12 lines to score 2 points, even if the proportions are not perfect"
        total = 2
    elif cube_shape:
        # "fewer than 12 lines but a general cube shape is maintained"
        total = 1
    else:
        total = 0

    return {"total": total}


def score_cube_image(drawn_path: str) -> dict:
    """Describe a hand-drawn wire-cube copy via VLM and score it against the ACE-III cube criteria."""
    data = _describe_cube(drawn_path)
    result = score_cube(data)
    save_vlm_response("wire_cube", drawn_path, data, result)
    return result


if __name__ == "__main__":
    import os
    drawn_path = os.path.join(os.path.dirname(__file__), "wire_cube.png")
    print(f"Scoring: {os.path.basename(drawn_path)}")

    data = _describe_cube(drawn_path)

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in ["edges_present", "general_cube_shape", "notes"]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_cube(data)

    print("\n── ACE-III Wire Cube Score ─────────────────────────────────")
    print(f"  Total: {scores['total']} / 2")