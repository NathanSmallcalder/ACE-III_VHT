import os
from dotenv import load_dotenv

from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# ── VLM prompt ────────────────────────────────────────────────────────────────
clock_prompt = """You are analysing a hand-drawn clock image for clinical scoring purposes.
Describe only what is visible in the drawing. Where a field does not apply
(e.g. hand fields when no hands are drawn), use "none".

Before giving the JSON, reason step by step about the two hands specifically:
1. Locate the centre of the clock face.
2. For each hand, trace it from the centre to its tip and note roughly how far the tip is from the centre AS A FRACTION of the distance from the centre to the drawn circle's edge along that same direction (e.g. "about half way to the edge", "reaches about 70% of the way to the edge"). Do not judge length by comparing raw pixel/line length between the two hands, and do not judge it by which digit the hand happens to be near — a hand pointing toward a digit that sits closer to the (often uneven) circle edge in that direction can look longer even if it is proportionally the shorter hand. Compare the two fractions you computed to decide which hand is actually shorter and which is longer.

Respond with a single JSON object and nothing else — no explanation.
{
  "circle_present": "<yes/no>",
  "circle_closed": "<yes/no/unclear>",

  "all_12_present": "<yes/no>",
  "missing_numbers": "<comma-separated list of integers, or none>",
  "duplicated_numbers": "<comma-separated list of integers, or none>",
  "numbers_in_correct_order": "<yes/no — 1 to 12 in clockwise sequence>",
  "numbers_outside_circle": "<yes/no — any numbers where the entire digit is drawn on the exterior side of the circle line>",
  "numbers_evenly_spaced": "<yes/approximately/no — evenly distributed around the full circle, not bunched into one region. A slight rotation of the overall clock face is acceptable.>",

  "hand_count": "<0/1/2/more>",
  "hands_originate_from_centre": "<yes/no/unclear/none — do the hands start from roughly the centre of the clock face?>",
  "shorter_hand_points_to": "<integer 1-12, or none>",
  "longer_hand_points_to": "<integer 1-12, or none>",
  "hands_same_length": "<yes/no/none — if yes, report the two hands under shorter/longer in any order>",

  "notes": "<one short sentence flagging anything unusual not captured above, or none>"
}

Rules for hands: a hand's direction is the number its TIP points toward from the centre. If a hand has an arrowhead, follow the arrow. Do not report the number nearest the hand's shaft.
"""

llm = build_client(0.0, 20000)

def score_clock(data: dict) -> dict:
    """Score against the ACE-III / M-ACE clock criteria (0-5).

    Branch comments quote the ACE-III and M-ACE English Guide 2017 (updated 12/2/19).
    """
    def get(field):
        return str(data.get(field, "")).strip().lower()

    # ── Circle: "1 point maximum if it is a reasonable circle" ──
    circle_score = 1 if get("circle_present") == "yes" and get("circle_closed") == "yes" else 0

    # ── Numbers ──
    all_12          = get("all_12_present") == "yes"
    duplicates      = get("duplicated_numbers") not in ("none", "")
    numbers_outside = get("numbers_outside_circle") == "yes"
    correct_order   = get("numbers_in_correct_order") == "yes"
    spacing         = get("numbers_evenly_spaced")

    if not all_12 or duplicates:
        # "0 points if not all numbers are included"
        # Duplicates → 0 per the guide's exemplar: "there are 2 number 10s (0)"
        numbers_score = 0
    elif (not numbers_outside and correct_order
          and spacing in ("yes", "approximately")):
        # "2 points if all numbers are included within the circle and numbers are
        #  evenly distributed. A slight rotation to the overall clock face is acceptable."
        # (correct_order is our interpretation: a jumbled sequence is not treated
        #  as evenly distributed, even if geometrically spread out.)
        numbers_score = 2
    else:
        # "1 point if all numbers are included but the numbers are either outside
        #  of the circle or the numbers are unevenly spaced"
        numbers_score = 1

    # ── Hands ──
    hour_target, min_target = 5, 2  # ten past five: hour→5, minute→2
    target = {hour_target, min_target}

    hand_count  = get("hand_count")
    same_length = get("hands_same_length")

    def to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    h1 = to_int(get("shorter_hand_points_to"))  # shorter (hour) hand's target
    h2 = to_int(get("longer_hand_points_to"))   # longer (minute) hand's target

    if hand_count != "2":
        # "0 point if one hand is drawn" (and no credit without hands)
        hands_score = 0
    else:
        hand_positions = {x for x in [h1, h2] if x is not None}
        both_numbers_correct = hand_positions == target
        # Correct lengths means the hands genuinely differ in length AND the
        # shorter one is on the hour target, the longer on the minute target.
        lengths_differ  = same_length == "no"
        length_correct  = lengths_differ and h1 == hour_target and h2 == min_target

        if both_numbers_correct and length_correct:
            # "2 points if both hands are drawn, lengths are correct and placed
            #  on correct numbers"
            hands_score = 2
        elif both_numbers_correct:
            # "1 point if both hands are drawn and placed on the correct numbers
            #  but lengths are incorrect"
            hands_score = 1
        elif lengths_differ and (h1 == hour_target or h2 == min_target):
            # "1 point if both hands are drawn but only one hand is placed on the
            #  correct number and drawn with correct length"
            # lengths_differ guard: with even lengths the shorter/longer slots are
            # arbitrary, and per the guide's exemplar "one number correct but
            # lengths are even" scores 0.
            hands_score = 1
        else:
            # "0 points if two hands are drawn but both lengths are incorrect and
            #  one number is correct" / "0 point if two hands are drawn but both
            #  lengths and numbers are incorrect"
            hands_score = 0

    return {
        "circle":  circle_score,
        "numbers": numbers_score,
        "hands":   hands_score,
        "total":   circle_score + numbers_score + hands_score,
    }


def score_clock_image(image_path: str) -> dict:
    """Describe a hand-drawn clock image via VLM and score it against the ACE-III clock criteria."""
    data = describe_images(llm, clock_prompt, [image_path])
    result = score_clock(data)
    save_vlm_response("clock", image_path, data, result)
    return result


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "clock.png")
    print(f"Scoring: {os.path.basename(image_path)}")

    data = describe_images(llm, clock_prompt, [image_path])

    print("\n── Parsed Fields ────────────────────────────────────────────")
    for field in [
        "circle_present", "circle_closed", "all_12_present",
        "missing_numbers", "duplicated_numbers", "numbers_outside_circle",
        "numbers_evenly_spaced", "hand_count",
        "shorter_hand_points_to", "longer_hand_points_to", "hands_same_length",
    ]:
        print(f"  {field}: {str(data.get(field, '')).strip().lower()}")

    scores = score_clock(data)

    print("\n── ACE-III Clock Score ──────────────────────────────────────")
    print(f"  Circle:  {scores['circle']} / 1")
    print(f"  Numbers: {scores['numbers']} / 2")
    print(f"  Hands:   {scores['hands']} / 2")
    print(f"  Total:   {scores['total']} / 5")