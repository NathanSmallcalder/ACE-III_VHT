import random
import pandas as pd

items = [
    # Attention / Orientation (18)
    ("orientation_date",     "attention_orientation", 5),
    ("orientation_location", "attention_orientation", 5),
    ("registration",         "attention_orientation", 3),
    ("attention_serial7",    "attention_orientation", 5),
    # Memory (26)
    ("recall_3words",        "memory", 3),
    ("anterograde_memory",   "memory", 7),
    ("retrograde_memory",    "memory", 4),
    ("delayed_recall",       "memory", 7),
    ("recognition",          "memory", 5),
    # Fluency (14)
    ("verbal_fluency",       "fluency", 7),
    ("semantic_fluency",     "fluency", 7),
    # Language (26)
    ("comprehension",        "language", 3),
    ("writing",              "language", 2),
    ("verbal_repetition",    "language", 2),
    ("verbal_repetition_2",  "language", 2),
    ("naming",               "language", 12),
    ("comprehension_2",      "language", 4),
    ("reading",              "language", 1),
    # Visuospatial (16)
    ("infinity_copying",     "visuospatial", 1),
    ("cube_copying",         "visuospatial", 2),
    ("clock_copying",        "visuospatial", 5),
    ("dot_counting",         "visuospatial", 4),
    ("fragmented_letters",   "visuospatial", 4),
]

domains = ["attention_orientation", "memory", "fluency", "language", "visuospatial"]


recognition_thresholds = [2, 3, 5, 6, 7]

def recognition_floor(delayed_recall):
    """Lowest recognition score that can co-exist with `delayed_recall`.
    Recognition is only asked about address elements the participant failed to
    recall; anything they did recall is auto-awarded its point. So a participant
    who recalled all 7 elements must score the full 5 -- there is nothing left to
    ask them."""
    return sum(1 for threshold in recognition_thresholds if delayed_recall >= threshold)

ranges = { # https://pmc.ncbi.nlm.nih.gov/articles/PMC12207243/pdf/ENE-32-e70257.pdf
    "healthy": (0.88, 1.00),
    "mci": (0.77, 0.87),
    "dementia": (0.30, 0.76),
}
indicator = {"healthy": 0.33, "mci": 0.33, "dementia": 0.33}

def make_profile(pid: str) -> dict:
    # Sample cognitive status based on population weights
    selected_status = random.choices(list(indicator.keys()), weights=list(indicator.values()))[0]
    min_pct, max_pct = ranges[selected_status]

    for _ in range(200):
        profile = {"participant_id": pid, "cognitive_status": selected_status}
        base_ability = random.uniform(min_pct, max_pct)
        
        # Generate item-level scores with strict upper-bound capping
        for item_name, _domain, max_pts in items:
            noisy_pct = max(0.0, min(1.0, random.gauss(base_ability, 0.05)))
            profile[item_name] = min(max_pts, round(noisy_pct * max_pts))

        profile["recognition"] = max(profile["recognition"], recognition_floor(profile["delayed_recall"]))

        # Aggregate domain totals (Attention, Memory, Fluency, Language, Visuospatial)
        for dom in domains:
            profile[dom] = sum(profile[item_name] for item_name, d, _ in items if d == dom)

        # Total ACE-III score (Max 100)
        profile["ace3_total"] = sum(profile[dom] for dom in domains)

        if min_pct * 100 <= profile["ace3_total"] <= max_pct * 100:
            break

    return profile

def generate(n=200, seed=42):
    random.seed(seed)
    return pd.DataFrame([make_profile(f"Participant_{i}") for i in range(n)])

if __name__ == "__main__":
    df = generate(200)
    df.to_csv("synthetic_ace3.csv", index=False)
    print(f"Wrote {len(df)} profiles. Total score: mean {df['ace3_total'].mean():.1f}, "
          f"range {df['ace3_total'].min()}-{df['ace3_total'].max()}")
    print("\nStatus Breakdown:")
    print(df["cognitive_status"].value_counts())
    print("\nSample Rows:")
    print(df[["participant_id", "cognitive_status", "memory", "ace3_total"]].head())