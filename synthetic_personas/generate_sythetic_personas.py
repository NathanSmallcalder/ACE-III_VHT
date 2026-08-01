import random
import pandas as pd

ITEMS = [
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

DOMAINS = ["attention_orientation", "memory", "fluency", "language", "visuospatial"]

# Each cognitive status has a "percent correct" range. A persona's score on
# every item is max_score times a random percent drawn from its status's
# range, so the whole profile is consistently healthy/mci/dementia_risk
# rather than each item being independently random.
STATUS_RANGES = {
    "healthy": (0.90, 1.00),
    "mci": (0.65, 0.89),
    "dementia_risk": (0.30, 0.64),
}
STATUS_WEIGHTS = {"healthy": 0.7, "mci": 0.2, "dementia_risk": 0.1}

def make_profile(pid: str) -> dict:
    row = {"participant_id": pid}

    status = random.choices(list(STATUS_WEIGHTS), weights=list(STATUS_WEIGHTS.values()))[0]
    row["cognitive_status"] = status
    low, high = STATUS_RANGES[status]

    row["age_at_assessment"] = round(random.gauss(77, 2))

    for name, _domain, max_score in ITEMS:
        percent_correct = random.uniform(low, high)
        row[name] = round(percent_correct * max_score)

    for domain in DOMAINS:
        row[domain] = sum(row[name] for name, d, _ in ITEMS if d == domain)

    row["ace3_total"] = sum(row[domain] for domain in DOMAINS)
    row["test_complete"] = 1
    row["time_minutes"] = round(random.gauss(18.5, 2.5), 1)

    return row

def generate(n=1000, seed=42):
    random.seed(seed)
    return pd.DataFrame([make_profile(f"NSHD{100000 + i}") for i in range(n)])

if __name__ == "__main__":
    df = generate(1000)
    df.to_csv("synthetic_ace3.csv", index=False)
    print(f"Wrote {len(df)} profiles. Total score: mean {df['ace3_total'].mean():.1f}, "
          f"range {df['ace3_total'].min()}-{df['ace3_total'].max()}")
    print("\nStatus Breakdown:")
    print(df["cognitive_status"].value_counts())
    print("\nSample Rows:")
    print(df[["participant_id", "cognitive_status", "memory", "ace3_total"]].head())