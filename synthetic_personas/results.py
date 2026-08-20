import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

CSV_PATH = "synthetic_personas/synthetic_ace3.csv"

domains = {
    "Attention": "attention_orientation",
    "Memory": "memory",
    "Fluency": "fluency",
    "Language": "language",
    "Visuospatial": "visuospatial",
}

questions = {
    "orientation_date": {"Orientation to Time: What is the day, date, month, year, and season?"},
    "orientation_location": {"Orientation to Place: What is the number or floor, street, town, county, and country?"},
    "registration": {"Registration: Repeat the three words lemon, key, and ball."},
    "attention_serial7": {"Serial 7s: Subtract 7 from 100 and keep subtracting 7."},
    "recall_3words": {"Three-word recall: Which three words did I ask you to remember?"},
    "anterograde_memory": {"Name and address learning: Harry Barnes, 73 Orchard Close, Kingsbridge, Devon."},
    "retrograde_memory": {
        "Retrograde memory: Name of the current UK Prime Minister.",
        "Retrograde memory: Name of the first female UK Prime Minister.",
        "Retrograde memory: Name of the current President of the United States.",
        "Retrograde memory: Name of the US President assassinated in the 1960s.",
    },
    "delayed_recall": {"Delayed recall: Recall the name and address from earlier."},
    "recognition": {"Recognition: Identify correct name and address elements from choices."},
    "verbal_fluency": {"Letter fluency: Generate words beginning with the letter P in one minute."},
    "semantic_fluency": {"Category fluency: Name as many animals as possible in one minute."},
    "comprehension": {"Comprehension: Follow three-stage commands with pencil and paper."},
    "writing": {"Writing: Write two complete sentences."},
    "verbal_repetition": {"Word repetition: Repeat caterpillar, eccentricity, unintelligible, statistician."},
    "verbal_repetition_2": {
        "Sentence repetition: Repeat the phrase - All that glitters is not gold.",
        "Sentence repetition: Repeat the phrase - A stitch in time saves nine.",
    },
    "naming": {"Naming: Name the 12 pictures shown on screen."},
    "comprehension_2": {
        "Comprehension: Which picture is associated with the monarchy?",
        "Comprehension: Which picture is the marsupial?",
        "Comprehension: Which picture is found in the Antarctic?",
        "Comprehension: Which picture has a nautical connection?",
    },
    "reading": {"Reading: Read the following words aloud - sew, pint, soot, dough, height."},
    "infinity_copying": {"Infinity Diagram: Copy the infinity diagram."},
    "cube_copying": {"Wire Cube: Copy the wire cube."},
    "clock_copying": {"Clock: Draw a clock face showing ten past five."},
    "dot_counting": {
        "Dot counting: Count the dots in the first array.",
        "Dot counting: Count the dots in the second array.",
        "Dot counting: Count the dots in the third array.",
        "Dot counting: Count the dots in the fourth array.",
    },
    "fragmented_letters": {"Fragmented letters: Identify the fragmented letter."},
}

classes = ["healthy", "mci", "dementia"]

# VLM-scored drawings mismatch constantly; exclude them so the rest stands out
drawing_tasks = {"clock_copying", "cube_copying", "infinity_copying"}

def classify_total(total):
    if total >= 88:
        return "healthy"
    if total >= 77:
        return "mci"
    return "dementia"

def load_targets():
    df = pd.read_csv(CSV_PATH)
    return {(row["participant_id"], row["cognitive_status"]): row for _, row in df.iterrows()}

def load_synthetic_results():
    results = []
    for path in sorted(glob.glob(os.path.join("results", "*.json"))):
        with open(path) as f:
            result = json.load(f)
        name = result.get("patient", {}).get("name", "")
        participant_id, _, cognitive_status = name.rpartition("_")
        if cognitive_status in classes:
            results.append((path, result, participant_id, cognitive_status))
    return results

def print_mismatch_detail(questions, result):
    for q in result.get("questions", []):
        if q["question_text"] in questions:
            print(f"        Q: {q['question_text']}")
            print(f"        A: {q['response']}")

def compare(path, result, row):
    print(f"\n=== {os.path.basename(path)} ({row['participant_id']}_{row['cognitive_status']}) ===")

    for domain, column in domains.items():
        target = row[column]
        actual = result["domain_scores"].get(domain)
        flag = "" if target == actual else "   MISMATCH"
        print(f"  {domain:12} target={target:>3} actual={actual:>3}{flag}")
    flag = "" if row["ace3_total"] == result["total_score"] else "  <-- MISMATCH"
    print(f"  {'TOTAL':12} target={row['ace3_total']:>3} actual={result['total_score']:>3}{flag}")

    for column, question in questions.items():
        target = row[column]
        actual = sum(q["score"] for q in result.get("questions", []) if q["question_text"] in question)
        flag = "" if target == actual else "  <-- MISMATCH"
        print(f"    [{target}/{actual}] {column}{flag}")
        if target != actual:
            print_mismatch_detail(question, result)

def save_confusion_matrix(matrix, out_path="classification_confusion_matrix.png"):
    """Render the true-vs-predicted cognitive status matrix to a PNG."""
    cm = np.array([[matrix[true][pred] for pred in classes] for true in classes])
    correct = sum(matrix[c][c] for c in classes)
    total = int(cm.sum())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    accuracy = f"accuracy {correct}/{total} ({100 * correct / total:.1f}%)" if total else ""
    ax.set_title(f"ACE-III classification confusion matrix\n{accuracy}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\n  confusion matrix written to {out_path} -- {accuracy}")

def pct(hits, n):
    return f"{100 * hits / n:>9.1f}%" if n else f"{'-':>10}"

def print_domain_accuracy(hits, counts, perfect, near):
    """Percentage of participants whose domain score exactly matched the CSV target,
    split by true cognitive status."""
    print("\n=== Domain accuracy (exact score match) ===")
    print(f"  {'domain':14}" + "".join(f"{c:>10}" for c in classes) + f"{'overall':>10}")

    for domain in domains:
        row = "".join(pct(hits[c][domain], counts[c]) for c in classes)
        row += pct(sum(hits[c][domain] for c in classes), sum(counts.values()))
        print(f"  {domain:14}" + row)

    # Every domain of every participant, pooled
    per_class = {c: counts[c] * len(domains) for c in classes}
    row = "".join(pct(sum(hits[c].values()), per_class[c]) for c in classes)
    row += pct(sum(sum(hits[c].values()) for c in classes), sum(per_class.values()))
    print(f"  {'ALL DOMAINS':14}" + row)

    # Participants where every domain matched its target
    row = "".join(pct(perfect[c], counts[c]) for c in classes)
    row += pct(sum(perfect.values()), sum(counts.values()))
    print(f"  {'100% RUNS':14}" + row)

    # Visuospatial is VLM-scored, so also report agreement to within 1 point
    row = "".join(pct(near[c], counts[c]) for c in classes)
    row += pct(sum(near.values()), sum(counts.values()))
    print(f"  {'Visuo +/-1':14}" + row)

def print_drawing_agreement(exact, near, counts):
    """Per-task agreement for the VLM-scored drawings, exact and to within 1 point."""
    print("\n=== Drawing task agreement ===")
    print(f"  {'task':16}{'exact':>10}{'+/-1':>10}{'n':>6}")
    for column in drawing_tasks:
        n = counts[column]
        print(f"  {column:16}{pct(exact[column], n)}{pct(near[column], n)}{n:>6}")

def print_non_drawing_mismatches(entries):
    """Transcripts that mis-scored somewhere other than the clock, cube or infinity."""
    print("\n=== Mismatches outside the drawing tasks ===")
    if not entries:
        print("  none")
        return
    for participant_id, status, flagged in entries:
        print(f"  synthetic_transcripts/{participant_id}_{status}.json")
        for column, target, actual in flagged:
            print(f"      {column:22} target={target:>3} actual={actual:>3}")
    print(f"\n  {len(entries)} of the transcripts affected")

def main():
    targets = load_targets()
    matrix = {t: {p: 0 for p in classes} for t in classes}
    domain_hits = {c: {d: 0 for d in domains} for c in classes}
    status_counts = {c: 0 for c in classes}
    perfect_runs = {c: 0 for c in classes}
    visuo_near = {c: 0 for c in classes}
    mismatches = []
    draw_counts = {c: 0 for c in drawing_tasks}
    draw_exact = {c: 0 for c in drawing_tasks}
    draw_near = {c: 0 for c in drawing_tasks}
    for path, result, participant_id, cognitive_status in load_synthetic_results():
        key = (participant_id, cognitive_status)
        row = targets.get(key)
        if row is None:
            print(f"\n=== {os.path.basename(path)} ({key}) === no matching CSV row, skipped")
            continue
        compare(path, result, row)
        status = row["cognitive_status"]
        matrix[status][classify_total(result["total_score"])] += 1
        status_counts[status] += 1
        matched = 0
        for domain, column in domains.items():
            if row[column] == result["domain_scores"].get(domain):
                domain_hits[status][domain] += 1
                matched += 1
        if matched == len(domains):
            perfect_runs[status] += 1
        if abs(row["visuospatial"] - result["domain_scores"].get("Visuospatial", 0)) <= 1:
            visuo_near[status] += 1

        flagged = []
        for column, question in questions.items():
            if column in drawing_tasks:
                actual = sum(q["score"] for q in result.get("questions", []) if q["question_text"] in question)
                draw_counts[column] += 1
                draw_exact[column] += row[column] == actual
                draw_near[column] += abs(row[column] - actual) <= 1
                continue
            actual = sum(q["score"] for q in result.get("questions", []) if q["question_text"] in question)
            if row[column] != actual:
                flagged.append((column, row[column], actual))
        if flagged:
            mismatches.append((participant_id, status, flagged))

    save_confusion_matrix(matrix)
    print_domain_accuracy(domain_hits, status_counts, perfect_runs, visuo_near)
    print_drawing_agreement(draw_exact, draw_near, draw_counts)
    print_non_drawing_mismatches(mismatches)

if __name__ == "__main__":
    main()
