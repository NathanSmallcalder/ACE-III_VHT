import glob
import json
import os

import pandas as pd

CSV_PATH = "synthetic_personas/synthetic_ace3.csv"

DOMAIN_COLUMNS = {
    "Attention": "attention_orientation",
    "Memory": "memory",
    "Fluency": "fluency",
    "Language": "language",
    "Visuospatial": "visuospatial",
}

COLUMN_QUESTIONS = {
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

    for domain, column in DOMAIN_COLUMNS.items():
        target = row[column]
        actual = result["domain_scores"].get(domain)
        flag = "" if target == actual else "   MISMATCH"
        print(f"  {domain:12} target={target:>3} actual={actual:>3}{flag}")
    flag = "" if row["ace3_total"] == result["total_score"] else "  <-- MISMATCH"
    print(f"  {'TOTAL':12} target={row['ace3_total']:>3} actual={result['total_score']:>3}{flag}")

    for column, questions in COLUMN_QUESTIONS.items():
        target = row[column]
        actual = sum(q["score"] for q in result.get("questions", []) if q["question_text"] in questions)
        flag = "" if target == actual else "  <-- MISMATCH"
        print(f"    [{target}/{actual}] {column}{flag}")
        if target != actual:
            print_mismatch_detail(questions, result)

def print_confusion_matrix(matrix):
    print("\n=== Confusion matrix (rows=true, cols=predicted) ===")
    print("            " + "".join(f"{c:>10}" for c in classes))
    for true_class in classes:
        print(f"  {true_class:10}" + "".join(f"{matrix[true_class][pred]:>10}" for pred in classes))

def main():
    targets = load_targets()
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for path, result, participant_id, cognitive_status in load_synthetic_results():
        key = (participant_id, cognitive_status)
        row = targets.get(key)
        if row is None:
            print(f"\n=== {os.path.basename(path)} ({key}) === no matching CSV row, skipped")
            continue
        compare(path, result, row)
        matrix[row["cognitive_status"]][classify_total(result["total_score"])] += 1

    print_confusion_matrix(matrix)

if __name__ == "__main__":
    main()
