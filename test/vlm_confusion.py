import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import visual_tasks.clock_scorer as clock_scorer
import visual_tasks.infinity_scorer as infinity_scorer
import visual_tasks.cube_scorer as cube_scorer

CLOCKS_DIR = os.path.join(os.path.dirname(__file__), "Clocks")
INFINITY_DIR = os.path.join(os.path.dirname(__file__), "InfinitySymbol")
CUBE_DIR = os.path.join(os.path.dirname(__file__), "cube")


CLAUDE_MODEL = "claude-opus-5"
from contextlib import contextmanager

@contextmanager
def use_claude_vlm():
    client = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=os.environ["CLAUDE"],
        max_tokens=10000,
    )
    originals = {
        clock_scorer: clock_scorer.llm,
        cube_scorer: cube_scorer.llm,
        infinity_scorer: infinity_scorer.llm,
    }

    try:
        for module in originals:
            module.llm = client
        yield
    finally:
        for module, original_llm in originals.items():
            module.llm = original_llm

def collect_cases(DIR):
    cases = []
    for subfolder in os.listdir(DIR):
        expected_score = int(subfolder)
        subfolder_path = os.path.join(DIR, subfolder)
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            cases.append((image_path, expected_score, filename))
    return cases

def eval_vlm_clock():
    cases = collect_cases(CLOCKS_DIR)

    correct = 0
    total = len(cases)

    y_true = []
    y_pred = []

    print(f"Running VLM Evaluation on {total} cases...\n")

    for image_path, expected_score, filename in cases:
        actual_score = clock_scorer.score_clock_image(image_path)
        predicted_total = actual_score["total"]

        y_true.append(expected_score)
        y_pred.append(predicted_total)

        is_correct = predicted_total == expected_score
        if is_correct:
            correct += 1

        status = (
            "PASS"
            if is_correct
            else (
                f"FAIL "
                f"(Expected={expected_score}, "
                f"Predicted={predicted_total}, "
                f"Circle={actual_score['circle']}, "
                f"Numbers={actual_score['numbers']}, "
                f"Hands={actual_score['hands']})"
            )
        )

        print(f"[{status}] {filename}")

    accuracy = (correct / total) * 100 if total else 0

    print("\n" + "=" * 50)
    print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print("=" * 50)

    labels = [1, 2, 3, 4, 5]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Clock Drawing Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig("clock_confusion_matrix.png", dpi=300)
    plt.show()


with use_claude_vlm():
    eval_vlm_clock()