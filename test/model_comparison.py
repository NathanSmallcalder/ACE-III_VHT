import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, mean_squared_error, mean_absolute_error

from LLM.vlm import base_url, api_key, describe_images
from visual_tasks.clock_scorer import clock_prompt, score_clock
from visual_tasks.cube_scorer import cube_prompt, score_cube
from visual_tasks.infinity_scorer import infinity_prompt, score_infinity

load_dotenv()

clocks = os.path.join(os.path.dirname(__file__), "Clocks")
infinity = os.path.join(os.path.dirname(__file__), "InfinitySymbol")
cube = os.path.join(os.path.dirname(__file__), "cube")

models = [
    "qwen/qwen3-vl-4b",
    "claude-opus-5",
    "google/gemma-4-e4b",
    "qwen/qwen3-vl-8b",
    "google/gemini-3.6-flash",
]

tasks = {
    "clock": (clocks, clock_prompt, score_clock),
    "cube": (cube, cube_prompt, score_cube),
    "infinity": (infinity, infinity_prompt, score_infinity),
}

clock_runs = 3


def make_client(name: str):
    if name.startswith("claude"):
        return ChatAnthropic(model=name, api_key=os.environ["CLAUDE"], max_tokens=10000)
    if name.startswith("google/gemini"):
        return ChatGoogleGenerativeAI(model=name.split("/", 1)[1], temperature=0, max_output_tokens=10000)
    return ChatOpenAI(base_url=base_url, api_key=api_key, model=name, temperature=0, max_tokens=20000)


def collect_cases(directory):
    """(image_path, expected_score, filename) for every reference image, taking the
    expected score from the folder name."""
    cases = []
    for subfolder in sorted(os.listdir(directory)):
        expected_score = int(subfolder)
        subfolder_path = os.path.join(directory, subfolder)
        for filename in sorted(os.listdir(subfolder_path)):
            cases.append((os.path.join(subfolder_path, filename), expected_score, filename))
    return cases


def eval_vlm_clock(client, model):
    cases = collect_cases(clocks)
    correct = 0
    total = len(cases)

    y_true = []
    y_pred = []
    answers = []
    components = []

    print(f"Running {model} on {total} clock cases...\n")

    for image_path, expected_score, filename in cases:
        data = describe_images(client, clock_prompt, [image_path])
        answers.append(data or {})
        actual_score = score_clock(data) if data else {}
        components.append(actual_score)
        predicted_total = actual_score.get("total", 0)

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
                f"Circle={actual_score.get('circle')}, "
                f"Numbers={actual_score.get('numbers')}, "
                f"Hands={actual_score.get('hands')})"
            )
        )

        print(f"[{status}] {model} | {filename}")

    # what this model answered for every field, image by image (columns are the images below)
    fields = list(dict.fromkeys(field for answer in answers for field in answer))
    print(f"\n{model} field answers:")
    print(f"{'field':<30}" + "".join(f"{i + 1:>9}" for i in range(len(cases))))
    for field in fields:
        print(f"{field:<30}" + "".join(
            f"{str(answer.get(field, '-')).strip().lower()[:8]:>9}" for answer in answers))
    for i, (_, expected_score, filename) in enumerate(cases):
        print(f"  {i + 1} = {filename} (expected {expected_score})")

    accuracy = (correct / total) * 100 if total else 0
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"{model} Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("=" * 50)

    labels = [1, 2, 3, 4, 5]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{model} Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Clock Drawing Test Confusion Matrix\n{model}")
    plt.tight_layout()

    plt.savefig(f"clock_confusion_matrix_{model.replace('/', '_')}.png", dpi=300)
    plt.close(fig)

    return {"y_true": y_true, "y_pred": y_pred, "correct": correct, "total": total,
            "accuracy": correct / total if total else 0,
            "circle": sum(c.get("circle", 0) for c in components) / total if total else 0,
            "numbers": sum(c.get("numbers", 0) for c in components) / total if total else 0,
            "hands": sum(c.get("hands", 0) for c in components) / total if total else 0}


def eval_vlm_cube(client, model):
    cases = collect_cases(cube)
    correct = 0
    total = len(cases)

    y_true = []
    y_pred = []
    answers = []

    print(f"Running {model} on {total} cube cases...\n")

    for image_path, expected_score, filename in cases:
        data = describe_images(client, cube_prompt, [image_path])
        predicted_total = score_cube(data)["total"] if data else 0
        answers.append(data or {})

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
                f"Predicted={predicted_total})"
            )
        )

        print(f"[{status}] {model} | {filename}")

    # what this model answered for every field, image by image (columns are the images below)
    fields = list(dict.fromkeys(field for answer in answers for field in answer))
    print(f"\n{model} field answers:")
    print(f"{'field':<30}" + "".join(f"{i + 1:>9}" for i in range(len(cases))))
    for field in fields:
        print(f"{field:<30}" + "".join(
            f"{str(answer.get(field, '-')).strip().lower()[:8]:>9}" for answer in answers))
    for i, (_, expected_score, filename) in enumerate(cases):
        print(f"  {i + 1} = {filename} (expected {expected_score})")

    accuracy = (correct / total) * 100 if total else 0
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"{model} Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("=" * 50)

    labels = [0, 1, 2]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{model} Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Cube Drawing Test Confusion Matrix\n{model}")
    plt.tight_layout()

    plt.savefig(f"cube_confusion_matrix_{model.replace('/', '_')}.png", dpi=300)
    plt.close(fig)

    return {"y_true": y_true, "y_pred": y_pred, "correct": correct, "total": total,
            "accuracy": correct / total if total else 0}


def eval_vlm_infinity(client, model):
    cases = collect_cases(infinity)
    correct = 0
    total = len(cases)

    y_true = []
    y_pred = []
    answers = []

    print(f"Running {model} on {total} infinity cases...\n")

    for image_path, expected_score, filename in cases:
        data = describe_images(client, infinity_prompt, [image_path])
        predicted_total = score_infinity(data)["total"] if data else 0
        answers.append(data or {})

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
                f"Predicted={predicted_total})"
            )
        )

        print(f"[{status}] {model} | {filename}")

    # what this model answered for every field, image by image (columns are the images below)
    fields = list(dict.fromkeys(field for answer in answers for field in answer))
    print(f"\n{model} field answers:")
    print(f"{'field':<30}" + "".join(f"{i + 1:>9}" for i in range(len(cases))))
    for field in fields:
        print(f"{field:<30}" + "".join(
            f"{str(answer.get(field, '-')).strip().lower()[:8]:>9}" for answer in answers))
    for i, (_, expected_score, filename) in enumerate(cases):
        print(f"  {i + 1} = {filename} (expected {expected_score})")

    accuracy = (correct / total) * 100 if total else 0
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"{model} Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("=" * 50)

    labels = [0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{model} Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Infinity Symbol Test Confusion Matrix\n{model}")
    plt.tight_layout()

    plt.savefig(f"infinity_confusion_matrix_{model.replace('/', '_')}.png", dpi=300)
    plt.close(fig)

    return {"y_true": y_true, "y_pred": y_pred, "correct": correct, "total": total,
            "accuracy": correct / total if total else 0}


def main():
    model_res = {}
    for model in models:
        print(f"\n{'=' * 60}\n{model}\n{'=' * 60}")
        try:
            client = make_client(model)
        except Exception as e:
            print(f"  client failed: {e}")
            continue

        model_res[model] = {}
        try:
            print("\n-- clock --")
            # clock answers move run to run, so repeat it and report the spread
            runs = [eval_vlm_clock(client, model) for _ in range(clock_runs)]
            model_res[model]["clock"] = {
                "correct": sum(r["correct"] for r in runs),
                "total": sum(r["total"] for r in runs),
                "accuracy": sum(r["correct"] for r in runs) / sum(r["total"] for r in runs),
                "runs": [r["accuracy"] for r in runs],
                # clock is scored 0-5, so exact-match alone hides how near the misses are
                "within1": sum(sum(abs(t - p) <= 1 for t, p in zip(r["y_true"], r["y_pred"]))
                               for r in runs),
                "mae_runs": [mean_absolute_error(r["y_true"], r["y_pred"]) for r in runs],
                "circle": sum(r["circle"] for r in runs) / len(runs),
                "numbers": sum(r["numbers"] for r in runs) / len(runs),
                "hands": sum(r["hands"] for r in runs) / len(runs),
            }
            print("\n-- cube --")
            model_res[model]["cube"] = eval_vlm_cube(client, model)
            print("\n-- infinity --")
            model_res[model]["infinity"] = eval_vlm_infinity(client, model)
        except Exception as e:
            print(f"  failed: {e}")

    print("\n" + "=" * 60)
    print("FINAL ACCURACY LEADERBOARD")
    print("=" * 60)
    print(f"{'model':<28}" + "".join(f"{t:>12}" for t in tasks) + f"{'overall':>12}")
    for model, per_task in model_res.items():

        correct = sum(r["correct"] / (clock_runs if t == "clock" else 1) for t, r in per_task.items())
        total = sum(r["total"] / (clock_runs if t == "clock" else 1) for t, r in per_task.items())
        cells = "".join(f"{per_task[t]['accuracy'] * 100:>11.1f}%" if t in per_task else f"{'--':>12}"
                        for t in tasks)
        overall = f"{correct / total * 100:>11.1f}%" if total else f"{'--':>12}"
        print(f"{model:<28}{cells}{overall}")

    print("\n" + "=" * 60)
    print(f"CLOCK ACCURACY ACROSS {clock_runs} RUNS")
    print("=" * 60)
    print(f"{'model':<28}" + "".join(f"{'run' + str(i + 1):>6}" for i in range(clock_runs))
          + f"{'mean':>8}{'range':>9}{'+/-1pt':>9}{'MAE':>8}")
    for model, per_task in model_res.items():
        if "clock" not in per_task:
            continue
        per_run = per_task["clock"]["runs"]
        maes = per_task["clock"]["mae_runs"]
        within1 = per_task["clock"]["within1"] / per_task["clock"]["total"] * 100
        print(f"{model:<28}"
              + "".join(f"{a * 100:>5.0f}%" for a in per_run)
              + f"{sum(per_run) / len(per_run) * 100:>7.1f}%"
              + f"{f'{min(per_run) * 100:.0f}-{max(per_run) * 100:.0f}':>9}"
              + f"{within1:>8.1f}%"
              + f"{sum(maes) / len(maes):>8.2f}")

    print("\n" + "=" * 60)
    print(f"CLOCK COMPONENTS -- mean points awarded per image, over {clock_runs} runs")
    print("=" * 60)
    print(f"{'model':<28}{'circle /1':>12}{'numbers /2':>12}{'hands /2':>12}")
    for model, per_task in model_res.items():
        if "clock" not in per_task:
            continue
        clock_stats = per_task["clock"]
        print(f"{model:<28}{clock_stats['circle']:>12.2f}{clock_stats['numbers']:>12.2f}{clock_stats['hands']:>12.2f}")

if __name__ == "__main__":
    main()
