import json
import os

import pandas as pd

import synthetic_personas.persona_prompt as ps
from data_loader import get_session_config, _name_and_surname

CSV_PATH = "synthetic_personas/synthetic_ace3.csv"
OUTPUT_DIR = "synthetic_transcripts"
SAMPLE_SIZE = 1


def speak(instruction_text):
    """Send an instruction through the LLM and return the persona's spoken reply."""
    return ps._ask(ps.PREAMBLE + "\n" + instruction_text)

def item(question_text, score_cap, ground_truth, answered):
    return {"Question": question_text, "Score_Cap": score_cap, "Ground_Truth": ground_truth, "Answered": answered}

TIME_FIELD_QUESTIONS = {
    "time": "Orientation to Time: What day of the week is it?",
    "date": "Orientation to Time: What is today's date?",
    "month": "Orientation to Time: What month is it?",
    "year": "Orientation to Time: What year is it?",
    "season": "Orientation to Time: What season is it?",
}
PLACE_FIELD_QUESTIONS = {
    "floor": "Orientation to Place: Which number or floor is this?",
    "street": "Orientation to Place: Which street is this?",
    "town": "Orientation to Place: Which town are we in?",
    "county": "Orientation to Place: Which county are we in?",
    "country": "Orientation to Place: Which country are we in?",
}
def attention_items(row, session_config):
    items = []

    time_score = int(row["orientation_date"])
    time_texts = ps.time_field_instructions(time_score, ps.resolve_time(session_config))
    time_wrong = set(ps.TIME_DROP_ORDER[:5 - time_score])
    for field, text in time_texts.items():
        ground_truth = 0 if field in time_wrong else 1
        items.append(item(TIME_FIELD_QUESTIONS[field], 1, ground_truth, speak(text)))

    location_score = int(row["orientation_location"])
    location_texts = ps.location_field_instructions(location_score, ps.resolve_place(session_config))
    location_wrong = set(ps.LOCATION_DROP_ORDER[:5 - location_score])
    for field, text in location_texts.items():
        ground_truth = 0 if field in location_wrong else 1
        items.append(item(PLACE_FIELD_QUESTIONS[field], 1, ground_truth, speak(text)))

    score = int(row["registration"])
    items.append(item("Registration: Repeat the three words lemon, key, and ball.",
                       3, score, speak(ps.registration_instruction(score))))

    score = int(row["attention_serial7"])
    items.append(item("Serial 7s: Subtract 7 from 100 and keep subtracting 7.",
                       5, score, speak(ps.serial_sevens_instruction(score))))

    return items


def memory_items(row, session_config):
    items = []

    score = int(row["recall_3words"])
    items.append(item("Three-word recall: Which three words did I ask you to remember?",
                       3, score, speak(ps.delayed_recall(score))))

    score = int(row["anterograde_memory"])
    items.append(item("Name and address learning: Harry Barnes, 73 Orchard Close, Kingsbridge, Devon.",
                       7, score, speak(ps.anterograde_memory(score))))

    retrograde_questions = [
        "Retrograde memory: Name of the current UK Prime Minister.",
        "Retrograde memory: Name of the first female UK Prime Minister.",
        "Retrograde memory: Name of the current President of the United States.",
        "Retrograde memory: Name of the US President assassinated in the 1960s.",
    ]
    retro_score = int(row["retrograde_memory"])
    retro_texts = ps.retrograde_memory(retro_score, session_config)
    retro_wrong = set(ps.RETRO_DROP_ORDER[:4 - retro_score])
    for question_text, (field, text) in zip(retrograde_questions, retro_texts.items()):
        ground_truth = 0 if field in retro_wrong else 1
        items.append(item(question_text, 1, ground_truth, speak(text)))

    return items


def fluency_items(row):
    items = []

    score = int(row["verbal_fluency"])
    items.append(item("Letter fluency: Generate words beginning with the letter P in one minute.",
                       7, score, speak(ps.verbal_fluency(score))))

    score = int(row["semantic_fluency"])
    items.append(item("Category fluency: Name as many animals as possible in one minute.",
                       7, score, speak(ps.animal_fluency(score))))

    return items


def language_items(row):
    items = []

    score = int(row["comprehension"])
    items.append(item("Comprehension: Follow three-stage commands with pencil and paper.",
                       3, score, speak(ps.comprehension(score))))

    score = int(row["writing"])
    items.append(item("Writing: Write two complete sentences.", 2, score, speak(ps.writing(score))))

    score = int(row["verbal_repetition"])
    word_correct_count = {2: 4, 1: 3, 0: 0}[score]
    word_texts = list(ps.word_repetition(word_correct_count).items())
    # Real graph scores these 4 turns as one combined question (score_cap 2, via
    # sub_score_bands), not 4 independent 1-point words -- put the whole cap/ground
    # truth on the last turn so the summed total still matches the real one.
    for i, (word, text) in enumerate(word_texts):
        cap, ground_truth = (2, score) if i == len(word_texts) - 1 else (0, 0)
        items.append(item(f"Word repetition: Repeat after me - {word}.", cap, ground_truth, speak(text)))

    score = int(row["verbal_repetition_2"])
    sentence_1_score = 1 if score >= 1 else 0
    sentence_2_score = 1 if score == 2 else 0
    sentence_1 = "All that glitters is not gold"
    sentence_2 = "A stitch in time saves nine"
    items.append(item(f"Sentence repetition: Repeat the phrase - {sentence_1}.",
                       1, sentence_1_score, speak(ps.sentence_repetition(sentence_1_score, sentence_1))))
    items.append(item(f"Sentence repetition: Repeat the phrase - {sentence_2}.",
                       1, sentence_2_score, speak(ps.sentence_repetition(sentence_2_score, sentence_2))))

    naming_score = int(row["naming"])
    naming_text = " ".join(ps.naming(naming_score).values())
    items.append(item("Naming: Name the 12 pictures shown on screen.", 12, naming_score, speak(naming_text)))

    comprehension_questions = [
        "Comprehension: Which picture is associated with the monarchy?",
        "Comprehension: Which picture is the marsupial?",
        "Comprehension: Which picture is found in the Antarctic?",
        "Comprehension: Which picture has a nautical connection?",
    ]
    score = int(row["comprehension_2"])
    for i, (question_text, answer) in enumerate(zip(comprehension_questions, ps.COMPREHENSION_PICTURES.values())):
        sub_score = 1 if i < score else 0
        items.append(item(question_text, 1, sub_score, speak(ps.picture_comprehension(sub_score, answer))))

    score = int(row["reading"])
    items.append(item("Reading: Read the following words aloud - sew, pint, soot, dough, height.",
                       1, score, speak(ps.reading(score))))

    return items


def visuospatial_items(row):
    items = []

    score = int(row["infinity_copying"])
    items.append(item("Infinity Diagram: Copy the infinity diagram.", 1, score, ps.infinity_scorer(score)))

    score = int(row["cube_copying"])
    items.append(item("Wire Cube: Copy the wire cube.", 2, score, ps.cube_scorer(score)))

    score = int(row["clock_copying"])
    items.append(item("Clock: Draw a clock face showing ten past five.", 5, score, ps.clock_scorer(score)))

    dot_questions = [
        "Dot counting: Count the dots in the first array.",
        "Dot counting: Count the dots in the second array.",
        "Dot counting: Count the dots in the third array.",
        "Dot counting: Count the dots in the fourth array.",
    ]
    score = int(row["dot_counting"])
    for i, (question_text, true_count) in enumerate(zip(dot_questions, ps.DOT_ANSWERS)):
        sub_score = 1 if i < score else 0
        items.append(item(question_text, 1, sub_score, speak(ps.dot_counting(sub_score, true_count))))

    score = int(row["fragmented_letters"])
    for i, true_letter in enumerate(ps.FRAGMENTED_LETTERS_ANSWERS):
        sub_score = 1 if i < score else 0
        items.append(item("Fragmented letters: Identify the fragmented letter.",
                           1, sub_score, speak(ps.fragmented_letters(sub_score, true_letter))))

    recall_score = int(row["delayed_recall"])
    items.append(item("Delayed recall: Recall the name and address from earlier.",
                       7, recall_score, speak(ps.recall_memory(recall_score))))


    # Mimicing the graph recall questions when a user can recall one element.
    RECALL_GROUP_END = {"name": 2, "number": 3, "street": 5, "town": 6, "county": 7}
    wrong_left = 5 - int(row["recognition"])
    for key in ps.RECOGNITION_DROP_ORDER:
        if recall_score >= RECALL_GROUP_END[key]:
            # Not actually asked -- graph.py auto-credits this element since it was
            # already recalled correctly in delayed recall (see _recognition_recalled).
            # Deliberately not added to `items`: this list doubles as replay_pipeline's
            # literal audio-answer feed, and an entry here with no real spoken answer
            # would misalign every real question asked after it.
            continue
        if wrong_left > 0:
            text = (f"You are asked whether the {key} was {ps.RECOGNITION_OPTIONS[key]}. "
                    f"Say anything but '{ps.RECOGNITION_OPTIONS[key]}'.")
            ground_truth = 0
            wrong_left -= 1
        else:
            text = f"You correctly say '{ps.RECOGNITION_ANSWERS[key]}'."
            ground_truth = 1
        items.append(item(f"Recognition: {ps.RECOGNITION_QUESTIONS[key]}", 1, ground_truth, speak(text)))

    return items


def run_persona(row, session_config):
    return {
        "Attention": attention_items(row, session_config),
        "Memory": memory_items(row, session_config),
        "Fluency": fluency_items(row),
        "Language": language_items(row),
        "Visuospatial": visuospatial_items(row),
    }

def main():
    df = pd.read_csv(CSV_PATH)
    session_config = get_session_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _, row in df.head(SAMPLE_SIZE).iterrows():
        participant_id = row["participant_id"]
        cognitive_status = row["cognitive_status"]
        print(f"Running {participant_id} ({cognitive_status})...")

        result = {
            "participant_id": participant_id,
            "cognitive_status": cognitive_status,
            **run_persona(row, session_config),
            
        }

        out_path = os.path.join(OUTPUT_DIR, f"{participant_id}_{cognitive_status}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
