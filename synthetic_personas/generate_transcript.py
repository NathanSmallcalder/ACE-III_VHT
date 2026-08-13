import glob
import json
import random
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from data_loader import get_session_config, resolve_dynamic_answers
from marking.marking import CATEGORY_FLUENCY_BANDS, LETTER_FLUENCY_BANDS

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=1, max_output_tokens=2000)
# Full (non-lite) model for multi-step conditional logic the lite model keeps getting wrong
# (recognition's credited/included/correctness math) -- scoped to just that item.
llm_reasoning = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2, max_output_tokens=8000)
REQUEST_INTERVAL_SECONDS = 0


def get_real_values():
    """Today's actual day/date/month/year/season, the assessment location, and
    the current UK PM / US President -- the real-world values orientation and
    retrograde_memory answers must be checked against, same source data_loader
    gives the real running system."""
    session_config = get_session_config()
    day_of_week, date_window, month, year, season = resolve_dynamic_answers(
        ["DYNAMIC:day_of_week", "DYNAMIC:date", "DYNAMIC:month", "DYNAMIC:year", "DYNAMIC:season"],
        session_config,
    )
    today = date_window[len(date_window) // 2]
    location = session_config["location"]
    return {
        "day_of_week": day_of_week, "date": today, "month": month, "year": year, "season": season,
        "floor": location["number"], "street": location["street"], "town": location["town"],
        "county": location["county"], "country": location["country"],
        "current_uk_pm": session_config["current_uk_pm"],
        "previous_uk_pm": session_config.get("previous_uk_pm", "unknown"),
        "current_us_president": session_config["current_us_president"],
        "previous_us_president": session_config.get("previous_us_president", "unknown"),
    }


def generate_answer(item_prompt, cognitive_status, model=None,
                     response_note="Respond with ONLY the patient's answer text for this item -- no explanation, no JSON, nothing else."):
    """Combine the persona delivery style with one item's instructions, ask the
    model for just that item's answer text, and return it."""
    persona_text = set_persona(cognitive_status)
    prompt = f"""You are role-playing a patient taking the ACE-III cognitive test, with cognitive status: {cognitive_status}.

    ## Delivery Style
    {persona_text}

    ## This Item
    {item_prompt}

    {response_note}"""
    time.sleep(REQUEST_INTERVAL_SECONDS)
    response = (model or llm).invoke([HumanMessage(content=prompt)]).content
    if not isinstance(response, str):
        response = "".join(block.get("text", "") for block in response if isinstance(block, dict))
    return response.strip()


def set_persona(persona):
    if persona == "healthy":
        status =  """ Spoken delivery is fluent, confident, and complete.
                    Zero hesitation markers, stuttering, or filler words."""
        return status
    if persona == "mci":
        status = """Occasional mild hesitation on effortful items (e.g., "um...", "let me think...")
                  Remains on track throughout—never rambles, loses train of thought, or stutters unnaturally."""
        return status
    if persona == "dementia":
        status = """Delivery is slow, effortful, and erratic -- frequent fillers ("uh", "um", "erm", "well",
                  "hold on"). Hesitation MUST occur throughout ALL answers, including CORRECT ones -- at
                  least 2 separate disruption points per item. A single leading filler before an otherwise
                  clean word ("uh... Tuesday") is WRONG, that reads as confident.
                  *Filled pauses*: "um"/"uh"/"erm" between or within phrases.
                  *Whole-word / onset repeats*: repeat a sound, syllable, or whole word before continuing
                  -- "t-t-Tuesday", "the county... the county is...". For a multi-word answer (a name,
                  address, sentence, place name) the repeat can land on ANY of its word
                  boundaries, not just once for variety -- "King... King... bridge", "Cant... Cantur...bury
                  rbury", "seventy - seventy, seventy-nine".
                  *Phrase-level false starts*: an abandoned attempt using a different real word or generic
                  phrase, not a truncated fragment -- "is it... hold on..." -- never a cut-off half-word
                  like "Ki-"; ASR either recognizes the completed word or drops it, never half of one.
                  Never invent a letter, sound, or spelling that isn't one of these four.
                  **Short items** (letters, numbers, shapes): a filled pause around the whole token
                  ("uh... M") or a quick onset repeat ("s-six") -- never letter-by-letter spelling.
                  **Wrong items**: fail via hesitation, trailing off, landing on a different plausible whole
                  value, or a genuine self-correction (state one value, then correct it with "no", "wait",
                  "actually" to a different one, e.g. "key... no wait, ball") -- not a clean confident
                  wrong fact, and not "I don't know" unless told to."""
        return status
    raise ValueError(f"Unknown persona: {persona!r}")

def serial_sevens(attention_serial7, cognitive_status):
    prompt = f"""You are taking the serial sevens test: starting at 100, subtract 7 five times in a
                row. Correct sequence is
                100->93->86->79->72->65. Score {attention_serial7} means the first {attention_serial7} of those are said correctly; the
                remaining steps must each be wrong, pick any plausible arithmetic slip for each
                one (off by a different amount each time is fine and more realistic than a fixed
                offset), as long as no wrong step accidentally equals its own correct value OR
                any other position's correct value in the sequence. Also make sure NO wrong step
                differs from the number stated immediately before it by exactly 7 -- the real
                scorer credits any consecutive pair that's exactly 7 apart even when neither
                number is on the canonical chain, so two wrong numbers that happen to land 7
                apart will silently score as a correct step. None are self-corrected. 
                if {attention_serial7} is 0, that first step must be wrong
                too. If {attention_serial7} is 5, use the exact canonical numbers 93, 86, 79, 72,
                65 precisely -- do not drift off them on the later steps."""
    return generate_answer(prompt, cognitive_status, model=llm_reasoning)


def orientation_date(orientation_date, cognitive_status, real_values):
    prompt = f"""**orientation_date** (0-5): 5 items -- day of the week, date, month, year, season. Exactly
                {orientation_date} of these 5 must be correct -- you choose which; the rest must be
                wrong (a wrong date must be more than 2 days from today; other wrong fields just
                must not equal the true value).
                Real values: day_of_week={real_values['day_of_week']}, date={real_values['date']},
                month={real_values['month']}, year={real_values['year']}, season={real_values['season']}.
                Use these exact values for every field marked correct.
                Return a JSON array of exactly 5 objects, one per field, in this order --
                day of the week, date, month, year, season -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this field's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that field was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def orientation_location(orientation_location, cognitive_status, real_values):
    prompt = f"""**orientation_location** (0-5): 5 items -- floor, street, town, county, country.
                Exactly {orientation_location} of these 5 must be correct -- you choose which; the
                rest must be wrong.
                Real values: floor={real_values['floor']}, street={real_values['street']},
                town={real_values['town']}, county={real_values['county']}, country={real_values['country']}.
                Use these exact values for every field marked correct.
                Return a JSON array of exactly 5 objects, one per field, in this order --
                floor, street, town, county, country -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this field's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that field was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def registration(registration, cognitive_status):
    prompt = f"""**registration** (0-3): one item, "Repeat lemon, key, ball" -- Score_Cap 3,
                Ground_Truth = the raw score. Say exactly {registration} of the 3 words -- you
                choose which; omit the rest naturally (don't say "I don't know" for the omitted
                ones), just don't mention them or say words that sound or spelt close."""
    return generate_answer(prompt, cognitive_status)


def recall_3words(recall_3words, cognitive_status):
    prompt = f"""**recall_3words** (0-3): one item, Score_Cap 3, recall {recall_3words} of lemon/key/ball (pick
                which {recall_3words}, omit the rest don't say "I don't know" for the omitted ones, just don't
                mention them or say words that sound or spelt close."""
    return generate_answer(prompt, cognitive_status)


def anterograde_memory(anterograde_memory, cognitive_status):
    prompt = f"""**anterograde_memory** (0-7): one item, Score_Cap 7. Tokens in order: Harry, Barnes,
                73, Orchard, Close, Kingsbridge, Devon. Say the first {anterograde_memory} tokens
                correctly, in order, then trail off / say you can't remember the rest -- this one
                stays sequential, not free choice (a real recall attempt runs through and fades,
                it doesn't skip around). Before answering, silently count the tokens you're about
                to say and confirm the count matches {anterograde_memory} exactly -- a common slip
                is stating one extra or one too few before trailing off. This check is internal only
                -- the reply must be just the spoken words, no numbering, parentheticals, or
                verification text of any kind."""
    return generate_answer(prompt, cognitive_status)


def retrograde_memory(retrograde_memory, cognitive_status, real_values):
    prompt = f"""**retrograde_memory** (0-4): 4 items, Score_Cap 1 each: UK PM (real answer:
                {real_values['current_uk_pm']}), first female UK PM (real answer: Margaret Thatcher),
                US President (real answer: {real_values['current_us_president']}), JFK-assassination
                question (real answer: John F. Kennedy -- answer with the PRESIDENT'S name Exactly
                {retrograde_memory} of these 4 must be correct -- you choose which. Wrong UK PM /
                US President should avoid naming the *previous* holder of that office, instead a random name /
                hat sounds simular and has the potential of fuzzy matching with the target name.
                Real values for wrong-answer avoidance: previous UK PM={real_values['previous_uk_pm']},
                previous US President={real_values['previous_us_president']}.
                Return a JSON array of exactly 4 objects, one per field, in this order --
                UK PM, first female UK PM, US President, JFK-assassination question -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this field's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that field was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def _band_target_count(marks, bands):
    """Middle of the word-count range that earns `marks` per the real
    scorer's own band table (marking.py), so the persona is told an exact
    count instead of a table to infer one from -- avoids the model
    conflating the marks value itself with a word count."""
    lo, hi, _ = next(b for b in bands if b[2] == marks)
    return lo + 3 if hi == float("inf") else (lo + hi) // 2


def verbal_fluency(verbal_fluency, cognitive_status):
    target_count = _band_target_count(verbal_fluency, LETTER_FLUENCY_BANDS)
    prompt = f"""**verbal_fluency**: Say exactly {target_count} distinct, real words that start
                with the letter P, comma separated, no repeats -- this exact count is what the real
                scorer needs to land on {verbal_fluency} marks; do not use {verbal_fluency} itself as
                the word count. If {target_count} is 0, say you can't think of any."""
    return generate_answer(prompt, cognitive_status)


def semantic_fluency(semantic_fluency, cognitive_status):
    target_count = _band_target_count(semantic_fluency, CATEGORY_FLUENCY_BANDS)
    prompt = f"""**semantic_fluency**: Say exactly {target_count} distinct, real animals or mythical creatures 
                 this exact count is what the real scorer needs to land on
                {semantic_fluency} marks; do not use {semantic_fluency} itself as the word count. 
                Repeats do not count towards score neither does naming an animal thats a category of animals e.g fish -> trout, salmon would score 2 for both not 3 for fish
                same applies to gender varients of deer -> fown is a mark"""
    return generate_answer(prompt, cognitive_status)


def comprehension(comprehension, cognitive_status):
    # Deterministic yes/no completion record, not spoken text -- no LLM call needed.
    fields = ["paper_placed_on_pencil", "pencil_lifted_without_paper", "pencil_lifted_after_touching_paper"]
    return json.dumps({field: ("yes" if i < comprehension else "no") for i, field in enumerate(fields)})


def writing(writing, cognitive_status):
    prompt = f"""**writing** (0-2): one item, Score_Cap 2. The answer is the literal WRITTEN text the
                patient put on paper, not a transcript of them speaking it -- do NOT use any spoken
                delivery markers here (no "um"/"uh"/"erm", no "...", no word/onset repeats like
                "t-t-Tuesday", no self-correction like "no wait"). A patient does not stutter onto
                a page. Express difficulty the way handwriting actually would instead: a genuine
                spelling or grammar slip, a sentence that trails into a fragment, or simply stopping
                after fewer sentences than asked. This participant's target is {writing}: 2 -> two full
                sentences (subject+verb, no errors) about your day. 1 -> exactly one full sentence
                (subject+verb, no grammar or spelling errors -- a real full stop or "and", not a
                run-on/comma-splice joining two clauses), then stop.
                0 -> a fragment, no verb/subject structure."""
    return generate_answer(prompt, cognitive_status)


def verbal_repetition(verbal_repetition, cognitive_status):
    prompt = f"""**verbal_repetition**: Repeat these 4 words, one at a time, after being asked:
                caterpillar, eccentricity, unintelligible, statistician.
                Scoring: 3 correct words = 1 mark, 4 correct words = 2 marks, fewer than 3 correct = 0 marks.
                This participant scores {verbal_repetition} marks. Drop statistician first if any
                must be wrong, then unintelligible, then eccentricity. A wrong word is a genuine
                broken fragment of its start or end, never the complete correct word.
                Return a JSON array of exactly 4 objects, one per word, in this order --
                caterpillar, eccentricity, unintelligible, statistician -- each shaped like:
                {{"Ground_Truth": 0, "Answered": "<this word's own answer text, in the patient's voice>"}}
                For the first three words (caterpillar, eccentricity, unintelligible), Ground_Truth
                is 1 if that word was answered correctly, 0 if wrong. For statistician specifically,
                Ground_Truth is NOT just that word's own correctness -- the real scorer's cap for
                this whole 4-word item sits entirely on this last slot, so its Ground_Truth must be
                the TOTAL marks the full set of 4 words earns per the scoring table above (0, 1, or
                2), matching {verbal_repetition} exactly."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def verbal_repetition_2(verbal_repetition_2, cognitive_status):
    prompt = f"""**verbal_repetition_2** (0-2): 2 items, Score_Cap 1 each -- "All that glitters is not
                gold", "A stitch in time saves nine". This participant's score is {verbal_repetition_2}.
                Score >=1 -> first correct; score 2 -> both
                correct. Wrong = a genuine attempt at the REAL phrase that breaks down under
                dementia-style delivery, not a swapped-in unrelated sentence.
                It should land close to what a fuzzy matcher could
                plausibly still accept -- genuinely hard to call, not an obviously-wrong
                clean substitute. Just don't let it complete the real phrase intact.
                Return a JSON array of exactly 2 objects, one per phrase, in this order --
                "All that glitters is not gold", "A stitch in time saves nine" -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this phrase's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that phrase was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def naming(naming, cognitive_status):
    prompt = f"""**naming** (0-12): one item, Score_Cap 12. 12 pictures: spoon, book, kangaroo or wallaby,
                penguin, anchor, camel, harp, rhinoceros or rhino, barrel,tub or keg, crown, crocodile or alligator, accordion or squeeze box.
                Exactly {naming} of these 12 must be named correctly -- you choose which; the rest
                get a wrong guess (broken/effortful per delivery style, not silence). A wrong guess
                must NOT be any of the accepted alternate names for that same picture (kangaroo/
                wallaby, camel/dromedary, rhinoceros/rhino, barrel/keg/tub, crocodile/alligator,
                accordion/piano accordion/squeeze box are ALL scored as correct by the real
                scorer) -- guess a plausible but genuinely different object instead.
                **For a WRONG item specifically, go straight to the wrong guess -- do NOT
                self-correct from the right name into a wrong one (e.g. "anchor... no wait...
                hook" is forbidden even though the delivery-style guide allows self-correction
                elsewhere). The real scorer reads the whole answer as one list and does not
                understand retraction, so saying the correct name at all -- even before
                "correcting" away from it -- gets it credited. A wrong item's hesitation must
                never contain the correct name as a word.
                Before answering, silently count how many of the 12 you're about to name correctly
                and confirm that count is exactly {naming} -- a common slip is naming one extra or
                one too few. This check is internal only, the reply must be just the spoken answer,
                no verification text."""
    return generate_answer(prompt, cognitive_status, model=llm_reasoning)


def comprehension_2(comprehension_2, cognitive_status):
    prompt = f"""**comprehension_2** (0-4): 4 items, Score_Cap 1 each: monarchy->crown,
                marsupial->kangaroo, antarctic->penguin, nautical->anchor. Exactly
                {comprehension_2} of these 4 must be correct -- you choose which. For a wrong one,
                pick any item from the naming picture set (spoon, book, kangaroo, penguin, anchor,
                camel, harp, rhinoceros, barrel, crown, crocodile, accordion) OTHER than the
                correct answer for that item -- a semantically-near miss.
                Return a JSON array of exactly 4 objects, one per item, in this order --
                monarchy, marsupial, antarctic, nautical -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this item's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that item was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def reading(reading, cognitive_status):
    prompt = f"""**reading** (0-1): one item, Score_Cap 1, all-or-nothing on "sew, pint, soot, dough,
                height". This participant's score is {reading}. The {reading} is achieved by listing all
                5 correctly, this acheivable photenically too. If 0 missread one or more of the words. """
    return generate_answer(prompt, cognitive_status)


def infinity_copying(infinity_copying, cognitive_status):
    # Deterministic file path, no free text to generate -- no LLM call needed.
    # Picks randomly among every reference drawing available for this score,
    # instead of always the same one.
    folder = f"test/InfinitySymbol/{infinity_copying}"
    return random.choice(glob.glob(f"{folder}/*.png"))


def cube_copying(cube_copying, cognitive_status):
    # Deterministic file path, no free text to generate -- no LLM call needed.
    if cube_copying == 0:
        return "0"
    folder = f"test/cube/{cube_copying}"
    return random.choice(glob.glob(f"{folder}/*.png"))


def clock_copying(clock_copying, cognitive_status):
    # Deterministic file path, no free text to generate -- no LLM call needed.
    folder = f"test/Clocks/{clock_copying}"
    return random.choice(glob.glob(f"{folder}/*.png"))


def dot_counting(dot_counting, cognitive_status):
    prompt = f"""**dot_counting** (0-4): 4 items, Score_Cap 1 each. True counts: 10, 8, 9, 7. Exactly
                {dot_counting} of these 4 must be correct -- you choose which. Correct ones say the
                true count directly (not counted aloud one-by-one); wrong ones say any other
                number, stated directly.
                Return a JSON array of exactly 4 objects, one per array, in this order --
                first, second, third, fourth -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this array's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that array was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def fragmented_letters(fragmented_letters, cognitive_status):
    prompt = f"""**fragmented_letters** (0-4): 4 items, Score_Cap 1 each. True letters: M, A, T, K.
                Exactly {fragmented_letters} of these 4 must be correct -- you choose which; the
                rest say a different single letter.
                Return a JSON array of exactly 4 objects, one per letter, in this order --
                first, second, third, fourth -- each shaped like:
                {{"Ground_Truth": 1, "Answered": "<this letter's own answer text, in the patient's voice>"}}
                Ground_Truth is 1 if that letter was answered correctly, 0 if wrong."""
    return generate_answer(
        prompt, cognitive_status,
        response_note="Respond with ONLY the JSON array described above -- no markdown fences, no other text.",
    )


def delayed_recall(delayed_recall, cognitive_status):
    prompt = f"""**delayed_recall** (0-7): one item, Score_Cap 7. Tokens in order: Harry, Barnes,
                73, Orchard, Close, Kingsbridge, Devon -- being recalled again, unprompted, after the
                interference tasks Say the first {delayed_recall} tokens correctly, in order, then trail off / say you
                can't remember the rest -- stays sequential, not free choice. Before answering,
                silently count the tokens you're about to say and confirm the count matches
                {delayed_recall} exactly -- a common slip is stating one extra or one too few before
                trailing off. the reply must be just the spoken words,
                no numbering, parentheticals, or verification text of any kind."""
    return generate_answer(prompt, cognitive_status)


def recognition(recognition, delayed_recall, cognitive_status):
    prompt = f"""Input Parameters:
                - delayed_recall: {delayed_recall}
                - recognition: {recognition}

                Field Thresholds:
                - name (>= 2) | number (>= 3) | street (>= 5) | town (>= 6) | county (>= 7)

                Answer Options (Real Value):
                - name: Jerry Barnes, Harry Barnes, or Harry Bradford (Real: Harry Barnes)
                - number: 37, 73, or 76 (Real: 73)
                - street: Orchard Place, Oak Close, or Orchard Close (Real: Orchard Close)
                - town: Oakhampton, Kingsbridge, or Dartington (Real: Kingsbridge)
                - county: Devon, Dorset, or Somerset (Real: Devon)
                """
    return generate_answer(prompt, cognitive_status, model=llm_reasoning)

import copy
import os

import pandas as pd

csv_path = "synthetic_personas/synthetic_ace3.csv"
shell = "synthetic_personas/shell.json"
output = "synthetic_transcripts"


def _fill_one(entry, target, status, fn, *extra_args):
    entry["Ground_Truth"] = target
    entry["Answered"] = fn(target, status, *extra_args)


def _fill_group(entries, target, status, fn, *extra_args):
    """Multi-entry item (e.g. orientation_date's 5 sub-fields): one LLM call
    returns a JSON array with each sub-field's own Ground_Truth and Answered."""
    raw = fn(target, status, *extra_args).strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    parsed = json.loads(raw)
    for entry, item in zip(entries, parsed):
        entry["Ground_Truth"] = item["Ground_Truth"]
        entry["Answered"] = item["Answered"]


rec_fields = ["name", "number", "street", "town", "county"]
rec_fresholds = {"name": 2, "number": 3, "street": 5, "town": 6, "county": 7}
rec_questions = {
    "name": "Was the name Jerry Barnes, Harry Barnes, or Harry Bradford?",
    "number": "Was the number 37, 73, or 76?",
    "street": "Was the street Orchard Place, Oak Close, or Orchard Close?",
    "town": "Was the town Oakhampton, Kingsbridge, or Dartington?",
    "county": "Was the county Devon, Dorset, or Somerset?",
}


def _build_recognition_entries(recognition_score, delayed_recall, status):
    credited = [f for f in rec_fields if delayed_recall >= rec_fresholds[f]]
    included = [f for f in rec_fields if f not in credited]
    correct_among_included = max(0, recognition_score - len(credited))
    correct_fields = set(included[len(included) - correct_among_included:]) if correct_among_included else set()

    answer = recognition(recognition_score, delayed_recall, status)
    return [
        {
            "Question": f"Recognition: {rec_questions[field]}",
            "Score_Cap": 1,
            "Ground_Truth": 1 if field in correct_fields else 0,
            "Answered": answer,
        }
        for field in included
    ]


def generate_transcript(row, real_values, shell):
    transcript = copy.deepcopy(shell)
    transcript["participant_id"] = row["participant_id"]
    transcript["cognitive_status"] = row["cognitive_status"]
    status = row["cognitive_status"]

    _fill_group(transcript["Attention"][0:5], int(row["orientation_date"]), status, orientation_date, real_values)
    _fill_group(transcript["Attention"][5:10], int(row["orientation_location"]), status, orientation_location, real_values)
    _fill_one(transcript["Attention"][10], int(row["registration"]), status, registration)
    _fill_one(transcript["Attention"][11], int(row["attention_serial7"]), status, serial_sevens)

    _fill_one(transcript["Memory"][0], int(row["recall_3words"]), status, recall_3words)
    _fill_one(transcript["Memory"][1], int(row["anterograde_memory"]), status, anterograde_memory)
    _fill_group(transcript["Memory"][2:6], int(row["retrograde_memory"]), status, retrograde_memory, real_values)

    _fill_one(transcript["Fluency"][0], int(row["verbal_fluency"]), status, verbal_fluency)
    _fill_one(transcript["Fluency"][1], int(row["semantic_fluency"]), status, semantic_fluency)

    _fill_one(transcript["Language"][0], int(row["comprehension"]), status, comprehension)
    _fill_one(transcript["Language"][1], int(row["writing"]), status, writing)
    _fill_group(transcript["Language"][2:6], int(row["verbal_repetition"]), status, verbal_repetition)
    _fill_group(transcript["Language"][6:8], int(row["verbal_repetition_2"]), status, verbal_repetition_2)
    _fill_one(transcript["Language"][8], int(row["naming"]), status, naming)
    _fill_group(transcript["Language"][9:13], int(row["comprehension_2"]), status, comprehension_2)
    _fill_one(transcript["Language"][13], int(row["reading"]), status, reading)

    _fill_one(transcript["Visuospatial"][0], int(row["infinity_copying"]), status, infinity_copying)
    _fill_one(transcript["Visuospatial"][1], int(row["cube_copying"]), status, cube_copying)
    _fill_one(transcript["Visuospatial"][2], int(row["clock_copying"]), status, clock_copying)
    _fill_group(transcript["Visuospatial"][3:7], int(row["dot_counting"]), status, dot_counting)
    _fill_group(transcript["Visuospatial"][7:11], int(row["fragmented_letters"]), status, fragmented_letters)
    _fill_one(transcript["Visuospatial"][11], int(row["delayed_recall"]), status, delayed_recall)

    recognition_entries = _build_recognition_entries(
        int(row["recognition"]), int(row["delayed_recall"]), status
    )
    transcript["Visuospatial"].extend(recognition_entries)

    return transcript


def main():
    df = pd.read_csv(csv_path)
    with open(shell) as f:
        shell = json.load(f)
    real_values = get_real_values()
    os.makedirs(out_path, exist_ok=True)

    for _, row in df.iterrows():
        transcript = generate_transcript(row, real_values, shell)
        out_path = os.path.join(out_path, f"{row['participant_id']}_{row['cognitive_status']}.json")
        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
