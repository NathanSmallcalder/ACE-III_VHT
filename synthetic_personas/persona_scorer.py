


from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
from data_loader import resolve_dynamic_answers
from marking.marking import LETTER_FLUENCY_BANDS, CATEGORY_FLUENCY_BANDS

load_dotenv()

llm_warm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.05, max_output_tokens=2000)

def _ask(prompt):
    time.sleep(15)
    result = llm_warm.invoke([HumanMessage(content=prompt)])
    content = result.content
    if isinstance(content, str):
        return content.strip()
    return "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()

PREAMBLE = """You are role-playing someone taking a spoken test.
Follow each item's recall instruction exactly, one item at a time.
- If told you get something wrong, you MUST get it wrong -- but still give a
  plausible spoken answer for that item, within the subject area Never respond with a 
  refusal like "I don't know" unless that item's instruction explicitly
  tells you to say that.i
- If told you get something right, you MUST say the correct answer given.
- When several items are listed, respond to every single one of them, in
  order, with its own short phrase -- never collapse the whole response
  into one generic answer covering only part of the list.
- Never self-correct; never give a correct answer the instruction excludes.
- Speak naturally (pauses, "um"); output only what you say aloud, you can surround your awnsers with text like etc. "ok, well thats" or "uh, well...".
- Do not think step by step and do not show any reasoning — respond immediately with only the spoken answer, nothing else.
= The reply you give must ALWAYS be in context to the question, even if your asked to do it wrong.
"""
 
 
PROMPT_ATTENTION = """
What day of the week is it? 
> {time}
 
what is today's date?
>{date}

what month is it?
>{month}

what year is it?
>{year}

what season is it?
>{season}

What is the floor number? 
> {floor}
 
What is the street? 
> {street}

What is the town?
> {town}

What is the county?
> {county}

what is the country?
> {country}

Registration task — repeat these words back:
> {registration}
 
Subtract 7 from 100, and keep subtracting 7:
> {sevens}
"""


""" ATTENTION SECTION"""

def resolve_place(session_config):
    """Real orientation-to-place answers, straight from session_config's location block."""
    location = session_config["location"]
    return {
        "floor":   location["number"],
        "street":  location["street"],
        "town":    location["town"],
        "county":  location["county"],
        "country": location["country"],
    }

def resolve_time(session_config):
    """Real orientation-to-time answers for today, via the same DYNAMIC resolver the live pipeline uses."""
    day_of_week, date_window, month, year, season = resolve_dynamic_answers(
        ["DYNAMIC:day_of_week", "DYNAMIC:date", "DYNAMIC:month", "DYNAMIC:year", "DYNAMIC:season"],
        session_config,
    )
    today = date_window[len(date_window) // 2]  # middle of the +/-2 day tolerance window = today
    return {"time": day_of_week, "date": today, "month": month, "year": year, "season": season}

TIME_DROP_ORDER = ["date", "year", "time", "month", "season"]

def time_field_instructions(score, true):
    """Per-item recall instructions for orientation-to-time, one per PROMPT_ATTENTION placeholder.
    The "date" field's real answer allows a +/-2 day tolerance window (see
    resolve_dynamic_answers), so a vague wrong guess close to today can still
    score as correct -- the wrong instruction for that field spells out the
    2-day constraint instead of just "isn't {value}"."""
    assert 0 <= score <= 5
    wrong = set(TIME_DROP_ORDER[:5 - score])
    instructions = {}
    for field, value in true.items():
        if field not in wrong:
            instructions[field] = f"You correctly say {value}."
        elif field == "date":
            instructions[field] = f"You say a date that is more than 2 days away from {value}."
        else:
            instructions[field] = f"You say a value that isn't {value}."
    return instructions


LOCATION_DROP_ORDER = ["county", "floor", "street", "town", "country"]

def location_field_instructions(score, true):
    """Per-item recall instructions for orientation-to-place, one per PROMPT_ATTENTION placeholder."""
    assert 0 <= score <= 5
    wrong = set(LOCATION_DROP_ORDER[:5 - score])
    return {
        field: (f"You say a location that isn't {value}."
                 if field in wrong else f"You correctly say {value}.")
        for field, value in true.items()
    }

SERIAL_SEVENS_CORRECT = [93, 86, 79, 72, 65]

def serial_sevens_instruction(score):
    """Word instruction version, if going via the LLM + preamble path."""
    assert 0 <= score <= 5
    correct_steps = SERIAL_SEVENS_CORRECT[:score]
    if score == 5:
        sequence = ", ".join(str(n) for n in correct_steps)
        return f"Say exactly these five numbers, in this exact order, including the first one, then stop: {sequence}."

    wrong_count = 5 - score
    wrong_start = (correct_steps[-1] if correct_steps else 100) - 3
    wrong_steps = [wrong_start - 3 * i for i in range(wrong_count)]
    sequence = ", ".join(str(n) for n in correct_steps + wrong_steps)
    return (f"Say exactly these {5} numbers, in this exact order, including the very first one, "
            f"then stop -- the first {score} are correct subtractions of 7, the rest are wrong "
            f"and must NOT be corrected: {sequence}.")

REGISTRATION_WORDS = ["lemon", "key", "ball"]
REGISTRATION_DROP_ORDER = list(reversed(REGISTRATION_WORDS))

def registration_instruction(score=3):
    """Registration — usually preserved; full marks repeats all items back correctly, first try."""
    assert 0 <= score <= 3
    if score == 3:
        return "You correctly repeat all three words back: lemon, key, ball."
    wrong = set(REGISTRATION_DROP_ORDER[:3 - score])
    correct_words = [w for w in REGISTRATION_WORDS if w not in wrong]
    return f"You repeat back only {score} of the words correctly ({', '.join(correct_words)}) and get the rest wrong."


def build_attention_prompt(time_score, sevens_score, session_config, reg_score=3):
    """Fill the template from the per-item scores -> one attention prompt."""
    return PREAMBLE + "\n" + PROMPT_ATTENTION.format(
        **time_field_instructions(time_score, resolve_time(session_config)),
        **resolve_place(session_config),
        registration=registration_instruction(reg_score),
        sevens=serial_sevens_instruction(sevens_score),
    )

def run_attention_prompt(time_score, sevens_score, session_config, reg_score=3):
    """Send the built attention prompt to the local LLM and return the persona's spoken reply."""
    prompt = build_attention_prompt(time_score, sevens_score, session_config, reg_score)
    return _ask(prompt)

# ==== Memory ===
def delayed_recall(recall_score):
    if recall_score == 3:
        return "you say lemon, key and ball"
    if recall_score == 2:
        return "you recall only 2 of the items from lemon,key and ball, choose two to recall, do not say the others"
    if recall_score == 1:
        return "you recall only 1 of the three items lemon key and ball choose one to recall, do not say the others"
    if recall_score == 0:
        return "you recall 0 of the three items"

NAME_ADDRESS_TOKENS = ["Harry", "Barnes", "73", "Orchard", "Close", "Kingsbridge", "Devon"]

def anterograde_memory(score):
    """Learning-trial recall of the name & address (Harry Barnes, 73 Orchard Close,
    Kingsbridge, Devon), scored out of 7."""
    assert 0 <= score <= 7
    if score == 7:
        return "You correctly repeat the full name and address: Harry Barnes, 73 Orchard Close, Kingsbridge, Devon."
    if score == 0:
        return ("You are trying to recall a name and address: Harry Barnes, 73 Orchard Close, "
                "Kingsbridge, Devon. Say you cannot remember any of it at all.")
    correct_tokens = NAME_ADDRESS_TOKENS[:score]
    return (f"You are trying to recall a name and address: Harry Barnes, 73 Orchard Close, "
            f"Kingsbridge, Devon. Say {', '.join(correct_tokens)} out loud, then say you cannot remember the rest.")

RETRO_DROP_ORDER = ["assassinated_president", "female_pm", "us_president", "uk_prime_minister"]
def retrograde_memory(score, session_config):
    """Per-item recall instructions for the four retrograde person-name questions, 1 point each."""
    assert 0 <= score <= 4
    true = {
        "uk_prime_minister": session_config["current_uk_pm"],
        "female_pm": "Margaret Thatcher",
        "us_president": session_config["current_us_president"],
        "assassinated_president": "John F. Kennedy",
    }
    # Realistic wrong answer for a current officeholder is naming their predecessor.
    wrong_alt = {
        "uk_prime_minister": session_config.get("previous_uk_pm"),
        "us_president": session_config.get("previous_us_president"),
    }
    wrong = set(RETRO_DROP_ORDER[:4 - score])
    return {
        field: (f"You get this wrong, naming {wrong_alt[field]} instead."
                 if field in wrong and wrong_alt.get(field) else
                 f"You say a name that isn't {value}."
                 if field in wrong else f"You correctly say {value}.")
        for field, value in true.items()
    }


PROMPT_MEMORY = """
Which three words did I ask you to remember?
> {recall_3words}

Repeat back the name and address you were just told: Harry Barnes, 73 Orchard Close, Kingsbridge, Devon.
> {anterograde_memory}

Can you tell me the name of the current Prime Minister?
> {uk_prime_minister}

Can you tell me the name of the first female Prime Minister?
> {female_pm}

Can you tell me the name of the current President of the United States?
> {us_president}

Can you name the President of the United States who was assassinated in the 1960s?
> {assassinated_president}
"""

def build_memory_prompt(recall_score, anterograde_score, retrograde_score, session_config):
    """Fill the template from the per-item scores -> one memory prompt."""
    return PREAMBLE + "\n" + PROMPT_MEMORY.format(
        recall_3words=delayed_recall(recall_score),
        anterograde_memory=anterograde_memory(anterograde_score),
        **retrograde_memory(retrograde_score, session_config),
    )


def run_memory_prompt(recall_score, anterograde_score, retrograde_score, session_config):
    """Send the built memory prompt to the local LLM and return the persona's spoken reply."""
    prompt = build_memory_prompt(recall_score, anterograde_score, retrograde_score, session_config)
    return _ask(prompt)


#=== Fluency ===
def _fluency_target_count(score, bands):
    """Smallest word count that scores `score` under `bands` (avoids relying on the open-ended top band)."""
    for min_count, _max_count, band_score in bands:
        if band_score == score:
            return min_count
    return 0

def verbal_fluency(score):
    """Word instruction for letter (P-word) fluency, scored 0-7 via LETTER_FLUENCY_BANDS.
    Leaves word choice to the LLM -- score_letter_fluency validates any real word against
    WordNet, so no fixed word list is needed here."""
    assert 0 <= score <= 7
    count = _fluency_target_count(score, LETTER_FLUENCY_BANDS)
    if count == 0:
        return "You cannot think of any words beginning with the letter P and go silent."
    return (f"You say exactly {count} different real English words beginning with the "
            "letter P (no names or places), then stop.")

def animal_fluency(score):
    """Word instruction for category (animal) fluency, scored 0-7 via CATEGORY_FLUENCY_BANDS.
    Leaves word choice to the LLM -- score_animal_fluency validates any real animal name
    against WordNet, so no fixed word list is needed here."""
    assert 0 <= score <= 7
    count = _fluency_target_count(score, CATEGORY_FLUENCY_BANDS)
    if count == 0:
        return "You cannot think of any animals and go silent."
    return f"You name exactly {count} different real animals, then stop."


PROMPT_FLUENCY = """
Name as many words as you can that begin with the letter P (not names or places) in one minute:
> {verbal_fluency}

Name as many different animals as you can in one minute:
> {animal_fluency}
"""

def build_fluency_prompt(letter_score, animal_score):
    """Fill the template from the per-item scores -> one fluency prompt."""
    return PREAMBLE + "\n" + PROMPT_FLUENCY.format(
        verbal_fluency=verbal_fluency(letter_score),
        animal_fluency=animal_fluency(animal_score),
    )


def run_fluency_prompt(letter_score, animal_score):
    """Send the built fluency prompt to the local LLM and return the persona's spoken reply."""
    prompt = build_fluency_prompt(letter_score, animal_score)
    return _ask(prompt)


# ==== Language
WORD_REPETITION_WORDS = ["caterpillar", "eccentricity", "unintelligible", "statistician"]

WORD_DROP_ORDER = ["statistician", "unintelligible", "eccentricity", "caterpillar"]

WORD_REPETITION_WRONG = {
    "caterpillar": "table",
    "eccentricity": "window",
    "unintelligible": "pencil",
    "statistician": "chair",
}

def word_repetition(correct_count):
    """Per-word recall instructions -- each word is spoken and answered separately,
    one at a time, not as a single batch (see the item's `instructions` in ACE-III.json)."""
    assert 0 <= correct_count <= 4
    wrong = set(WORD_DROP_ORDER[:4 - correct_count])
    return {
        word: (f"You are asked to repeat the word '{word}'. Instead, you just say "
                f"'{WORD_REPETITION_WRONG[word]}'."
                if word in wrong else f"You correctly repeat '{word}'.")
        for word in WORD_REPETITION_WORDS
    }
def sentence_repetition(score, sentence):
    """Single sentence-repetition question, 1 point. `sentence` is the exact phrase to repeat."""
    assert score in (0, 1)
    if score:
        return f'You correctly repeat the phrase back word for word: "{sentence}".'
    return f'You say a different sentence instead of "{sentence}".'



# Canonical naming-picture order (top-left to bottom-right), matching the
# click grid in visual_tasks.visual.GRID_ITEMS.
NAMING_ITEMS = ["spoon", "book", "kangaroo", "penguin", "anchor", "camel",
                "harp", "rhinoceros", "barrel", "crown", "crocodile", "accordion"]
NAMING_DROP_ORDER = list(reversed(NAMING_ITEMS))

def naming(correct_count):
    """
    Per-picture naming instructions -- like word_repetition, each picture is
    pointed to and named one at a time, not as a single 12-item batch response.
    spoon; book; penguin; anchor; camel or dromedary; barrel, keg, or tub; crown;
    crocodile or alligator; harp; rhinoceros or rhino; kangaroo or wallaby; piano accordion, accordion or squeeze box.
    12 points max
    """
    assert 0 <= correct_count <= 12
    wrong = set(NAMING_DROP_ORDER[:12 - correct_count])
    return {
        item: (f"You cannot name this picture, saying nothing or 'I don't know'."
                if item in wrong else f"You correctly say '{item}'.")
        for item in NAMING_ITEMS
    }

# The four "Comprehension: Which picture..." questions -- answered by clicking
# a grid cell in the live pipeline (run_click_task in visual_tasks/visual.py),
# but the clicked item's name is scored as plain text exactly like a spoken
# answer, so it fits the same instruction pattern as everything else here.
COMPREHENSION_PICTURES = {
    "monarchy": "crown",
    "marsupial": "kangaroo",
    "antarctic": "penguin",
    "nautical": "anchor",
}

def picture_comprehension(score, correct_answer):
    """Single 'Which picture...' question, 1 point."""
    assert score in (0, 1)
    if score:
        return f"You correctly point to and say '{correct_answer}'."
    return f"You point to the wrong picture, saying something other than '{correct_answer}'."

def reading(score):
    """Read the five irregular words aloud, all-or-nothing -- 1 point only if all five are read correctly."""
    assert score in (0, 1)
    words = "sew, pint, soot, dough, height"
    if score:
        return f"You correctly read all five words aloud: {words}."
    return (f"You mispronounce at least one of the words ({words}), reading it phonetically "
            "instead of using its irregular pronunciation.")

PEN_PAPER_FIELDS = ["paper_placed_on_pencil", "pencil_lifted_without_paper", "pencil_lifted_after_touching_paper"]

def comprehension(score):
    """Follow three-stage pencil/paper commands, scored out of 3 (one point per correctly
    performed step): paper on top of pencil; pencil not paper; pencil after touching paper."""
    assert 0 <= score <= 3
    wrong = set(PEN_PAPER_FIELDS[score:])
    lines = "\n".join(f'  "{field}": "{"no" if field in wrong else "yes"}"' for field in PEN_PAPER_FIELDS)
    return (
        "You performed three pencil-and-paper actions. Report exactly what happened by "
        "responding with ONLY this JSON object, filled in exactly as shown below, and "
        "nothing else -- no preamble, no explanation, no markdown fences:\n"
        "{\n" + lines + "\n}"
    )

def writing(score):
    """Write two sentences, scored 0-2 by score_sentence_writing (visual_tasks/writing.py) --
    a sentence only counts if it has a clear subject and verb, and only then is it checked
    for spelling/grammar errors. The instruction targets that structure directly instead of
    a vague "wrong" (the persona's PREAMBLE otherwise reads "wrong" as factually wrong)."""
    assert 0 <= score <= 2
    if score == 2:
        return ("You write exactly two separate sentences about your day, each with a "
                "clear subject and verb, both fully correct with no spelling or grammar mistakes.")
    if score == 1:
        return ("You write exactly one sentence about your day, with a clear subject and "
                "verb and no spelling or grammar mistakes, then stop completely.")
    return ("You write a few words with no verb and no subject-verb structure at all -- "
            "not a sentence, just a fragment like a single noun or a short list of words.")


PROMPT_LANGUAGE = """
Repeat after me: caterpillar.
> {caterpillar}

Now say: eccentricity.
> {eccentricity}

Now say: unintelligible.
> {unintelligible}

Now say: statistician.
> {statistician}

Repeat this phrase: "All that glitters is not gold."
> {sentence_repetition_1}

Repeat this phrase: "A stitch in time saves nine."
> {sentence_repetition_2}

Name this picture:
> {spoon}

Name this picture:
> {book}

Name this picture:
> {kangaroo}

Name this picture:
> {penguin}

Name this picture:
> {anchor}

Name this picture:
> {camel}

Name this picture:
> {harp}

Name this picture:
> {rhinoceros}

Name this picture:
> {barrel}

Name this picture:
> {crown}

Name this picture:
> {crocodile}

Name this picture:
> {accordion}

Looking at the pictures on the screen, which one is associated with the monarchy?
> {monarchy}

Which one is the marsupial?
> {marsupial}

Which one is found in the Antarctic?
> {antarctic}

Which one has a nautical connection?
> {nautical}

I would like you to read these words aloud: sew, pint, soot, dough, height.
> {reading}
"""

def build_language_prompt(word_score, sentence1_score, sentence2_score, naming_score,
                           monarchy_score, marsupial_score, antarctic_score, nautical_score,
                           reading_score):
    """Fill the template from the per-item scores -> one language prompt."""
    comprehension_scores = {
        "monarchy": monarchy_score,
        "marsupial": marsupial_score,
        "antarctic": antarctic_score,
        "nautical": nautical_score,
    }
    return PREAMBLE + "\n" + PROMPT_LANGUAGE.format(
        **word_repetition(word_score),
        sentence_repetition_1=sentence_repetition(sentence1_score, "All that glitters is not gold"),
        sentence_repetition_2=sentence_repetition(sentence2_score, "A stitch in time saves nine"),
        **naming(naming_score),
        **{key: picture_comprehension(comprehension_scores[key], answer)
           for key, answer in COMPREHENSION_PICTURES.items()},
        reading=reading(reading_score),
    )


def run_language_prompt(word_score, sentence1_score, sentence2_score, naming_score,
                         monarchy_score, marsupial_score, antarctic_score, nautical_score,
                         reading_score):
    """Send the built language prompt to the local LLM and return the persona's spoken reply."""
    prompt = build_language_prompt(word_score, sentence1_score, sentence2_score, naming_score,
                                    monarchy_score, marsupial_score, antarctic_score, nautical_score,
                                    reading_score)
    return _ask(prompt)


# ==== Visual === #
def cube_scorer(cube_band):
    if cube_band == 1:
        cube = "test/cube/1/Screenshot 2026-07-20 220116.png"
        return cube
    if cube_band == 2:
        cube = "test/cube/2/Screenshot 2026-07-20 220158.png"
        return cube
    else:
        return 0

def infinity_scorer(infinity_band):
    if infinity_band == 1:
        infinity = "test/InfinitySymbol/1/Screenshot 2026-07-19 220820.png"
        return infinity
    if infinity_band == 0:
        infinity = "test/InfinitySymbol/0/mark_0.png"
        return infinity
    else:
        pass
 
def clock_scorer(clock_band):
    """Clock-Scorer"""
    if clock_band == 5:
        clock =  "test/Clocks/5/Clock_5_1.png"
        return clock
    if clock_band == 4:
        clock =  "test/Clocks/4/Clock_4_1.png"
        return clock
    if clock_band == 3:
        clock =  "test/Clocks/3/Clock_3_1.png"
        return clock
    if clock_band == 2:
        clock =  "test/Clocks/2/Clock_S2.png"
        return clock
    if clock_band == 1:
        clock =  "test/Clocks/1/Clock_1.png"
        return clock

# True dot counts / fragmented letters, in the same order as the four Dot
# counting and four Fragmented letters questions in json/ACE-III.json.
DOT_ANSWERS = 10, 8, 9, 7
FRAGMENTED_LETTERS_ANSWERS = "M", "A", "T", "K"

def dot_counting(score, true_count):
    """Single dot-array question, 1 point."""
    assert score in (0, 1)
    if score:
        return f"You correctly count {true_count} dots."
    return (f"You miscount, saying any number of dots except {true_count}, without counting "
            f"each dot out loud one by one (that would say the correct {true_count} in passing) "
            f"-- just state your wrong total directly.")

def fragmented_letters(score, true_letter):
    """Single fragmented-letter question, 1 point."""
    assert score in (0, 1)
    if score:
        return f"You correctly identify the letter {true_letter}."
    return (f"You get this wrong, saying any letter except {true_letter}, without comparing "
            f"it to other letters out loud -- just state your wrong letter directly.")

def recall_memory(score):
    """Visuospatial section's "Delayed recall", asked again at the very end of the test,
    after the interference tasks (fluency/language/visuospatial). Same content and
    scoring as anterograde_memory (the original learning trial), so reuse it."""
    return anterograde_memory(score)

# Order elements go wrong in as recognition score drops from 5 to 0.
RECOGNITION_ANSWERS = {
    "name": "Harry Barnes",
    "number": "73",
    "street": "Orchard Close",
    "town": "Kingsbridge",
    "county": "Devon",
}

RECOGNITION_QUESTIONS = {
    "name": "Was the name Jerry Barnes, Harry Barnes, or Harry Bradford?",
    "number": "Was the number 37, 73, or 76?",
    "street": "Was the street Orchard Place, Oak Close, or Orchard Close?",
    "town": "Was the town Oakhampton, Kingsbridge, or Dartington?",
    "county": "Was the county Devon, Dorset, or Somerset?",
}
RECOGNITION_OPTIONS = {
    "name": "Jerry Barnes, Harry Barnes, or Harry Bradford",
    "number": "37, 73, or 76",
    "street": "Orchard Place, Oak Close, or Orchard Close",
    "town": "Oakhampton, Kingsbridge, or Dartington",
    "county": "Devon, Dorset, or Somerset",
}

RECOGNITION_DROP_ORDER = list(RECOGNITION_ANSWERS)

def recognition(score):
    """Instructions for the recognition elements actually asked about, 1 point each.
    Real ACE-III rule: only elements NOT already recalled get a multiple-choice hint --
    elements already known are skipped entirely, not asked. That's exactly the ones
    this score marks wrong, so only those are returned here.

    Uses a direct "Say X" command rather than the "When asked X, you get this wrong,
    answering Y instead of Z" phrasing -- the latter reliably makes the model either
    stall waiting for "the test question" or just echo the question back."""
    assert 0 <= score <= 5
    wrong = set(RECOGNITION_DROP_ORDER[:5 - score])
    return {
        key: f"You are asked whether the {key} was {RECOGNITION_OPTIONS[key]}. Say anything but '{RECOGNITION_OPTIONS[key]}'."
        for key in wrong
    }


PROMPT_VISUOSPATIAL = """
How many dots can you see in this picture?
> {dot_1}

How many dots can you see in this picture?
> {dot_2}

How many dots can you see in this picture?
> {dot_3}

How many dots can you see in this picture?
> {dot_4}

What letter is shown in this picture?
> {letter_1}

What letter is shown in this picture?
> {letter_2}

What letter is shown in this picture?
> {letter_3}

What letter is shown in this picture?
> {letter_4}

Now, tell me what you remember about that name and address we were repeating at the beginning:
> {delayed_recall}

Was the name Jerry Barnes, Harry Barnes, or Harry Bradford?
> {name}

Was the number 37, 73, or 76?
> {number}

Was the street Orchard Place, Oak Close, or Orchard Close?
> {street}

Was the town Oakhampton, Kingsbridge, or Dartington?
> {town}

Was the county Devon, Dorset, or Somerset?
> {county}
"""

def build_visuospatial_prompt(dot_scores, letter_scores, recall_score, recognition_score):
    """dot_scores/letter_scores are 4-item sequences of 0/1 scores, in json/ACE-III.json order.
    Fill the template from the per-item scores -> one visuospatial prompt (spoken/click items only --
    the drawing tasks (Infinity, Cube, Clock) are scored from a photo, not an LLM persona)."""
    assert len(dot_scores) == 4 and len(letter_scores) == 4
    dots = {f"dot_{i + 1}": dot_counting(s, DOT_ANSWERS[i]) for i, s in enumerate(dot_scores)}
    letters = {f"letter_{i + 1}": fragmented_letters(s, FRAGMENTED_LETTERS_ANSWERS[i]) for i, s in enumerate(letter_scores)}
    return PREAMBLE + "\n" + PROMPT_VISUOSPATIAL.format(
        **dots,
        **letters,
        delayed_recall=recall_memory(recall_score),
        **recognition(recognition_score),
    )

def run_visuospatial_prompt(dot_scores, letter_scores, recall_score, recognition_score):
    """Send the built visuospatial prompt to the local LLM and return the persona's spoken reply."""
    prompt = build_visuospatial_prompt(dot_scores, letter_scores, recall_score, recognition_score)
    return _ask(prompt)