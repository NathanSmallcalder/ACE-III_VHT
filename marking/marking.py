from metaphone import doublemetaphone
import re
from nltk.corpus import wordnet as wn
import rapidfuzz
from marking.preprocessing import *

LETTER_FLUENCY_BANDS = [
    (0, 1, 0),
    (2, 3, 1),
    (4, 5, 2),
    (6, 7, 3),
    (8, 10, 4),
    (11, 13, 5),
    (14, 17, 6),
    (18, float("inf"), 7)
]


CATEGORY_FLUENCY_BANDS = [
    (0, 4, 0),
    (5, 6, 1),
    (7, 8, 2),
    (9, 10, 3),
    (11, 13, 4),
    (14, 16, 5),
    (17, 21, 6),
    (22, float("inf"), 7)
]

FUZZY_THRESHOLD = 90

_ANIMALS = None

def phonetic_equal(a, b):
    pa, sa = doublemetaphone(a)
    pb, sb = doublemetaphone(b)
    if not pa or not pb:
        return False
    codes_a = {c for c in (pa, sa) if c}
    codes_b = {c for c in (pb, sb) if c}
    return bool(codes_a & codes_b)

def score_exact(response, answers):
    """Single-character exact match (fragmented letters). Checks if the expected letter appears as a word in the response."""
    words = clean_response(response).split()
    return 1 if any(clean_response(a) in words for a in answers) else 0

def score_integer(response, answers):
    """Parsed integer match via sliding window (dot counting)."""
    response_words = clean_response(response).split()
    expected = [normalise_number(clean_response(a)) for a in answers]
    for y in expected:
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                if normalise_number(" ".join(response_words[i:i + n])) == y:
                    return 1
    return 0

def score_serial_sevens(response):
    """Scores each correct subtraction of 7 from the previous number said, starting from 100."""
    words = clean_response(response).split()
    numbers = []
    i = 0
    while i < len(words):
        found = False
        for n in range(min(2, len(words) - i), 0, -1):
            val = normalise_number(" ".join(words[i:i + n]))
            try:
                num = int(val)
                if 0 <= num < 100:
                    numbers.append(num)
                    i += n
                    found = True
                    break
            except ValueError:
                pass
        if not found:
            i += 1

    score = 0
    prev = 100
    for num in numbers[:5]:
        if prev - num == 7:
            score += 1
        prev = num
    return score

def score_fuzzy(response, answers):
    """Any answer from alternatives list matches = 1 point."""
    response_words = clean_response(response).split()
    for answer in answers:
        expected = normalise_number(clean_response(answer))
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                window = normalise_number(" ".join(response_words[i:i + n]))
                if rapidfuzz.fuzz.ratio(window, expected) >= FUZZY_THRESHOLD or phonetic_equal(window, expected):
                    return 1
    return 0

def score_fuzzy_list(response, answers):
    """Each answer in list scored separately via sliding window."""
    score = 0
    matched = set()
    response_words = clean_response(response).split()
    expected = [normalise_number(clean_response(a)) for a in answers]

    for y in expected:
        if y in matched:
            continue
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                window = normalise_number(" ".join(response_words[i:i + n]))
                if rapidfuzz.fuzz.ratio(window, y) >= FUZZY_THRESHOLD or phonetic_equal(window, y):
                    matched.add(y)
                    score += 1
                    break
            else:
                continue
            break
    return score


def get_animals():
    animals = set()
    for synset in wn.synsets("animal", pos=wn.NOUN):
        for hyponym in synset.closure(lambda s: s.hyponyms()):
            for lemma in hyponym.lemmas():
                animals.add(lemma.name().lower().replace("_", " "))
    return animals

def scaled_count(count, bands):
    for min_count, max_count, score in bands:
        if min_count <= count <= max_count:
            return score
    return 0

def is_valid_p_word(word):
    word = word.lower().strip()
    if not word.startswith("p"):
        return False
    synsets = wn.synsets(word)
    if not synsets:
        return False
    for synset in synsets:
        for lemma in synset.lemmas():
            if lemma.name().lower() == word and lemma.name()[0].islower():
                return True
    return False

def score_letter_fluency(response):
    words = clean_response(response).split()
    unique_valid = {w for w in set(words) if is_valid_p_word(w)}
    return scaled_count(len(unique_valid), LETTER_FLUENCY_BANDS)

def score_animal_fluency(response):
    global _ANIMALS
    if _ANIMALS is None:
        _ANIMALS = get_animals()
    words = clean_response(response).split()
    unique_valid = set()
    used_indices = set()

    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in _ANIMALS:
            unique_valid.add(bigram)
            used_indices.add(i)
            used_indices.add(i + 1)

    for i, w in enumerate(words):
        if i not in used_indices and w in _ANIMALS:
            unique_valid.add(w)

    return scaled_count(len(unique_valid), CATEGORY_FLUENCY_BANDS)

def parse_spoken_prompts(question: dict) -> list[str]:
    """Extract every spoken prompt from instructions, one entry per Wait-for-response pause."""
    instructions = question.get("instructions", "")
    chunks = instructions.split("Wait for a response.")
    prompts = []
    for chunk in chunks[:-1]:
        # Find all Speak: '...' entries — lookahead ensures closing ' is not followed
        # by a letter, so apostrophes inside text (today's) are handled correctly.
        speaks = re.findall(r"[Ss]peak[^']*'(.*?)'(?=[^a-zA-Z]|$)", chunk, re.DOTALL)
        if speaks:
            prompts.append(" ".join(speaks))
        else:
            # Non-Speak-prefixed prompts (e.g. word repetition continuations)
            first_q = chunk.find("'")
            last_q = chunk.rfind("'")
            if first_q != -1 and last_q > first_q:
                prompts.append(chunk[first_q + 1:last_q])
    return prompts

def get_sub_prompts(question: dict) -> list[str]:
    """Returns prompts only when each answer has its own spoken prompt (multi-prompt tracking)."""
    prompts = parse_spoken_prompts(question)
    n_answers = len(question.get("answers", []))
    if len(prompts) == n_answers and n_answers > 1:
        return prompts
    return []

def score_mixed_list(response, answers):
    """Each answer scored separately; numeric answers use integer match, strings use fuzzy."""
    score = 0
    matched = set()
    response_words = clean_response(response).split()

    for answer in answers:
        cleaned = clean_response(answer)
        normed = normalise_number(cleaned)
        is_number = normed.isdigit()

        if answer in matched:
            continue
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                window_raw = " ".join(response_words[i:i + n])
                window = normalise_number(window_raw)
                hit = (window == normed) if is_number else (rapidfuzz.fuzz.ratio(window_raw, cleaned) >= FUZZY_THRESHOLD)
                if hit:
                    matched.add(answer)
                    score += 1
                    break
            else:
                continue
            break
    return score

def score_question(response, question, sub_index=None):
    """Main dispatch. Returns None for manual questions (flag for review), int otherwise."""
    match_type = question.get("match_type", "fuzzy_list")
    answers = question.get("answers", [])

    if match_type == "manual":
        return None
    if match_type == "fluency_letter":
        return score_letter_fluency(response)
    if match_type == "fluency_animal":
        return score_animal_fluency(response)
    if not answers:
        return 0

    if sub_index is not None:
        return score_fuzzy(response, [answers[sub_index]]) if sub_index < len(answers) else 0

    dispatch = {
        "exact":        score_exact,
        "integer":      score_integer,
        "fuzzy":        score_fuzzy,
        "fuzzy_list":   score_fuzzy_list,
        "mixed_list":   score_mixed_list,
        "serial_sevens": lambda r, a: score_serial_sevens(r)
    }
    fn = dispatch.get(match_type)
    return fn(response, answers) if fn else 0