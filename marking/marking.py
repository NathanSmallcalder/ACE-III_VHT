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

""" """
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

_NAME_FILLER_WORDS = {
    "his", "her", "its", "it's", "name", "is", "was", "the", "a", "an",
    "um", "uh", "i", "think", "that's", "that", "mr", "mrs", "ms", "dr",
    "president", "minister", "prime",
}
_CORRECTION_MARKERS = {"no", "not", "sorry", "actually", "wait", "mean", "rather"}

def _norm(text):
    return normalise_number(clean_response(text)).split()

def _same(a, b):
    return rapidfuzz.fuzz.ratio(a, b) >= FUZZY_THRESHOLD or phonetic_equal(a, b)

def score_person_name(response, answers):
    """1 if the response names the person. A bare surname (with or without
    filler/honorifics) counts; a surname preceded by a substantive but wrong
    given name does not."""
    full = [a for a in answers if len(a.split()) > 1]
    if score_fuzzy(response, full):
        return 1

    surnames = [_norm(a)[0] for a in answers if len(a.split()) == 1]
    if not surnames:
        return 0

    full_tokens = [_norm(a) for a in full]
    words = _norm(response)

    for surname in surnames:
        # every accepted given-name sequence ending in this surname
        givens = [t[:-1] for t in full_tokens if _same(t[-1], surname)]

        for i, w in enumerate(words):
            if not _same(w, surname):
                continue

            claimed, j = [], i - 1
            while j >= 0 and words[j] not in _NAME_FILLER_WORDS \
                         and words[j] not in _CORRECTION_MARKERS:
                claimed.insert(0, words[j])
                j -= 1

            if not claimed:
                return 1
            if any(len(g) == len(claimed) and all(_same(x, y) for x, y in zip(claimed, g))
                   for g in givens):
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

def score_all_correct_list(response, answers):
    """All-or-nothing: 1 point only if every item in the list was matched,
    else 0. Used where the guide gives no partial credit (e.g. Reading:
    'Score 1 point if all five words are read correctly')."""
    return 1 if score_fuzzy_list(response, answers) == len(answers) else 0

def score_sentence_repetition(response, answers):
    """Whole-phrase fuzzy match. Unlike score_fuzzy's n-gram sliding window
    (built for short answers), a repeated sentence must be judged as one
    unit against the full expected phrase."""
    expected = clean_response(answers[0])
    return 1 if rapidfuzz.fuzz.ratio(clean_response(response), expected) >= FUZZY_THRESHOLD else 0


_ANIMAL_ROOTS = None

def _get_animal_roots():
    global _ANIMAL_ROOTS
    if _ANIMAL_ROOTS is None:
        _ANIMAL_ROOTS = set(wn.synsets("animal", pos=wn.NOUN))
    return _ANIMAL_ROOTS

def _is_animal_synset(synset):
    roots = _get_animal_roots()
    return synset in roots or bool(roots & set(synset.closure(lambda s: s.hypernyms())))

def get_animals():
    animals = set()
    for synset in wn.synsets("animal", pos=wn.NOUN):
        for hyponym in synset.closure(lambda s: s.hyponyms()):
            for lemma in hyponym.lemmas():
                animals.add(lemma.name().lower().replace("_", " "))
    return animals

def _drop_subsumed_categories(unique_valid):
    """If a more specific animal word is also present, drop the higher-order
    category word it's subsumed under (e.g. 'fish' dropped when 'salmon' and
    'trout' are also said) — only the specific exemplars should count."""
    synset_of = {}
    for w in unique_valid:
        for ss in wn.synsets(w.replace(" ", "_"), pos=wn.NOUN):
            if _is_animal_synset(ss):
                synset_of[w] = ss
                break

    to_drop = set()
    for w, ss_w in synset_of.items():
        for x, ss_x in synset_of.items():
            if w != x and ss_w in ss_x.closure(lambda s: s.hypernyms()):
                to_drop.add(w)
                break
    return unique_valid - to_drop

def scaled_count(count, bands):
    for min_count, max_count, score in bands:
        if min_count <= count <= max_count:
            return score
    return 0

def p_word_root(word):
    """Return the dictionary root of `word` if it's a valid common P-word, else None.
    Normalizing to the WordNet root (via morphy) merges perseverations and plurals
    (pay/paid/pays -> pay, pot/pots -> pot) into a single countable word."""
    word = word.lower().strip()
    if not word.startswith("p"):
        return None
    root = wn.morphy(word)
    if root is None:
        return None
    for synset in wn.synsets(root):
        for lemma in synset.lemmas():
            if lemma.name().lower() == root and lemma.name()[0].islower():
                return root
    return None

def score_letter_fluency(response):
    words = clean_response(response).split()
    roots = {p_word_root(w) for w in words}
    roots.discard(None)
    return scaled_count(len(roots), LETTER_FLUENCY_BANDS)

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
        if i in used_indices:
            continue
        if w in _ANIMALS:
            unique_valid.add(w)
        else:
            root = wn.morphy(w, wn.NOUN)
            if root and root in _ANIMALS:
                unique_valid.add(root)

    unique_valid = _drop_subsumed_categories(unique_valid)
    return scaled_count(len(unique_valid), CATEGORY_FLUENCY_BANDS)

def parse_spoken_prompts(question: dict) -> list[str]:
    """Extract every spoken prompt from instructions, one entry per Wait-for-response pause."""
    instructions = question.get("instructions", "")
    chunks = instructions.split("Wait for a response.")
    prompts = []
    for chunk in chunks[:-1]:
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

def score_mixed_list_detailed(response, answers):
    """Like score_mixed_list but reports which answers matched, in answer order.
    Used to carry per-element recall results into a later recognition task."""
    matched_text = set()
    matched = [False] * len(answers)
    response_words = clean_response(response).split()

    for idx, answer in enumerate(answers):
        cleaned = clean_response(answer)
        normed = normalise_number(cleaned)
        is_number = normed.isdigit()

        if cleaned in matched_text:
            continue
        for n in range(1, min(3, len(response_words)) + 1):
            for i in range(len(response_words) - n + 1):
                window_raw = " ".join(response_words[i:i + n])
                window = normalise_number(window_raw)
                hit = (window == normed) if is_number else (rapidfuzz.fuzz.ratio(window_raw, cleaned) >= FUZZY_THRESHOLD)
                if hit:
                    matched_text.add(cleaned)
                    matched[idx] = True
                    break
            else:
                continue
            break
    return matched

def score_mixed_list(response, answers):
    """Each answer scored separately; numeric answers use integer match, strings use fuzzy."""
    return sum(score_mixed_list_detailed(response, answers))

def _score_clock_visual(image_path, response):
    from visual_tasks.clock_scorer import score_clock_image
    return score_clock_image(response)["total"]

def _score_wire_cube_visual(image_path, response):
    if not response:
        return None
    from visual_tasks.cube_scorer import score_cube_image
    return score_cube_image(response)["total"]

def _score_infinity_diagram_visual(image_path, response):
    if not response:
        return None
    from visual_tasks.infinity_scorer import score_infinity_image
    return score_infinity_image(response)["total"]


def _score_writing_visual(image_path, response):
    if not response:
        return None
    from visual_tasks.writing import score_writing_image
    return score_writing_image(response)["total"]


def _score_pen_paper_visual(image_path, response):
    if not response:
        return None
    from visual_tasks.pen_paper_scorer import score_pen_paper_video
    return score_pen_paper_video(response)["total"]


# Maps a question_text prefix to its scorer. Extend here when a new visual task gets an automated scorer.
VISUAL_SCORERS = {
    "Clock": _score_clock_visual,
    "Wire Cube": _score_wire_cube_visual,
    "Infinity Diagram": _score_infinity_diagram_visual,
    "Writing": _score_writing_visual,
    "Comprehension": _score_pen_paper_visual,
}


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
    if match_type == "visual":
        text = question.get("question_text", "")
        for prefix, scorer in VISUAL_SCORERS.items():
            if text.startswith(prefix + ":"):
                return scorer(question.get("image"), response)
        return None  # no automated scorer yet for this visual task
    if not answers:
        return 0

    if sub_index is not None:
        if sub_index >= len(answers):
            return 0
        alt = answers[sub_index]
        # Some dynamic sub-answers (e.g. date +/- tolerance) resolve to a list
        # of acceptable alternatives rather than a single string.
        return score_fuzzy(response, alt if isinstance(alt, list) else [alt])

    dispatch = {
        "exact":        score_exact,
        "integer":      score_integer,
        "fuzzy":        score_fuzzy,
        "fuzzy_list":   score_fuzzy_list,
        "person_name":  score_person_name,
        "mixed_list":   score_mixed_list,
        "all_correct_list": score_all_correct_list,
        "serial_sevens": lambda r, a: score_serial_sevens(r),
        "sentence_repetition": score_sentence_repetition,
    }
    fn = dispatch.get(match_type)
    return fn(response, answers) if fn else 0