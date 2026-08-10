import pytest
from marking.marking import (
    score_exact,
    score_integer,
    score_fuzzy,
    score_fuzzy_list,
    score_all_correct_list,
    score_person_name,
    score_sentence_repetition,
    score_mixed_list,
    score_serial_sevens,
    score_letter_fluency,
    score_animal_fluency,
)

# score_exact: Fragmented letters (Visuospatial) -- answers M, A, T, K
exact_cases = {
    "test": ("It's an M", ["M"], 1),
    "letter_A": ("A", ["A"], 1),
    "letter_T": ("um it's a t I think", ["T"], 1),
    "letter_K": ("K", ["K"], 1),
    "case_insensitive": ("m", ["M"], 1),
    "no_match": ("It's a K", ["M"], 0),
    "not_word_boundary": ("map", ["M"], 0),
    "spam_lands_on_correct_letter": ("A,B,C, wait its definitely M", ["M"], 1),
    "spam_lands_on_wrong_letter": ("A,B,C,M wait its definitely D", ["M"], 0),
    "empty_response": ("", ["M"], 0),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    exact_cases.values(),
    ids=exact_cases.keys()
)
def test_score_exact(response, answers, expected):
    result = score_exact(response, answers)
    assert result == expected


# score_integer: Dot counting (Visuospatial) -- answers 10, 8, 9, 7
integer_cases = {
    "test": ("Um it's ten", ["10"], 1),
    "digit_response": ("8", ["8"], 1),
    "no_match": ("it's nine", ["7"], 0),
    "embedded_in_sentence": ("there's about eight of them", ["8"], 1),
    "multiple_answers": ("it's seven", ["10", "7"], 1),
    "ordinal_suffix": ("it's the 10th", ["10"], 1),
    "spam_lands_on_target_last": ("is it seven, eight, nine, or ten", ["10"], 1),
    "drifts_to_different_number_after_target": ("is it ten, nine, eight, or seven", ["10"], 0),
    "genuine_correction_away_from_target": ("is it ten no wait seven", ["10"], 0),
    "spam_same_awnser": ("ten yes ten, ten", ["10"],1)
}

@pytest.mark.parametrize(
    "response, answers, expected",
    integer_cases.values(),
    ids=integer_cases.keys()
)
def test_score_integer(response, answers, expected):
    result = score_integer(response, answers)
    assert result == expected


# score_fuzzy: Comprehension picture questions (Language) -- crown/10, kangaroo/3, penguin/4, anchor/5
fuzzy_cases = {
    "test": ("it's the crown", ["crown"], 1),
    "no_match": ("it's the anchor", ["crown",], 0),
    "marsupial": ("kangaroo", ["kangaroo"], 1),
    "embedded_in_sentence": ("I think it's the penguin one", ["penguin"], 1),
    "phonetic_match": ("pengwin", ["penguin"], 1),
    "repeat_words_stutter_eccentricity":("e-eccen... tri... ci... ty.", ["eccentricity"], 1),
    "repeat_words_small_stutter":("ca-ter... pil-lar.", ["caterpillar"], 1),
    "repeat_words_missing_syllable":("un... in... tell... i... ble.", ["unintelligible"], 0),
    "repeat_words_stutter":("e-e-eccen... tri... city.", ["eccentricity"], 1) 
}

@pytest.mark.parametrize(
    "response, answers, expected",
    fuzzy_cases.values(),
    ids=fuzzy_cases.keys()
)
def test_score_fuzzy(response, answers, expected):
    result = score_fuzzy(response, answers)
    assert result == expected


# score_fuzzy_list: Three-word registration/recall (Attention/Memory) -- lemon, key, ball
fuzzy_list_cases = {
    "test": ("lemon key ball", ["lemon", "key", "ball"], 3),
    "spam_answer": ("orange lemon spoon key coin ball", ["lemon", "key", "ball"], 3),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    fuzzy_list_cases.values(),
    ids=fuzzy_list_cases.keys()
)
def test_score_fuzzy_list(response, answers, expected):
    result = score_fuzzy_list(response, answers)
    assert result == expected


# score_all_correct_list: Reading (Language) -- sew, pint, soot, dough, height
all_correct_list_cases = {
    "test": ("sew pint soot dough height", ["sew", "pint", "soot", "dough", "height"], 1),
    "spam_answer": ("sew sew red pint so-soot dough dough height", ["sew", "pint", "soot", "dough", "height"], 1),
    "misread_word_via_stray_letter_fragment_correctly_fails": (
        "s-sew... p-pint... s-soot... d-duff... h-height...",
        ["sew", "pint", "soot", "dough", "height"], 0,
    ),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    all_correct_list_cases.values(),
    ids=all_correct_list_cases.keys()
)
def test_score_all_correct_list(response, answers, expected):
    result = score_all_correct_list(response, answers)
    assert result == expected


# score_person_name: Retrograde memory (Memory) -- first female UK PM: Margaret Thatcher / Thatcher.
# response is always the single value extract_final_answer already resolved it to
# (self-correction/listing-every-option already settled upstream), never raw patient speech.
person_name_cases = {
    "full_name": ("Margaret Thatcher", ["Margaret Thatcher", "Thatcher"], 1),
    "bare_surname": ("Thatcher", ["Margaret Thatcher", "Thatcher"], 1),
    "wrong_given_name_with_right_surname_not_credited": ("Winston Thatcher", ["Margaret Thatcher", "Thatcher"], 0),
    "no_commitment_extracted_as_empty": ("", ["Margaret Thatcher", "Thatcher"], 0),
    "title_stripped": ("President Kennedy", ["John F. Kennedy", "John Kennedy", "Kennedy", "JFK"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    person_name_cases.values(),
    ids=person_name_cases.keys()
)
def test_score_person_name(response, answers, expected):
    result = score_person_name(response, answers)
    assert result == expected


# score_sentence_repetition: Sentence repetition (Language) -- "All that glitters is not gold"
sentence_repetition_cases = {
    "test": ("um let me see, all that glitters is not gold", ["All that glitters is not gold"], 1),
    "spam_answer": ("a stitch in time saves nine, then, all that glitters is not gold", ["All that glitters is not gold"], 1),
    "completely_different_sentence": ("every dog has its day", ["All that glitters is not gold"], 0),
    "trailing_off_before_last_word_is_wrong": (
        "A stitch... in time... um... saves... saves...",
        ["A stitch in time saves nine"], 0,
    ),
    "heavily_hedged_but_complete_is_still_correct": (
        "um... let me think... a stitch... in time... um... saves... nine, I think",
        ["A stitch in time saves nine"], 1,
    ),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    sentence_repetition_cases.values(),
    ids=sentence_repetition_cases.keys()
)
def test_score_sentence_repetition(response, answers, expected):
    result = score_sentence_repetition(response, answers)
    assert result == expected
    
# score_mixed_list: Name and address learning/recall (Memory) -- Harry, Barnes, 73, Orchard, Close, Kingsbridge, Devon
mixed_list_cases = {
    "test": ("Harry lives at 73", ["Harry", "73"], 2),
    "spam_answer": ("Barry 17 Harry 99 73 close", ["Harry", "73"], 2),
    "plain_wrong_answer": ("I don't know, I can't remember anything", ["Harry", "73"], 0),
    "numbers_as_words": ("seventy three Orchard Close", ["73", "Orchard"], 2),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    mixed_list_cases.values(),
    ids=mixed_list_cases.keys()
)
def test_score_mixed_list(response, answers, expected):
    result = score_mixed_list(response, answers)
    assert result == expected


# score_serial_sevens: Serial 7s (Attention) -- 93, 86, 79, 72, 65
serial_sevens_cases = {
    "test": ("93 86 79 72 65", 5),
    "spoken_number_words": ("ninety three eighty six seventy nine seventy two sixty five", 5),
    "single_step_error_does_not_cascade": ("93 86 80 73 66", 4),
    "consistently_wrong_from_start": ("94 88 82 76 70", 0),
    "extra_number_inserted_early": ("93 92 86 79 72 65", 3),
    "more_than_five_numbers_only_first_five_scored": ("93 86 79 72 65 58 51", 5),
    "numbers_over_100_filtered_out": ("100 93 186 79 72 65", 3),
    "negative_values": ("93 86 79 -2 -9", 3),
    "single_number_only": ("93", 1),
    "no_numbers_said": ("um let me think", 0),
    "empty_response": ("", 0),
    "answer": ("100 93 93 86 79 72 65", 5),
    "real_transcript_stutter_prefix_on_a_number": (
        "nine-ninety-three... eighty-six... sev-seventy-nine... "
        "um... sev-seventy-four... and... sixty-eight...",
        3,
    ),
    "echoed_question_before_the_sequence_still_works": (
        "one hundred minus seven is ninety three, eighty six, seventy nine, seventy two, sixty five",
        5,
    ),
    "ones_digit_completing_a_compound_not_mistaken_for_a_stutter": (
        "ninety three, eighty six, seventy nine, seventy two, sixty five",
        5,
    ),
}

@pytest.mark.parametrize(
    "response, expected",
    serial_sevens_cases.values(),
    ids=serial_sevens_cases.keys()
)
def test_score_serial_sevens(response, expected):
    result = score_serial_sevens(response)
    assert result == expected


# score_letter_fluency: Letter fluency, letter P (Fluency) -- open-ended, no fixed answer list in json
letter_fluency_cases = {
    "test": ("pen pot paper pay", 2),
    "empty_response": ("", 0),
    "band_0_one_word": ("pen", 0),
    "band_1_two_words": ("pen pot", 1),
    "band_2_four_words": ("pen pot paper park", 2),
    "band_3_six_words": ("pen pot paper park paint phone", 3),
    "band_4_eight_words": ("pen pot paper park paint phone plate pencil", 4),
    "band_5_eleven_words": ("pen pot paper park paint phone plate pencil purse pillow puzzle", 5),
    "band_6_fourteen_words": ("pen pot paper park paint phone plate pencil purse pillow puzzle pumpkin potato planet", 6),
    "band_7_eighteen_words": ("pen pot paper park paint phone plate pencil purse pillow puzzle pumpkin potato planet pirate palace pepper piano", 7),
    "excluded_place_name": ("pen paris", 0),
    "excluded_person_name_first_name": ("pen peter", 0),
    "ordinary_words_not_over_excluded": ("pen park page pail", 2),
    "manual_rule1_repetitions": ("pen pen pen pot", 1),
    "manual_rule2_perseverations": ("pay paid pays pot", 1),
    "manual_rule3_intrusions": ("pen cat dog pot", 1),
    "manual_rule4_proper_names": ("pen peter paris pot", 1),
    "manual_rule5_plurals": ("pot pots pen", 1),
}

@pytest.mark.parametrize(
    "response, expected",
    letter_fluency_cases.values(),
    ids=letter_fluency_cases.keys()
)
def test_score_letter_fluency(response, expected):
    result = score_letter_fluency(response)
    assert result == expected


# score_animal_fluency: Category fluency (Fluency) -- open-ended, no fixed answer list in json
animal_fluency_cases = {
    "test": ("dog cat salmon trout fish", 0),
    "band_0_four_animals": ("dog cat horse cow", 0),
    "band_1_five_animals": ("dog cat horse cow goat", 1),
    "band_2_seven_animals": ("dog cat horse cow goat sheep pig", 2),
    "band_3_nine_animals": ("dog cat horse cow goat sheep pig chicken duck", 3),
    "band_4_eleven_animals": ("dog cat horse cow goat sheep pig chicken duck goose rabbit", 4),
    "band_5_fourteen_animals": ("dog cat horse cow goat sheep pig chicken duck goose rabbit mouse rat fox", 5),
    "band_6_seventeen_animals": ("dog cat horse cow goat sheep pig chicken duck goose rabbit mouse rat fox wolf bear lion", 6),
    "band_7_twentytwo_animals": ("dog cat horse cow goat sheep pig chicken duck goose rabbit mouse rat fox wolf bear lion tiger elephant giraffe zebra monkey", 7),
    "category_word_dropped_when_exemplars_also_said": ("dog cat horse cow salmon trout fish", 1),
    "category_word_kept_when_no_exemplar_said": ("dog cat horse cow bird", 1),
    "repeated_animal_counted_once": ("dog cat horse cow dog goat", 1),
    "plural_normalised_via_wordnet_morphy": ("Dog.. urm  cat ugh? horse cow and yes goats", 1),
    "two_word_animal_name_matched_as_bigram": ("dog cat horse cow guinea pig", 1),
    "no_animals_said": ("car table lamp chair", 0),
    "empty_response": ("", 0),
}

@pytest.mark.parametrize(
    "response, expected",
    animal_fluency_cases.values(),
    ids=animal_fluency_cases.keys()
)
def test_score_animal_fluency(response, expected):
    result = score_animal_fluency(response)
    assert result == expected
