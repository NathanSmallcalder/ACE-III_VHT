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

exact_cases = {
    "test": ("d", ["d"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    exact_cases.values(),
    ids=exact_cases.keys()
)
def test_score_exact(response, answers, expected):
    result = score_exact(response, answers)
    assert result == expected


integer_cases = {
    "test": ("Um it's ninety three", ["93"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    integer_cases.values(),
    ids=integer_cases.keys()
)
def test_score_integer(response, answers, expected):
    result = score_integer(response, answers)
    assert result == expected


fuzzy_cases = {
    "test": ("aeroplane", ["airplane"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    fuzzy_cases.values(),
    ids=fuzzy_cases.keys()
)
def test_score_fuzzy(response, answers, expected):
    result = score_fuzzy(response, answers)
    assert result == expected


fuzzy_list_cases = {
    "test": ("apple table penny", ["apple", "table", "penny"], 3),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    fuzzy_list_cases.values(),
    ids=fuzzy_list_cases.keys()
)
def test_score_fuzzy_list(response, answers, expected):
    result = score_fuzzy_list(response, answers)
    assert result == expected


all_correct_list_cases = {
    "test": ("apple table penny", ["apple", "table", "penny"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    all_correct_list_cases.values(),
    ids=all_correct_list_cases.keys()
)
def test_score_all_correct_list(response, answers, expected):
    result = score_all_correct_list(response, answers)
    assert result == expected


person_name_cases = {
    "test": ("It's Mr Smith", ["John Smith", "Smith"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    person_name_cases.values(),
    ids=person_name_cases.keys()
)
def test_score_person_name(response, answers, expected):
    result = score_person_name(response, answers)
    assert result == expected


sentence_repetition_cases = {
    "test": ("um let me see, the cat sat on the mat", ["the cat sat on the mat"], 1),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    sentence_repetition_cases.values(),
    ids=sentence_repetition_cases.keys()
)
def test_score_sentence_repetition(response, answers, expected):
    result = score_sentence_repetition(response, answers)
    assert result == expected


mixed_list_cases = {
    "test": ("apple 42", ["apple", "42"], 2),
}

@pytest.mark.parametrize(
    "response, answers, expected",
    mixed_list_cases.values(),
    ids=mixed_list_cases.keys()
)
def test_score_mixed_list(response, answers, expected):
    result = score_mixed_list(response, answers)
    assert result == expected


serial_sevens_cases = {
    "test": ("93 86 79 72 65", 5),
}

@pytest.mark.parametrize(
    "response, expected",
    serial_sevens_cases.values(),
    ids=serial_sevens_cases.keys()
)
def test_score_serial_sevens(response, expected):
    result = score_serial_sevens(response)
    assert result == expected


letter_fluency_cases = {
    "test": ("pen pot paper pay", 2),
}

@pytest.mark.parametrize(
    "response, expected",
    letter_fluency_cases.values(),
    ids=letter_fluency_cases.keys()
)
def test_score_letter_fluency(response, expected):
    result = score_letter_fluency(response)
    assert result == expected


animal_fluency_cases = {
    "test": ("dog cat salmon trout fish", 0),
}

@pytest.mark.parametrize(
    "response, expected",
    animal_fluency_cases.values(),
    ids=animal_fluency_cases.keys()
)
def test_score_animal_fluency(response, expected):
    result = score_animal_fluency(response)
    assert result == expected
