import pytest
from LLM.dialogue import classify_turn, extract_final_answer

classify_turn_cases = {
    "incomplete_trailing_sy": ("Um, I think, I think it is...well im not acctually too sy", "What month is it?", "incomplete"),
    "incomplete_trailing_hmmm": ("Um, I think, I think it is...well im not acctually too hmmm", "What month is it?", "incomplete"),
    "incomplete_trailing_sure": ("Um, I think, I think it is...well im not acctually too sure", "What month is it?", "incomplete"),
    "answer_hedged_short": ("Um well i dunno maybe i think its May", "What month is it?", "answer"),
    "answer_hedged_long": ("Um well i dunno maybe i think its May, i think that sounds right im not sure though", "What month is it?", "answer"),
    "repeat_bare": ("can you repeat that", "What month is it?", "repeat"),
    "repeat_with_guess": ("can you repeat that, if it was about a month maybe but im not sure", "What month is it?", "repeat"),
    "didnt respond":("", "What month is it?", "incomplete"),
    "off_topic_weather": ("Oh it's such a lovely day, I had eggs for breakfast.", "What day is it?", "off_topic"),
    "off_topic_family": ("My daughter is coming to visit me later.", "What year is it?", "off_topic"),
    "incomplete_person_name": ("Umm, it's... the Prime Minister is...", "Who is the current Prime Minister?", "incomplete"),
    "repeat_recognition": ("Sorry, could you say the options again?", "Was the street Orchard Place, Oak Close, or Orchard Close?", "repeat"),
    "answer_recognition_choice": ("It was Orchard Close.", "Was the street Orchard Place, Oak Close, or Orchard Close?", "answer"),
}

@pytest.mark.parametrize(
    "last_response, question_text, expected",
    classify_turn_cases.values(),
    ids=classify_turn_cases.keys()
)
def test_classify_turn(last_response, question_text, expected):
    result = classify_turn(
        last_response=last_response,
        question_text=question_text
    )
    assert result == expected


extract_final_answer_cases = {
    "direct_answer": ("Oh, lets see last month it was febuary, so it must be march", "What month is it?", "march"),
    "self_correction_single": ("Monday, no wait, it's Tuesday.", "What day is it?", "Tuesday"),
    "self_correction_multi_choice": ("Was the county Devon, Dorset, or Somerset? Dorset. No, actually, Devon.", "Was the county Devon, Dorset, or Somerset?", "Devon"),
    "reasoning_aside_ignored": ("It must be March, since my birthday is in April and that's next month.", "What month is it?", "March"),
    "trailing_reasoning_after_answer": ("Tuesday, because that's usually market day.", "What day is it?", "Tuesday"),
    "hedged_but_committed": ("Um, I think, maybe it's May.", "What month is it?", "May"),
    "brute_force_list_to_empty": ("Monday, Tuesday, Wednesday, Thursday, or Friday, could be any of them.", "What day is it?", ""),
    "no_commitment_i_dont_know": ("I don't know, I really can't say.", "What day is it?", ""),
    "retrograde_self_correction": ("Umm, Joe Biden... oh wait no, it's Donald Trump.", "Who is the current President of the United States?", "Donald Trump"),
    "season_hedged": ("Well it's not quite summer, more like spring I'd say.", "What season is it?", "spring"),
    "year_self_correction": ("Uh, 2024, no wait, I think it's 2026.", "What year is it?", "2026"),
    "recognition_reasoning": ("Well it's definitely not Oak Close, so I'll say Orchard Close.", "Was the street Orchard Place, Oak Close, or Orchard Close?", "Orchard Close"),
    "picture_comprehension_reasoning": ("It's not the penguin, penguins live in the Antarctic not near the monarchy, it's the crown.", "Which picture is associated with the monarchy?", "crown"),
}

@pytest.mark.parametrize(
    "last_response, question_text, expected",
    extract_final_answer_cases.values(),
    ids=extract_final_answer_cases.keys()
)
def test_extract_final_answer(last_response, question_text, expected):
    result = extract_final_answer(
        last_response=last_response,
        question_text=question_text
    )
    assert result == expected
