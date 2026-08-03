import pytest
from LLM.dialogue import classify_turn
test =  {
    "test": ("Um, I think, I think it is...well im not acctually too sy", "What month is it?", "incomplete"),
    "test2": ("Um, I think, I think it is...well im not acctually too hmmm", "What month is it?", "incomplete"),
    "test3": ("Um, I think, I think it is...well im not acctually too sure", "What month is it?", "incomplete"),
    "test4": ("Um well i dunno maybe i think its May", "What month is it?", "answer"),
    "test5": ("Um, I think, I think it is...well im not acctually too hmmm", "What month is it?", "incomplete"),
    "test6": ("Um, I think, I think it is...well im not acctually too sure", "What month is it?", "incomplete"),
    "test4": ("Um well i dunno maybe i think its May, i think that sounds right im not sure though", "What month is it?", "answer"),
    "test7": ("can you repeat that", "What month is it?", "repeat"),
    "test7": ("can you repeat that, if it was about a month maybe but im not sure", "What month is it?", "repeat"),
}

@pytest.mark.parametrize(
    "last_response, question_text, expected", 
    test.values(), 
    ids=test.keys()  
)
def test_classify_turn_incomplete(last_response, question_text, expected):
    result = classify_turn(
        last_response=last_response,
        question_text=question_text
    )
    assert result == expected