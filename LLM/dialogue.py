import re
from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.LLM import llm_strict, llm_warm

""" Introduce the patient to the assessment. """
def introduce(patient_name: str) -> str:
    """Brief, warm one-time introduction spoken before the very first question of the session."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor about to begin the ACE-III cognitive "
            "assessment with a patient. Reply with ONLY a brief (1-2 sentence) friendly "
            "introduction: greet the patient by name and let them know you'll be asking "
            "some questions now. Do not explain the test mechanics, do not ask a question "
            "yourself, do not use quotes."
        )),
        HumanMessage(content=f"Patient's name: {patient_name}")
    ])
    return result.content.strip().strip('"')

"""Short, natural acknowledgment of the patient's last answer, spoken before the next question."""
def acknowledge(last_response: str) -> str:
    """Short, natural acknowledgment of the patient's last answer, spoken before the next question."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You must not reply with any positive or negative judgment of the patient's answer."
            "Reply with ONLY a brief (3-6 word) natural acknowledgment of the "
            "patient's last answer, to say before moving on to the next question. "
            "Do not ask a question, do not repeat their answer, do not use quotes."
        )),
        HumanMessage(content=f"Patient said: {last_response}")
    ])
    return result.content.strip().strip('"')

""" Question Rephrasing """
def rephrase_question(question_text: str) -> str:
    """Rephrased version of the question spoken when the patient didn't understand."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor. The patient did not understand the question. "
            "Reply with ONLY a brief rephrased version of the question "
            "to help them understand. Do not add new information, do not judge their "
            "response, do not use quotes."
        )),
        HumanMessage(content=f"Question: {question_text}")
    ])
    return result.content.strip().strip('"')

# Patient has given an Answer
# Patient Needs a Repeat
# Patient is Off Topic
# Patient's Answer is Incomplete
LABELS = {"answer", "repeat", "off_topic", "incomplete"}

def classify_turn(last_response: str, question_text: str = "") -> str:
    out = llm_strict.invoke([
        SystemMessage(content=(
            "Classify the patient's utterance during a cognitive test, given the question "
            "they were just asked. "
            "Reply with EXACTLY one of these words and nothing else: "
            "answer, repeat, off_topic, incomplete.\n"
            "- repeat: the patient is asking YOU to repeat or re-say the question "
            "(e.g. 'what?', 'sorry?', 'can you say that again'), or says they didn't hear it\n"
            "- off_topic: the utterance is not a plausible attempt at this specific question "
            "(rambling, nonsense syllables, unrelated remarks, talking about something else "
            "entirely) — even if it is wrong, it is off_topic unless it is at least a genuine "
            "attempt to address what was asked\n"
            "- incomplete: began a plausible attempt but there is clear evidence they were "
            "cut off or trailed off before finishing (hesitation fillers like 'um'/'uh', "
            "self-interruption, or trailing punctuation like '...'). "
            "A short or single-word reply is NOT incomplete just because it is brief — "
            "many valid answers (e.g. a single number, name, ordinal, or word, with or "
            "without a leading article like 'the') are naturally terse and complete as-is. "
            "'first', 'the first', 'the second', etc. are complete answers on their own, "
            "not unfinished phrases. Only use incomplete when the utterance itself shows "
            "actual signs of being unfinished, not merely because it lacks extra detail\n"
            "- answer: gave what they intend as their answer to THIS question, whether "
            "correct or incorrect, no matter how short. Saying the same word or phrase "
            "multiple times in a row is still an answer, not a request to repeat the "
            "question\n"
            "Judge only whether the utterance is a genuine attempt at answering the given "
            "question — do not judge whether it is factually correct.\n\n"
            "Examples:\n"
            "Question asked: What day of the week is it?\nPatient said: Wednesday.\n-> answer\n"
            "Question asked: What is today's date?\nPatient said: first\n-> answer\n"
            "Question asked: What is today's date?\nPatient said: the first\n-> answer\n"
            "Question asked: What season is it?\nPatient said: Summer, summer, summer.\n-> answer\n"
            "Question asked: What day of the week is it?\nPatient said: What? Can you repeat that?\n-> repeat\n"
            "Question asked: What month is it?\nPatient said: Um, I think, I think it is...\n-> incomplete"
        )),
        HumanMessage(content=f"Question asked: {question_text}\nPatient said: {last_response}")
    ])

    content = out.content.strip().lower()
    matches = re.findall(r"\b(" + "|".join(LABELS) + r")\b", content)
    if matches:
        return matches[-1]
    return "repeat"

