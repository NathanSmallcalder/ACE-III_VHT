from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.LLM import llm_strict, llm_warm


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


def soften_repeat() -> str:
    """Short reassuring phrase spoken before repeating a question the patient missed."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor. The patient didn't catch the "
            "question. Reply with ONLY a brief (3-6 word) reassuring phrase to "
            "say before repeating it. Do not repeat the question itself."
        ))
    ])
    return result.content.strip().strip('"')

# Patient has given an Answer
# Patient Needs a Repeat
# Patient is Off Topic
# Patient's Answer is Incomplete
_LABELS = {"answer", "repeat", "off_topic", "incomplete"}

def classify_turn(last_response: str) -> str:
    out = llm_strict.invoke([
        SystemMessage(content=(
            "Classify the patient's utterance during a cognitive test. "
            "Reply with EXACTLY one of these words and nothing else: "
            "answer, repeat, off_topic, incomplete.\n"
            "- repeat: asked you to repeat or didn't hear\n"
            "- off_topic: spoke but did not attempt the task\n"
            "- incomplete: began answering but seems unfinished\n"
            "- answer: gave what they intend as their answer\n"
            "Classify intent only. Do not judge correctness."
        )),
        HumanMessage(content=f"Patient said: {last_response}")
    ])
    label = out.content.strip().lower().split()[0].strip(".,'\"")
    return label if label in _LABELS else "answer"