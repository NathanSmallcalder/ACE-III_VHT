import re
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.LLM import llm_strict, llm_warm

""" Introduce the patient to the assessment. """
def introduce(patient_name: str) -> str:
    """Brief, warm one-time introduction spoken before the very first question of the session."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor about to begin the ACE-III cognitive "
            "assessment with a patient. Reply with a friendly introduction"
            "greet the patient by name and let them know you'll be asking "
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
            "The patient just answered a question in a cognitive test. Reply with a short "
            "neutral acknowledgment (3-6 words) that closes the turn before the next question. "
            "Your reply is always a statement, never a question, and never ends with '?'. "
            "Do not judge the answer, repeat it, or comment on its content.\n\n"
            "Examples:\n"
            "Patient: It's summer.\n-> Okay, thank you.\n"
            "Patient: Wednesday, I think.\n-> Alright, thanks.\n"
            "Patient: The twelfth.\n-> Got it, thank you.\n"
            "Patient: I'm not sure.\n-> That's alright, thank you.\n"
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
            "Reply with ONLY a rephrased version of the question "
            "to help them understand. Do not add new information, do not judge their "
            "response, do not use quotes. Do not ask a question based off the users response."
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
            "Classify a patient's utterance during a cognitive test, given the question "
            "they were just asked. Reply with EXACTLY one of these words, nothing else: "
            "answer, repeat, off_topic, incomplete.\n"
            "- answer: a genuine attempt at THIS question, right or wrong, however short. "
            " replies (a single number, word, name, ordinal, or 'the first') are "
            "complete answers. Repeating the same word in context thats relivent to the question is an answer. "
            "If a real answer value appears anywhere in the utterance, it counts as answer even "
            "when wrapped in dismissive commentary, sarcasm, attitude, or hesitation fillers "
            "('um'/'uh') — judge the content, not the tone or delivery.\n"
            "- repeat: asking YOU to say the question again ('what?', 'sorry?', 'say that "
            "again') or saying they didn't hear it.\n"
            "- off_topic: not a plausible attempt at this question — rambling, nonsense "
            "syllables, or talking about something else. A wrong answer is still an "
            "answer, not off_topic.\n"
            "- incomplete: hesitation fillers or self-interruption AND the utterance never "
            "actually lands on an answer — it trails off before stating one. If it starts "
            "hesitant but still ends with a clear answer value, that's answer, not incomplete.\n"
            "Judge only whether it's a genuine attempt, not whether it's factually correct.\n\n"
            "Examples:\n"
            "Q: What day is it?\nPatient: Wednesday.\n-> answer\n"
            "Q: What is today's date?\nPatient: the first\n-> answer\n"
            "Q: What season is it?\nPatient: Repetition of the same word.\n-> answer\n"
            "Q: What month is it?\nPatient: Very good, actually. Who cares? It's July.\n-> answer\n"
            "Q: What season is it?\nPatient: uh... is uh... winter\n-> answer\n"
            "Q: Which street is this?\nPatient: Um, Hospital Road.\n-> answer\n"
            "Q: What day is it?\nPatient: What? Can you repeat that?\n-> repeat\n"
            "Q: What month is it?\nPatient: Um, I think, I think it is...\n-> incomplete"
        )),
        HumanMessage(content=f"Question asked: {question_text}\nPatient said: {last_response}")
    ])

    content = out.content.strip().lower()
    matches = re.findall(r"\b(" + "|".join(LABELS) + r")\b", content)
    if matches:
        return matches[-1]
    return "repeat"

