import re
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.LLM import llm_strict, llm_warm

def introduce(patient_name: str) -> str:
    """Brief, warm one-time introduction spoken before the very first question of the session."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor about to begin the ACE-III cognitive "
            "assestion sment with a patient. Reply with a friendly introduction"
            "greet the patient by name and let them know you'll be asking "
            "some questions now. Do not explain the test mechanics, do not ask a ques"
            "yourself, do not use quotes. inform the patient that they will need a pen and a few peices of paper in front of them"
        )),
        HumanMessage(content=f"Patient's name: {patient_name}")
    ])
    return result.content.strip().strip('"')


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
            "Patient: Twentyfour\n -> alright, moving on"
        )),
        HumanMessage(content=f"Patient said: {last_response}")
    ])
    return result.content.strip().strip('"')


def transition() -> str:
    """Cue spoken when moving to a different kind of task (domain or response
    modality changed). Never names the clinical domain/task, to avoid priming
    the patient."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor moving from one part of a cognitive "
            "test to a different kind of task. Reply with a short, natural spoken "
            "cue (roughly 5-12 words) telling the patient you're moving on to "
            "something a little different, without saying what it is, without "
            "naming any clinical domain or task (never say memory, attention, "
            "language, etc.), and without asking a question. "
            "Your reply is always a statement, never ends with '?'. Do not use quotes. "
            "Vary your wording each time -- do not default to the same stock phrase, "
            "\n\n"
            "Examples:\n"
            "-> Okay, now we'll try something a little different.\n"
            "-> Let's move on to something else now.\n"
            "-> Alright, let's try a different kind of task.\n"
            "-> Right, on to the next part.\n"
            "-> Good, let's carry on with something new.\n"
        ))
    ])
    return result.content.strip().strip('"')


def transition_que(state, patient_name: str, domain: str, task_type: str, sub_index: int) -> str:
    """Transitional que when changing domains or achknowledgement of last question"""
    if not state["messages"]:
        return introduce(patient_name)

    if sub_index != 0:
        return ""

    previous_signature = state.get("previous_task_signature")
    """Transition Task to Paper or click on screen"""
    if previous_signature is not None and previous_signature != (domain, task_type):
        return transition()
    if domain == "Visuospatial" and task_type == "click" and previous_signature != (domain, task_type):
        return transition()

    last_patient = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )
    if last_patient:
        return acknowledge(last_patient)
    return ""

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

def is_finished_drawing(response: str) -> bool:
    """True if the patient's utterance means they're finished and ready to
    show their drawing, as opposed to anything else (still working,
    describing progress, a question, an unrelated remark, not ready yet).
    Used during the drawing-task capture wait, where the patient hasn't been
    asked a specific question """
    out = llm_strict.invoke([
        SystemMessage(content=(
            "The patient is doing a drawing task and has not been asked a specific "
            "question. Decide whether their utterance means they are FINISHED and "
            "ready to show their drawing (e.g. 'I'm done', 'okay, finished', 'you can "
            "look now', 'ready') versus anything else (still working, describing what "
            "they're drawing, a question, an unrelated remark, explicitly not ready yet). "
            "Reply with EXACTLY one word: yes or no.\n\n"
            "Examples:\n"
            "Patient: I'm done.\n-> yes\n"
            "Patient: Okay, I've finished.\n-> yes\n"
            "Patient: You can look now.\n-> yes\n"
            "Patient: Not yet, one more minute.\n-> no\n"
            "Patient: I'm drawing the numbers now.\n-> no\n"
            "Patient: How many numbers do I need again?\n-> no\n"
        )),
        HumanMessage(content=f"Patient said: {response}")
    ])
    return out.content.strip().lower().startswith("yes")


def check_in() -> str:
    """Short spoken check-in during a drawing task, asking whether the patient
    is finished and ready to show their work."""
    result = llm_warm.invoke([
        SystemMessage(content=(
            "You are a warm clinical assessor. The patient is in the middle of a drawing "
            "task and has not been asked a specific question. Reply with a short, natural "
            "spoken check-in (roughly 3-8 words) asking whether they're finished and ready "
            "to show their drawing. Do not name the task or describe what they're drawing. "
            "Do not use quotes. Vary your wording each time -- do not default to the same "
            "stock phrase, especially if they were just checked in on.\n\n"
            "Examples:\n"
            "-> Are you ready to show me?\n"
            "-> Tell me when you're done.\n"
            "-> How are you getting on?\n"
            "-> Let me know when you're finished.\n"
        ))
    ])
    return result.content.strip().strip('"')

# Patient has given an Answer
# Patient Needs a Repeat
# Patient is Off Topic
# Patient's Answer is Incomplete
labels = {"answer", "repeat", "off_topic", "incomplete"}

def classify_turn(last_response: str, question_text: str = "") -> str:
    out = llm_strict.invoke([
        SystemMessage(content=(
            "Classify a patient's speach during a cognitive test, given the question "
            "they were just asked. Reply with EXACTLY one of these words, nothing else: "
            "answer, repeat, off_topic, incomplete.\n"
            "- answer: a genuine attempt at THIS question, right or wrong, however short. "
            " replies (a single number, word, name, ordinal, or 'the first') are "
            "complete answers. Repeating the same word in context thats relivent to the question is an answer. "
            "If a real answer value appears anywhere in the utterance, it counts as answer even "
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
            "Q: What letter is this\nPatient: No..\n-->repeat"
        )),
        HumanMessage(content=f"Question asked: {question_text}\nPatient said: {last_response}")
    ])

    content = out.content.strip().lower()
    matches = re.findall(r"\b(" + "|".join(labels) + r")\b", content)
    if matches:
        return matches[-1]
    return "repeat"

def extract_final_answer(last_response: str, question_text: str = "") -> str:
    """Resolves a patient's utterance down to the single value they actually
    settled on, for questions later matched by fuzzy string comparison
    (orientation day/date/month/season, recognition multiple-choice,
    picture-comprehension choices).."""
    out = llm_strict.invoke([
        SystemMessage(content=(
            "The patient was asked a question with one correct value in mind. Reply with "
            "just the single value they actually settled on as their answer, nothing else "
            "(no explanation, no punctuation beyond what's in the value itself, and no "
            "leading article like 'the'/'a'/'an'). "
            "If they corrected themselves, use the corrected value, not the discarded one. "
            "If they mentioned another value only in passing (e.g. explaining or reasoning "
            "about their answer), ignore it and keep their actually stated answer. "
            "If they never commit to a single value and instead list out every plausible "
            "option, reply with nothing.\n\n"
            "Examples:\n"
            "Q: Which county are we in?\nPatient: The county is Kent, oh wait, I'm in Essex.\n-> Essex\n"
            "Q: What day is it?\nPatient: Tuesday, ah well, my family come today.\n-> Tuesday\n"
            "Q: What day is it?\nPatient: Monday, Tuesday, Wednesday, Thursday, Friday.\n-> \n"
            "Q: What day is it?\nPatient: It's Tuesday today, as yesterday was Monday.\n-> Tuesday\n"
            "Q: Was the county Devon, Dorset, or Somerset?\nPatient: Dorset. No, actually, Devon.\n-> Devon\n"
            "Q: Which picture is associated with the monarchy?\nPatient: It's not the penguin, it's the crown.\n-> crown\n"
        )),
        HumanMessage(content=f"Question asked: {question_text}\nPatient said: {last_response}")
    ])
    return out.content.strip().strip('"')

