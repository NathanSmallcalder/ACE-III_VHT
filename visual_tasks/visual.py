from PIL import Image
from rapidfuzz import fuzz
from langchain_core.messages import AIMessage, HumanMessage

from LLM.dialogue import introduce, acknowledge, rephrase_question
from marking.marking import parse_spoken_prompts


def run_visual_task(state, question: dict, tts, audio, session_config: dict) -> dict:
    spoken = parse_spoken_prompts(question)
    text = spoken[0] if spoken else question["question_text"]

    if state.get("needs_repeat"):
        wrapper = rephrase_question(text)
    elif not state["messages"]:
        wrapper = introduce(session_config["patient"]["name"])
    else:
        last_patient = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None
        )
        wrapper = acknowledge(last_patient) if last_patient else ""

    if question.get("image"):
        Image.open(question["image"]).show()

    spoken_text = f"{wrapper} {text}".strip() if wrapper else text

    print("Assessor:", spoken_text)
    tts.speak(spoken_text)

    for _ in range(5):
        user_input = audio.capture_response()

        if not user_input:
            print("[no response detected]")
            continue

        if fuzz.partial_ratio(user_input.lower(), spoken_text.lower()) > 75:
            print("[echo detected, ignoring]")
            continue

        break
    else:
        return {"needs_repeat": True}

    print("Patient:", user_input)
    return {"messages": [AIMessage(content=spoken_text), HumanMessage(content=user_input)]}
