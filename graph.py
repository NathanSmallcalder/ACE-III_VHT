from langgraph.graph import StateGraph, MessagesState, START, END


class ACEState(MessagesState):
    current_domain: str
    question_index: int
    sub_question_index: int
    question_score: int
    scores: dict
    domain_queue: list
    complete: bool
    needs_repeat: bool


def conversation_node(state: ACEState) -> dict:
    pass


def scoring_node(state: ACEState) -> dict:
    pass


def advance_node(state: ACEState) -> dict:
    pass


def report_node(state: ACEState) -> dict:
    pass


def router(state: ACEState) -> str:
    if state.get("needs_repeat"):
        return "repeat_question"
    # next_sub_question | next_question | next_domain | report
    pass


builder = StateGraph(ACEState)
builder.add_node("conversation", conversation_node)
builder.add_node("scoring", scoring_node)
builder.add_node("advance", advance_node)
builder.add_node("report", report_node)

builder.add_edge(START, "conversation")
builder.add_edge("conversation", "scoring")
builder.add_conditional_edges("scoring", router, {
    "repeat_question": "conversation",
    "next_sub_question": "conversation",
    "next_question": "advance",
    "next_domain": "advance",
    "report": "report",
})
builder.add_edge("advance", "conversation")
builder.add_edge("report", END)

graph = builder.compile()