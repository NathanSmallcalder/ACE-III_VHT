from graph import graph, ACE_DATA

DOMAIN_ORDER = list(ACE_DATA.keys())

initial_state = {
    "messages": [],
    "current_domain": DOMAIN_ORDER[0],
    "question_index": 0,
    "sub_question_index": 0,
    "question_score": 0,
    "scores": {domain: 0 for domain in DOMAIN_ORDER},
    "domain_queue": DOMAIN_ORDER[1:],
    "complete": False,
    "needs_repeat": False,
    "repeat_count": 0,
}

if __name__ == "__main__":
    graph.invoke(initial_state)
