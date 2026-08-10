# LLM/LLM.py
from langchain_openai import ChatOpenAI

BASE_URL = "http://localhost:1234/v1"
API_KEY  = "lm-studio"
MODEL    = "google/gemma-4-e4b"

"""
Bulds the LLM
"""
def _build(temperature: float, max_tokens: int) -> ChatOpenAI:
    """
    Bulds the LLM - Currently Gemma-4-e4b
    """
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

llm_strict = _build(temperature=0.0, max_tokens=2000)  # classify_turn — deterministic
llm_warm   = _build(temperature=0.7, max_tokens=2000)   # introduce, soften_repeat — natural