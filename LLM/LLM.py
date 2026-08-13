# LLM/LLM.py
from langchain_openai import ChatOpenAI

base_url = "http://localhost:1234/v1"
api_key  = "lm-studio"
model    = "google/gemma-4-e4b"

def _build(temperature: float, max_tokens: int) -> ChatOpenAI:
    """
    Bulds the LLM - Currently Gemma-4-e4b
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

llm_strict = _build(temperature=0.0, max_tokens=2000)  # classify_turn — deterministic
llm_warm   = _build(temperature=0.7, max_tokens=2000)   # introduce, soften_repeat — natural