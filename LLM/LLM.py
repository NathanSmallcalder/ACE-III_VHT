# LLM/LLM.py
from langchain_openai import ChatOpenAI

_BASE_URL = "http://localhost:1234/v1"
_API_KEY  = "lm-studio"
_MODEL    = "google/gemma-4-e4b"

def _build(temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=_BASE_URL,
        api_key=_API_KEY,
        model=_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

llm_strict = _build(temperature=0.0, max_tokens=2000)  # classify_turn — deterministic
llm_warm   = _build(temperature=0.7, max_tokens=2000)   # introduce, soften_repeat — natural