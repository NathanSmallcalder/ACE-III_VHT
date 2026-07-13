import base64
import os
from langchain_openai import ChatOpenAI

# Single source of truth for the local VLM backend connection.
# Used by cube_scorer.py and infinity_scorer.py (clock_scorer.py keeps its own copy).
BASE_URL = "http://localhost:1234/v1"
API_KEY  = "lm-studio"
MODEL    = "google/gemma-4-e4b"

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def build_client(temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )


def encode_image(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = MIME_TYPES.get(ext, "image/png")
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"