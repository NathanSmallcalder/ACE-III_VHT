import base64
import io
import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

BASE_URL = "http://localhost:1234/v1"
API_KEY  = "lm-studio"
MODEL    = "qwen/qwen3-vl-4b"
_BACKEND = "lm_studio"  # "lm_studio" | "gemini" 

VLM_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "vlm_responses")

MIN_DIMENSION = 1024  #  upscale before sending to the VLM

def build_client(temperature: float, max_tokens: int):
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

def encode_image(image_path: str) -> str:
    """Encodes an image, upscaling it first if its smaller dimension is below
    MIN_DIMENSION so the VLM has more pixels to read fine detail (digits, hand tips) from."""
    img = Image.open(image_path)
    scale = MIN_DIMENSION / min(img.size)
    if scale > 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"

def describe_images(client: ChatOpenAI, prompt: str, image_paths: list[str]) -> dict:
    """Send `prompt` plus one or more images to `client` and parse its structured
    JSON reply. """
    try:
        response = client.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": {"url": encode_image(p)}} for p in image_paths],
            ])
        ])
    except Exception as e:
        print(f"describe_images failed: {e}")
        return {}

    raw = response.content
    if isinstance(raw, list):
        raw = raw[0]["text"]

    # Strip markdown fences if the model ignores instructions
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}

def save_vlm_response(task: str, image_path: str, raw_response: dict, score: dict) -> str:
    """
    Used to audit the VLMs response. Saves the VLM response and determinstic score to a JSON file in VLM_LOG_DIR.
    """
    os.makedirs(VLM_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(VLM_LOG_DIR, f"{task}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "raw_response": raw_response,
            "score": score,
        }, f, indent=2)
    return out_path
