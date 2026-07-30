import os

import pytest
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

import visual_tasks.clock_scorer as clock_scorer
import visual_tasks.infinity_scorer as infinity_scorer
import visual_tasks.cube_scorer as cube_scorer

load_dotenv()

CLOCKS_DIR = os.path.join(os.path.dirname(__file__), "Clocks")
INFINITY_DIR = os.path.join(os.path.dirname(__file__), "InfinitySymbol")
CUBE_DIR = os.path.join(os.path.dirname(__file__), "cube")

CLAUDE_MODEL = "claude-opus-5"

@pytest.fixture(autouse=True, scope="module")
def use_claude_vlm():
    client = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=os.environ["CLAUDE"],
        max_tokens=8000,
        thinking={"type": "adaptive"},
    )
    originals = {
        clock_scorer: clock_scorer.llm,
        cube_scorer: cube_scorer.llm,
        infinity_scorer: infinity_scorer.llm,
    }
    for module in originals:
        module.llm = client
    yield
    for module, original_llm in originals.items():
        module.llm = original_llm

def _collect_cases(DIR):
    """Each subfolder of DIR is named after the expected total score
    (e.g. Clocks/3/ contains clock images that should score 3)."""
    cases = []
    for subfolder in os.listdir(DIR):
        expected_score = int(subfolder)
        subfolder_path = os.path.join(DIR, subfolder)
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            cases.append(pytest.param(image_path, expected_score, id=f"{subfolder}/{filename}"))
    return cases

@pytest.mark.parametrize("image_path,expected_score", _collect_cases(CLOCKS_DIR))
def test_clock_score(image_path, expected_score):
    result = clock_scorer.score_clock_image(image_path)
    assert result["total"] == expected_score

@pytest.mark.parametrize("image_path,expected_score", _collect_cases(CUBE_DIR))
def test_cube_score(image_path, expected_score):
    result = cube_scorer.score_cube_image(image_path)
    assert result["total"] == expected_score

@pytest.mark.parametrize("image_path,expected_score", _collect_cases(INFINITY_DIR))
def test_infinity_score(image_path, expected_score):
    result = infinity_scorer.score_infinity_image(image_path)
    assert result["total"] == expected_score