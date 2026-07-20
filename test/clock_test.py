import os

import pytest

import visual_tasks.clock_scorer as clock_scorer
import visual_tasks.infinity_scorer as infinity_scorer
import visual_tasks.cube_scorer as cube_scorer
CLOCKS_DIR = os.path.join(os.path.dirname(__file__), "Clocks")
INFINITY_DIR = os.path.join(os.path.dirname(__file__), "InfinitySymbol")
CUBE_DIR = os.path.join(os.path.dirname(__file__), "cube")

def _collect_cases(DIR):
    """Each subfolder of Clocks/ is named after the expected total score
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