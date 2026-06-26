from metaphone import doublemetaphone
import re
from nltk.corpus import wordnet as wn
import rapidfuzz
from preprocessing import *

def phonetic_equal(a, b):
    pa, sa = doublemetaphone(a)
    pb, sb = doublemetaphone(b)
    if not pa or not pb:
        return False
    codes_a = {c for c in (pa, sa) if c}
    codes_b = {c for c in (pb, sb) if c}
    return bool(codes_a & codes_b)


def score_exact(response, answers):
    """Single-character exact match (fragmented letters). Checks if the expected letter appears as a word in the response."""
    words = clean_response(response).split()
    return 1 if any(clean_response(a) in words for a in answers) else 0

