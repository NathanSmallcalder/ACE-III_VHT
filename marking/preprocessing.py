import re
from word2number import w2n

_ORDINAL_WORDS = {
    "first": "one", "second": "two", "third": "three", "fourth": "four",
    "fifth": "five", "sixth": "six", "seventh": "seven", "eighth": "eight",
    "ninth": "nine", "tenth": "ten", "eleventh": "eleven", "twelfth": "twelve",
    "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
    "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen",
    "nineteenth": "nineteen", "twentieth": "twenty", "thirtieth": "thirty",
}

def clean_response(text):
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

_NUMBER_WORDS = set(w2n.american_number_system.keys())

# e.g Converts 85 into eighty-five
def normalise_number(text):
    text = " ".join(_ORDINAL_WORDS.get(w, w) for w in text.split())

    # "and" isn't itself a number word, but word2number already parses it fine
    # as a connector ("two thousand and twenty six") — only the membership
    # check below needs to tolerate it, not the actual conversion call. Still
    # require at least one real number word so a stray "and" alone (or "and
    # and") never reaches word2number, which throws non-ValueError errors
    # (e.g. IndexError) on that kind of malformed input rather than failing cleanly.
    words = text.split()
    if not all(w.isdigit() or w in _NUMBER_WORDS or w == "and" for w in words):
        return text
    if not any(w.isdigit() or w in _NUMBER_WORDS for w in words):
        return text
    try:
        return str(w2n.word_to_num(text))
    except (ValueError, IndexError):
        return text
