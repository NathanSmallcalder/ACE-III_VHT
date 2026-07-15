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

    if not all(w.isdigit() or w in _NUMBER_WORDS for w in text.split()):
        return text
    try:
        return str(w2n.word_to_num(text))
    except ValueError:
        return text
    
def tokenize(text):
    return clean_response(text).split()
