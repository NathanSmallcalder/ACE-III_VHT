import re
from word2number import w2n

ordinals = {
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

number_words = set(w2n.american_number_system.keys())

# e.g Converts 85 into eighty-five
def normalise_number(text):
    text = " ".join(ordinals.get(w, w) for w in text.split())
    words = text.split()
    if not all(w.isdigit() or w in number_words or w == "and" for w in words):
        return text
    if not any(w.isdigit() or w in number_words for w in words):
        return text
    try:
        return str(w2n.word_to_num(text))
    except (ValueError, IndexError):
        return text
