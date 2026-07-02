import re
from word2number import w2n

# word2number only understands cardinal words ("one", "twenty"), not ordinal
# words ("first", "twentieth") — digit-form ordinals ("1st") are already
# handled by clean_response's suffix-stripping regex, but word-form ordinals
# need to be mapped back to their cardinal counterpart before w2n can parse them.
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

# e.g Converts 85 into eighty-five
def normalise_number(text):
    text = " ".join(_ORDINAL_WORDS.get(w, w) for w in text.split())
    try:
        return str(w2n.word_to_num(text))
    except ValueError:
        return text
    
def tokenize(text):
    return clean_response(text).split()
