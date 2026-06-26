import re
from word2number import w2n

def clean_response(text):
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Converts 85 into eighty-five
def normalise_number(text):
    try:
        return str(w2n.word_to_num(text))
    except ValueError:
        return text
    
def tokenize(text):
    return clean_response(text).split()
