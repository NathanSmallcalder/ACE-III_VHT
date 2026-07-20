import atexit

import spacy
from dotenv import load_dotenv

from LLM.LLM import llm_strict
from LLM.vlm import build_client, describe_images, save_vlm_response

load_dotenv()

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

_GRAMMAR_PROMPT = (
    'You are checking a single sentence written by a patient during a cognitive '
    'assessment for grammar or spelling errors.\n\n'
    'Sentence: "{sentence}"\n\n'
    'Does this sentence contain any grammar or spelling error (including a '
    'wrong-word/homophone mistake)? Respond with exactly one word: YES or NO.'
)


def extract_verbs(sentence):
    doc = nlp(sentence)
    verbs = []

    for token in doc:
        if token.pos_ == "VERB":
            simple_verb = token.text
            verb_modifiers = [t for t in token.lefts if t.dep_ in ("aux", "auxpass", "neg")]
            verb_phrase = "".join([t.text_with_ws for t in verb_modifiers]) + simple_verb
            verbs.append(verb_phrase.strip())
    return verbs

def count_sentences(text):
    doc = nlp(text)
    return len(list(doc.sents))

def extract_subject(sentence):
    doc = nlp(sentence)
    subjects = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            simple_subject = token.text
            subject_phrase = "".join([t.text_with_ws for t in token.lefts]).strip() + " " + simple_subject
            subjects.append(subject_phrase.strip())

    return subjects

def sentence_has_error_llm(sentence_text):
    response = llm_strict.invoke(_GRAMMAR_PROMPT.format(sentence=sentence_text))
    return response.content.strip().upper().startswith("Y")

def classify_sentences(text):
    """Split text into sentences and, for each, report whether it has a
    subject+verb (i.e. counts as a sentence rather than a fragment) and
    whether it contains any grammar/spelling errors. Error detection is
    LLM-judged per sentence """
    doc = nlp(text)

    classified = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)

        classified.append({
            "text": sent_text,
            "is_valid_sentence": has_subject and has_verb,
            "has_errors": sentence_has_error_llm(sent_text),
        })

    return classified

def score_sentence_writing(text):
    sentences = classify_sentences(text)
    valid_sentences = [s for s in sentences if s["is_valid_sentence"]]
    clean_sentences = [s for s in valid_sentences if not s["has_errors"]]

    if len(valid_sentences) >= 2 and len(clean_sentences) == len(valid_sentences):
        return 2
    if len(valid_sentences) >= 2:
        return 1
    if len(valid_sentences) == 1 and len(clean_sentences) == 1:
        return 1
    return 0


_TRANSCRIBE_PROMPT = """You are transcribing handwritten text from a photo, for a cognitive assessment.

Transcribe exactly what is written, as plain text, preserving sentence boundaries and
punctuation. If a word is illegible, write [illegible] in its place. Do not correct
spelling or grammar — transcribe it as written.

Respond with a single JSON object and nothing else — no preamble, no explanation, no markdown fences.

{
  "transcription": "<the transcribed text>"
}
"""

_transcribe_llm = build_client(0.0, 4000)


def transcribe_writing(image_path: str) -> str:
    """Send the photographed writing sample to the VLM and return its plain-text
    transcription. Returns "" on any connection failure or malformed response."""
    data = describe_images(_transcribe_llm, _TRANSCRIBE_PROMPT, [image_path])
    return data.get("transcription", "")


def score_writing_image(image_path: str) -> dict:
    """Transcribe the photographed writing sample via VLM, then score it against
    the ACE-III sentence-writing criteria."""
    text = transcribe_writing(image_path)
    total = score_sentence_writing(text) if text else 0
    result = {"total": total}
    save_vlm_response("writing", image_path, {"transcription": text}, result)
    return result

