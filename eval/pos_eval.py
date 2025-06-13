import sys
import spacy
from difflib import SequenceMatcher

# load spacy model
nlp = spacy.load("en_core_web_sm")

# load arguments
sample = sys.argv[1]
reference = sys.argv[2]

# function to extract POS tags
def pos_sequence(text):
    doc = nlp(text)
    return [token.pos_ for token in doc]

pos_sample = pos_sequence(sample)
pos_ref = pos_sequence(reference)
matcher = SequenceMatcher(None, pos_sample, pos_ref)
score = matcher.ratio()
print(f"Syntactic similarity: {score:.2f}")
