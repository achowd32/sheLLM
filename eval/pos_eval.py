#!/usr/bin/env python3

import sys
from difflib import SequenceMatcher

import spacy
from spacy.util import is_package
from spacy.cli import download

# download spacy model if it hasn't already been downloaded
if not is_package("en_core_web_sm"):
    download("en_core_web_sm")

# load in spacy model
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
print(f"{score:.4f}")
