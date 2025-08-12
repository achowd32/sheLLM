import sys
import logging
from bert_score import score

# ignore warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# get strings
candidate = sys.argv[1] 
reference = sys.argv[2] 

# compute BERTscore
P, R, F1 = score(
    [candidate],
    [reference],
    lang="en", 
    rescale_with_baseline=True, # more readable scores
    verbose=False,
    )

# print f1 
print(f"{F1[0]:.4f}")
