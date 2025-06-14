import sys
import torch
from bert_score import score

# get strings
candidate = sys.argv[1] 
reference = sys.argv[2] 

# compute BERTscore
P, R, F1 = score([candidate], [reference], lang="en", verbose=False)

# print precision, recall, and f1 
print(f"BERTScores: {P[0]:.4f} {R[0]:.4f} {F1[0]:.4f}")
