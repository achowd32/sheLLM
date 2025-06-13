import sys
import torch
from bert_score import score

# get strings
candidate = sys.argv[1] 
reference = sys.argv[2] 

# compute BERTscore
P, R, F1 = score([candidate], [reference], lang="en", verbose=False)

# Print results
print(f"BERTScore Precision: {P[0]:.4f}")
print(f"BERTScore Recall   : {R[0]:.4f}")
print(f"BERTScore F1       : {F1[0]:.4f}")
