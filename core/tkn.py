import sys
import json
from transformers import AutoTokenizer, logging

def main():
    tokenizer = AutoTokenizer.from_pretrained("../model")
    while True:
        text = sys.stdin.read()
        if not text:
            break
        tokens = tokenizer(text) #TODO: customize this
        token_json = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        print(json.dumps(token_json), flush=True)

if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)
    main()