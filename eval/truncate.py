import sys
from transformers import AutoTokenizer

def main():
    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")

    # Tokenize with truncation (but no padding), and limit length
    max_tokens = int(sys.argv[2])
    tokens = tokenizer(sys.argv[1], truncation=True, max_length=max_tokens, add_special_tokens=False)

    # Decode back into text
    truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    print(truncated_text)

if __name__=="__main__":
    main()