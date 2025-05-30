import sys
from transformers import pipeline, logging, AutoTokenizer

def main():
    # load model and tokenizer
    load_path = "../model"
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    generator = pipeline("text-generation", model=load_path, pad_token_id=tokenizer.eos_token_id)

    # generate text
    output = generator(sys.argv[1], max_new_tokens=100, return_full_text=False)[0]["generated_text"]

    # Print output
    print(output)

if __name__ == "__main__":
    logging.set_verbosity_error()
    main()