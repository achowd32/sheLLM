import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    # load model and tokenizer
    load_path = "model"
    model = GPT2LMHeadModel.from_pretrained(load_path)
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    # tokenize input and convert to tensor
    tokens = tokenizer(sys.argv[1], return_tensors="pt")

    # generate text
    output = model.generate(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        max_length = 100,
        temperature=0.5,
        do_sample = True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print output
    print(generated_text + "\n" + "------------------" + "\n")

if __name__ == "__main__":
    main()