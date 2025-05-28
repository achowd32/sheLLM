import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # load model and tokenizer
    load_path = "../model"
    model = AutoModelForCausalLM.from_pretrained(load_path)
    tokenizer = AutoTokenizer.from_pretrained(load_path)

    # tokenize input and convert to tensor
    tokens = tokenizer(sys.argv[1], return_tensors="pt")

    # generate text
    model.eval()
    output = model.generate(
        input_ids=tokens.input_ids[:1024],
        attention_mask=tokens.attention_mask[:1024],
        max_new_tokens = 100,
        temperature=0.5,
        do_sample = True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print output
    print(generated_text)

if __name__ == "__main__":
    main()