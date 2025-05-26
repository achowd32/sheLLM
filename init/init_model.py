import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def init_model():
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # define model configuration
    config = {"vocab_size": len(tokenizer)}

    # initialize the model
    config = AutoConfig.from_pretrained('gpt2', **config)
    model = AutoModelForCausalLM.from_config(config)

    # save model
    save_dir = "../model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    init_model()