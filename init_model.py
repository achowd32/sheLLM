import os
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Tokenizer

def init_model():
    # define model configuration
    config = {"vocab_size": len(GPT2Tokenizer.from_pretrained("gpt2"))}

    # initialize the model
    config = AutoConfig.from_pretrained('gpt2', **config)
    model = AutoModelForCausalLM.from_config(config)

    # save model
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

if __name__ == "__main__":
    init_model()