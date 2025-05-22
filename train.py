import sys
import json
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")

def train_model(hf_dataset):
    # load in current model
    load_path = "model"
    model = AutoModelForCausalLM.from_pretrained(load_path)

    # training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints/",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=25,
        save_steps=500,
    )

    # initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset
    )

    # train
    trainer.train()

    # save model
    model.save_pretrained("model")

def main():
    for line in sys.stdin:
        data = json.loads(line)
        hf_dataset = Dataset.from_dict(data)
        train_model(hf_dataset)

if __name__ == "__main__":
    main()