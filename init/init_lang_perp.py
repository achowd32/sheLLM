import sys
from datasets import load_dataset

def main():
    # load dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    data_split = sys.argv[1]
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(sample[data_split].replace("\n", " "), flush = True)
    
if __name__ == "__main__":
    main()
    sys.exit(0)