import sys
from datasets import load_dataset

def main():
    # load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation")

    data_split = sys.argv[1]
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        lines_removed = sample[data_split].replace("\n", " ")
        truncated = lines_removed[:1024]
        print(truncated, flush = True)
    
if __name__ == "__main__":
    main()
    sys.exit(0)