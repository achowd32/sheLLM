import sys
from datasets import load_dataset

def init_data(numfiles):
    # load dataset
    dataset = load_dataset("allenai/c4", "en", streaming = True)

    for i, sample in enumerate(dataset['train']):
        if i >= numfiles:
            break
        print(sample['text'], flush = True)
    
if __name__ == "__main__":
    numfiles = int(sys.argv[1])
    init_data(numfiles)