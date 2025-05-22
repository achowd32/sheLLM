import sys
import json

MAX_TOKENS = 128

def main():
    for line in sys.stdin:
        # get tokens from stdin
        tokens = json.loads(line)

        for i in range(0, len(tokens["input_ids"]), MAX_TOKENS):
            #define chunks
            new_ids = [tokens["input_ids"][i:i+MAX_TOKENS]]
            new_mask = [tokens["attention_mask"][i:i+MAX_TOKENS]]
            new_labels = [tokens["input_ids"][i:i+MAX_TOKENS][1:] + [-100]]

            #correctly format last chunk
            pad_len = MAX_TOKENS - len(new_ids[0])
            if pad_len > 0:
                new_ids[0] += [50256] * pad_len
                new_mask[0] += [0] * pad_len
                new_labels[0] += [-100] * pad_len

            #correctly format output and return
            output = {
                "input_ids": new_ids,
                "attention_mask": new_mask,
                "labels": new_labels
            }
            print(json.dumps(output), flush=True)

if __name__ == "__main__":
    main()