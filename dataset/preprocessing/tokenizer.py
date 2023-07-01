from operator import mod
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    first_flag = True
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                desc=f"Tokenize: {os.path.basename(input_file)}"):
            if first_flag:
                first_flag = False
                continue
            line_list = line.strip().split('\t')
            seq_id = line_list[0]
            text = line_list[1]
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            outFile.write(json.dumps(
                {"id":seq_id, "ids":ids}
            ))
            outFile.write("\n")

def tokenize_item_text(args, tokenizer):
    output = f"{args.data_path}/{args.dataset}/{args.dataset}.tokenize.json"
    input = f"{args.data_path}/{args.dataset}/{args.dataset}.text"
    tokenize_file(tokenizer, input, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--data_path", type=str, default="./dataset")
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    tokenize_item_text(args, tokenizer)
