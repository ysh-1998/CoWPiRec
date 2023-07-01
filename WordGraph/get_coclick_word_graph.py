import argparse
import os
import scipy.sparse as sp
from scipy.io import mmread,mmwrite
from transformers import BertConfig,BertTokenizer
import numpy as np
from tqdm import tqdm
import json
import pickle
import multiprocessing as mp
import math
from collections import Counter
mp.set_start_method('spawn', True)
import sys
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Food')
    parser.add_argument('--data_path', type=str, default="./dataset")
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--max_len', type=int, default=64)
    return parser.parse_args()
def load_tokenize_json(json_dir,item2index):
    id2token_dict = {}
    with open(json_dir) as f:
        for line in f:
            es = json.loads(line)
            id, ids = es["id"], es["ids"]
            id = item2index[id]
            if id not in id2token_dict:
                id2token_dict[id] = ids
    return id2token_dict
def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index
def get_word_graph(chunk_no,cpus, block_offset, item_pairs_file_path, lines_per_chunk, excess_lines, output_folder,vocab_size,item_word_dict,max_len):
    if os.path.exists(os.path.join(output_folder, f'word_graph-{chunk_no}.mtx')):
        return
    if chunk_no == cpus-1:
        lines_per_chunk = excess_lines
    # init word graph
    WG = sp.dok_matrix((vocab_size, vocab_size), dtype=np.int32)
    position = chunk_no + 1
    pbar = tqdm(total=lines_per_chunk, desc="Running for chunk {}".format(
        str(chunk_no).zfill(2)), position=position, file=sys.stdout)
    with open(item_pairs_file_path, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(lines_per_chunk):
            line = f.readline()
            line_list = line.strip().split("\t")
            item_words_1 = item_word_dict[int(line_list[0])][:max_len]
            item_words_2 = item_word_dict[int(line_list[1])][:max_len]
            counts1 = Counter(item_words_1) # {word: count}
            counts2 = Counter(item_words_2)
            words1, counts1 = zip(*counts1.items())
            words2, counts2 = zip(*counts2.items())
            WG[np.ix_(words1, words2)] += np.outer(counts1, counts2)
            WG[np.ix_(words2, words1)] += np.outer(counts2, counts1)
            # for w1 in item_words_1:
            #     for w2 in item_words_2:
            #         WG[w1,w2] += 1
            #         WG[w2,w1] += 1
            pbar.update()
    pbar.close()
    mmwrite(os.path.join(output_folder, f'word_graph-{chunk_no}.mtx'),WG)
if __name__ == "__main__":
    args = parse_args()
    bert_config = BertConfig.from_pretrained(args.plm)
    tokenizer = BertTokenizer.from_pretrained(args.plm)
    vocab_size = bert_config.vocab_size
    if not os.path.exists(os.path.join(args.data_path, args.dataset, f'{args.dataset}.wordgraph.mtx')):
        # load item word dict
        item2index = load_unit2index(os.path.join(args.data_path, args.dataset, f'{args.dataset}.item2index'))
        item_word_dict = load_tokenize_json(os.path.join(args.data_path, args.dataset, f'{args.dataset}.tokenize.json'),item2index)
        # load coclick item pairs
        item_pairs_file_path = os.path.join(args.data_path, args.dataset, f'{args.dataset}.coclick.items')
        # multiprocessing
        num_lines = sum(1 for _ in open(item_pairs_file_path))
        print(f"{num_lines} coclick item pairs totally")
        cpus = max(1, args.num_workers)
        print("get word graph with %i cpus" % cpus)
        excess_lines = num_lines % cpus
        number_of_chunks = cpus
        if excess_lines > 0:
            number_of_chunks = cpus - 1
            excess_lines = num_lines % number_of_chunks
        lines_per_chunk = num_lines // number_of_chunks
        print(f"{lines_per_chunk} lines per chunk")
        print(f"{excess_lines} lines for last chunk")
        assert (number_of_chunks * lines_per_chunk + excess_lines) == num_lines
        # block offset
        block_offset = dict()
        if cpus < 2:
            block_offset[0] = 0
        else:  # Compute offset for item pairs for each chunk to be processed
            output_file = os.path.join(args.data_path, args.dataset, f'{args.dataset}.blocks.offset.{cpus}.cpus')
            if not os.path.exists(output_file):
                pbar = tqdm(total=num_lines + 1, desc="Computing chunks for each processor",file=sys.stdout)
                with open(item_pairs_file_path) as f:
                    current_chunk = 0
                    counter = 0
                    line = True
                    while(line):
                        if counter % lines_per_chunk == 0:
                            block_offset[current_chunk] = f.tell()
                            current_chunk += 1
                        line = f.readline()
                        pbar.update()
                        counter += 1
                        if counter == num_lines:
                            break
                pbar.close()
                pickle.dump(block_offset, open(output_file, 'wb'))
            else:
                block_offset = pickle.load(open(output_file, 'rb'))
            print(f"block_offset:\n{block_offset}")
        output_folder = os.path.join(args.data_path, args.dataset, "tmp")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        if cpus < 2:  # Single CPU, compute directly.
            get_word_graph(0, cpus, block_offset, item_pairs_file_path, lines_per_chunk,excess_lines, output_folder,vocab_size,item_word_dict,args.max_len)
        else:
            pbar = tqdm(total=cpus, position=0,file=sys.stdout)
            def update(*a):  # Update progress bar
                pbar.update()
            pool = mp.Pool(cpus)
            jobs = []
            for i in range(len(block_offset)):
                jobs.append(pool.apply_async(get_word_graph, args=(
                    i, cpus, block_offset, item_pairs_file_path, lines_per_chunk,excess_lines, output_folder,vocab_size,item_word_dict,args.max_len), callback=update))
            for job in jobs:
                job.get()
            pool.close()
            pbar.close()
        WG = sp.dok_matrix((vocab_size, vocab_size), dtype=np.int32)
        for i in tqdm(range(len(block_offset)),desc="merge graph"):
            WG_tmp = mmread(os.path.join(output_folder, f'word_graph-{i}.mtx'))
            WG = WG + WG_tmp
        mmwrite(os.path.join(args.data_path, args.dataset, f'{args.dataset}.wordgraph.mtx'),WG)