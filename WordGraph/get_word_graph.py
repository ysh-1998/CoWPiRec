import argparse
import os
from scipy.io import mmread,mmwrite
from transformers import BertConfig,BertTokenizer
from tqdm import tqdm
import json
import math
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Food')
    parser.add_argument('--data_path', type=str, default="./dataset")
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--topN', type=int, default=30)
    return parser.parse_args()
args = parse_args()
bert_config = BertConfig.from_pretrained(args.plm)
tokenizer = BertTokenizer.from_pretrained(args.plm)
vocab_size = bert_config.vocab_size
print("load word graph ...")
WG = mmread(os.path.join(args.data_path, args.dataset, f'{args.dataset}.wordgraph.mtx'))
print("get raws and cols ...")
WG = WG.tolil()
rows = WG.rows
cols = WG.T.rows
word_count = 0
# get word df
df_dict = {}
with open(os.path.join(args.data_path, args.dataset, f'{args.dataset}.df.tsv'),'w') as outf:
    outf.write(f"word\tdf\n")
    for word_id in tqdm(range(len(cols)),desc="get df"):
        doc_list = cols[word_id]
        df = len(doc_list)
        df_dict[word_id] = df
        outf.write(f"{tokenizer.convert_ids_to_tokens(word_id)}\t{df}\n")
        if df != 0:
            word_count += 1
# get word idf
idf_dict = {}
with open(os.path.join(args.data_path, args.dataset, f'{args.dataset}.idf.tsv'),'w') as outf:
    outf.write(f"word\tidf\n")
    for word_id, df in df_dict.items():
        if df != 0:
            idf = math.log(word_count / df,10)
            idf_dict[word_id] = idf
            outf.write(f"{tokenizer.convert_ids_to_tokens(word_id)}\t{idf}\n")
# filter coclick word
wg_file = open(os.path.join(args.data_path, args.dataset, f'{args.dataset}.word_graph.jsonl'),"w")
with open(os.path.join(args.data_path, args.dataset, f'{args.dataset}.top_coclick_words.tsv'),'w') as outf:
    outf.write("word\tcoclick_word(tf_idf)\n")
    for word_id in tqdm(range(len(rows)),desc="get tf-idf"):
        word_list = rows[word_id]
        word = tokenizer.convert_ids_to_tokens(word_id)
        outf.write(f"{word}\n")
        if len(word_list) == 0:
            continue
        tf_idf_dict = {}
        tf_sum = WG[word_id].sum()
        for coclick_word_id in word_list:
            if coclick_word_id == word_id:
                continue
            tf = WG[word_id,coclick_word_id] / tf_sum
            idf = idf_dict[coclick_word_id]
            tf_idf = tf * idf
            tf_idf_dict[coclick_word_id] = tf_idf
        tf_idf_dict = sorted(tf_idf_dict.items(), key=lambda item: item[1], reverse=True)[:30]
        coclick_word_id_list = []
        for coclick_word_id, tf_idf in tf_idf_dict:
            coclick_word = tokenizer.convert_ids_to_tokens(coclick_word_id)
            outf.write(f"{coclick_word}\t{round(tf_idf,4)}\n")
            coclick_word_id_list.append(coclick_word_id)
        wg_file.write(json.dumps(
            {"id":word_id, "ids":coclick_word_id_list}
        ))
        wg_file.write("\n")