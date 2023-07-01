import argparse
import collections
import os
from tqdm import tqdm
import pandas as pd
from utils import check_path


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        fp.readline()
        for line in tqdm(fp, desc='Load ratings'):
            user, item, time = line.strip().split(',')
            users.add(user)
            items.add(item)
            inters.add((user, item, int(time)))
    return users, items, inters


def load_meta_items(file):
    df_item = pd.read_csv(file,sep=',')
    items = df_item["item_id"].values.tolist()
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []
    
    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in tqdm(inters):
            if unit[1] in can_items:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, timestamp = inter
        user2inters[user].append((user, item, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[2])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = args.dataset
    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path,"interaction", dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)
    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path,"metadata", f'meta_{dataset_full_name}.csv')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, time = line
        users.add(user)
        items.add(item)
    return users, items


def generate_text(args, items):
    dataset_full_name = args.dataset
    meta_file_path = os.path.join(args.input_path,"metadata", f'meta_{dataset_full_name}.csv')
    item_text_list = []
    with open(meta_file_path,"r",encoding="utf-8") as f:
        for line in f:
            item_id, item_text = line.strip().split(",")
            if item_id in items:
                item_text_list.append([item_id, item_text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, rating_inters):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_user_item_from_ratings(rating_inters)
    
    # load item text and clean
    item_text_list = generate_text(args, rating_items)
    print('\n')

    # return: list of (item_ID, cleaned_item_text)
    return item_text_list


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + \
               len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')

def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-10:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-10:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-10:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default="./dataset/raw_data")
    parser.add_argument('--output_path', type=str, default="./dataset")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    # save useful data
    write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))
