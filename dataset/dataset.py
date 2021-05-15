"""
@Time : 2021/4/158:10
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from utils.finetuning_argparse import get_argparse
from utils.utils import logger

max_len = 512


class DuEEEventDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, args, data_path, tag_path, tokenizer):
        # 加载id2entity
        self.label_vocab = {}
        examples = []
        tokenized_examples = []

        for line in open(tag_path, 'r', encoding='utf-8'):
            value, key = line.strip('\n').split('\t')
            self.label_vocab[key] = int(value)


        # 加载准备好的训练集
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                examples.append([words, labels])
        self.label_num = len(self.label_vocab)

        with tqdm(enumerate(examples), total=len(examples), desc="tokenizing...") as pbar:
            for i, example in pbar:
                tokenized_example = tokenizer.encode_plus(
                    example[0],
                    padding="max_length",
                    max_length=args.max_len,
                    is_split_into_words=True,
                    truncation=True
                )
                # 把 label 补齐
                labels = example[1]
                pad_len = args.max_len - 2 - len(labels)
                if pad_len >= 0:
                    labels += ["PAD"] * pad_len
                else:
                    labels = labels[:pad_len]
                labels = ["PAD"] + labels + ["PAD"]
                labels = [self.label_vocab.get(label, -1) for label in labels]

                tokenized_example["labels"] = labels
                tokenized_examples.append(tokenized_example)

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


class DuEEClsDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, args, data_path, tag_path, tokenizer):
        # 加载id2entity
        self.label_vocab = {}
        examples = []
        tokenized_examples = []

        for line in open(tag_path, 'r', encoding='utf-8'):
            value, key = line.strip('\n').split('\t')
            self.label_vocab[key] = int(value)
        df = pd.read_csv(data_path, delimiter="\t", quoting=3)
        examples = df.values.tolist()

        with tqdm(enumerate(examples), total=len(examples), desc="tokenizing...") as pbar:
            for i, example in pbar:
                tokenized_example = tokenizer.encode_plus(
                    example[1],
                    padding="max_length",
                    max_length=args.max_len,
                    truncation=True
                )
                # 把 label 补齐
                labels = self.label_vocab.get(example[0], -1)

                tokenized_example["labels"] = labels
                tokenized_examples.append(tokenized_example)
        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]


class DuEEJointDataset(Dataset):
    """DuEventExtraction"""

    def __init__(self, args, tagger_data_path, role_data_path, tagger_tag_path, role_tag_path, tokenizer):
        self.tagger_dataset = DuEEEventDataset(args,
                                               tagger_data_path,
                                               tagger_tag_path,
                                               tokenizer)
        self.role_dataset = DuEEEventDataset(args,
                                             role_data_path,
                                             role_tag_path,
                                             tokenizer)

        self.algt_tagger = []
        self.algt_role = []

        self.alignment()

        assert len(self.algt_tagger) == len(self.algt_role), '数据长度不一致'

    def alignment(self):
        tag_idx = 0
        role_idx = 0
        while tag_idx < len(self.tagger_dataset) and role_idx < len(self.role_dataset):
            tag_token_ids = self.tagger_dataset[tag_idx]['input_ids']
            role_token_idx = self.role_dataset[role_idx]['input_ids']

            if tag_token_ids == role_token_idx:
                self.algt_tagger.append(self.tagger_dataset[tag_idx])
                self.algt_role.append(self.role_dataset[role_idx])

                assert self.tagger_dataset[tag_idx]['attention_mask'] == self.role_dataset[role_idx][
                    'attention_mask'], 'atten_mask 不一致'

                tag_idx += 1
                role_idx += 1
            else:
                tag_idx -= 1

        return len(self.role_dataset)

    def __len__(self):
        return len(self.algt_role)

    def __getitem__(self, index):
        tragger = self.algt_tagger[index]
        role = self.algt_role[index]

        joint_data = {}
        joint_data['input_ids'] = tragger['input_ids']
        joint_data['token_type_ids'] = tragger['token_type_ids']
        joint_data['attention_mask'] = tragger['attention_mask']
        joint_data['tagger_labels'] = tragger['labels']
        joint_data['role_labels'] = role['labels']

        return joint_data


def joint_collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_tagger_labels = torch.tensor([x["tagger_labels"][:max_len] for x in batch])
    all_role_labels = torch.tensor([x["role_labels"][:max_len] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_tagger_labels": all_tagger_labels,
        "all_role_labels": all_role_labels,
        "all_seq_lens": seq_lens
    }


def collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"][:max_len] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
        "all_seq_lens": seq_lens
    }

def cls_collate_fn(batch):
    max_len = max([sum(x['attention_mask']) for x in batch])
    seq_lens = torch.tensor([sum(x["attention_mask"]) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
        "all_seq_lens": seq_lens
    }


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")
    dataset = DuEEEventDataset(args,
                           data_path="../data/DuEE-Fin/trigger/train.tsv",
                           tag_path="../conf/DuEE-Fin/trigger_tag.dict",
                           tokenizer=tokenizer)
    test_iter = DataLoader(dataset,
                           shuffle=False,
                           batch_size=10,
                           collate_fn=collate_fn,
                           num_workers=20)
    for index, batch in enumerate(test_iter):
        print(index)
