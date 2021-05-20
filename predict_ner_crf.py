"""
@Time : 2021/4/158:09
@Auth : 周俊贤
@File ：run_ner.py.py
@DESCRIPTION:
"""
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from dataset.dataset import collate_fn, DuEEEventDataset
from model.model import DuEEEvent_model,DuEEEvent_crf_model
from utils.finetuning_argparse import get_argparse
from utils.utils import init_logger, seed_everything, logger, read_by_lines, write_by_lines

def main():
    parser = get_argparse()
    parser.add_argument("--fine_tunning_model_path",
                        type=str,
                        required=True,
                        help="fine_tuning model path")
    parser.add_argument("--test_json",
                        type=str,
                        required=True,
                        help="test json path")
    args = parser.parse_args()

    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))
    seed_everything(args.seed)

    args.output_model_path = os.path.join(args.output_dir, args.dataset, args.event_type, "best_model.pkl")
    # 设置保存目录
    if not os.path.exists(os.path.dirname(args.output_model_path)):
        os.makedirs(os.path.dirname(args.output_model_path))

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # dataset & dataloader
    args.test_data = "./data/{}/{}/test.tsv".format(args.dataset, args.event_type)
    args.tag_path = "./conf/{}/{}_tag.dict".format(args.dataset, args.event_type)
    test_dataset = DuEEEventDataset(args,
                                   args.test_data,
                                   args.tag_path,
                                   tokenizer)
    logger.info("The nums of the test_dataset features is {}".format(len(test_dataset)))
    test_iter = DataLoader(test_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=10)

    # load data from predict file
    sentences = read_by_lines(args.test_json)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    # 用于evaluate
    args.label2it = test_dataset.label_vocab
    args.id2label = {val: key for key, val in args.label2it.items()}
    args.num_classes = len(args.id2label)

    #
    model = DuEEEvent_crf_model(args.model_name_or_path, num_classes=args.num_classes)
    model.to(args.device)
    model.load_state_dict(torch.load(args.fine_tunning_model_path))

    results = []
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            logits = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )


            probs_ids = model.crf.decode(logits)
            seq_lens = batch["all_seq_lens"]
            for  p_ids, seq_len in zip( probs_ids, seq_lens.tolist()):
                label_one = [args.id2label[pid] for pid in p_ids[1: seq_len - 1]]
                results.append({"labels": label_one})
    print(len(results))
    print(len(sentences))
    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    args.predict_save_path = os.path.join("./output", args.dataset, args.event_type, "test_result.json")
    print("saving data {} to {}".format(len(sentences), args.predict_save_path))
    write_by_lines(args.predict_save_path, sentences)


if __name__ == '__main__':
    main()
