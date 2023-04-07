import os
import torch
import logging
import argparse
import random
import json
from tqdm import tqdm
import numpy as np
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys
sys.path.append('../')

from models import build_or_load_gen_model
from utils import SimpleIntClsDataset, calculate_topk_precision, calculate_non_others_top1_precision, calculate_metrics_based_on_confidence_threshold, match_and_add_predictions_to_test_data
from configs import add_args, set_seed, set_dist


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    intent_dict = json.load(open(args.intent_dict_file))
    dataset = SimpleIntClsDataset(tokenizer, pool, args, data_file, intent_dict)
    data_len = len(dataset)
    logger.info(f"Data length: {data_len}.")
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.cpu_count, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_acc(args, eval_dataloader, model, tokenizer):
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    pred, gold, pred_score, data_idx = [], [], [], []
    all_scores = np.empty((0, args.num_cls), float)
    with torch.no_grad():
        for step, examples in enumerate(tqdm(eval_dataloader), 1):
            source_ids = torch.tensor(
                [ex.source_ids for ex in examples], dtype=torch.long
            ).to(args.local_rank)
            source_mask = source_ids.ne(tokenizer.pad_id)
            logits = model(
                cls=True,
                input_ids=source_ids,
                labels=None,
                attention_mask=source_mask
            )
            prediction = torch.argmax(logits, dim=-1).cpu().numpy()
            pred.extend(prediction)
            gold.extend([ex.y for ex in examples])
            data_idx.extend([ex.example_id for ex in examples])

            normalized_score = torch.nn.functional.softmax(logits, dim=-1)
            highest_score = torch.max(normalized_score, dim=-1)[0].cpu().numpy()
            pred_score.extend(highest_score)
            all_scores = np.append(all_scores, normalized_score.cpu().numpy(), axis=0)
    logger.info("\n" + classification_report(gold, pred, digits=4))

    result_dict = {}
    report_dict = classification_report(gold, pred, digits=4, output_dict=True)
    intent_dict = json.load(open(args.intent_dict_file))
    inverse_intent_dict = {str(index): intent for intent, index in intent_dict.items()}
    for index, result in report_dict.items():
        if index in inverse_intent_dict:
            result_dict[inverse_intent_dict[index]] = result
        else:
            result_dict[index] = result

    # calculate top3 and top5 precision
    top3_precision = calculate_topk_precision(gold, all_scores, k=3)
    logger.info("\nTop3 precision: " + str(top3_precision)[:6])
    top5_precision = calculate_topk_precision(gold, all_scores, k=5)
    logger.info("\nTop5 precision: " + str(top5_precision)[:6])
    result_dict['top3_precision'] = top3_precision
    result_dict['top5_precision'] = top5_precision

    # top1 precision for model's non-others prediction
    result_dict['non_others_top1_precision'] = calculate_non_others_top1_precision(gold, pred)

    # coverage: percentage of cases that we predict a non-others intent
    result_dict['coverage'] = np.count_nonzero(pred) * 1.0 / len(pred)

    # recall: percentage of non-others intents that can be correctly found by model
    found_count = 0
    for i in range(len(pred)):
        if gold[i] != 0 and pred[i] == gold[i]:
            found_count += 1
    result_dict['recall'] = found_count * 1.0 / len(pred)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, "test_result.json")
    with open(output_path, 'w') as fp:
        json.dump(result_dict, fp)

    # save confidence scores
    hit_scores, miss_scores, non_others_hit_scores, non_others_miss_scores = [], [], [], []
    for i in range(len(pred_score)):
        if pred[i] == gold[i]:
            hit_scores.append(float(pred_score[i]))
            if pred[i] != 0:
                non_others_hit_scores.append(float(pred_score[i]))
        else:
            miss_scores.append(float(pred_score[i]))
            if pred[i] != 0:
                non_others_miss_scores.append(float(pred_score[i]))
    conf_scores = {"hit": hit_scores, 
                "miss": miss_scores,
                "non_others_hit": non_others_hit_scores,
                "non_others_miss": non_others_miss_scores}
    score_output_path = os.path.join(args.output_dir, "confidence_scores.json")
    with open(score_output_path, 'w') as fp:
        json.dump(conf_scores, fp)

    # metrics if applying different confidence threshold
    thresholds = [x/100 for x in range(20, 100, 5)]
    all_result_dicts = {}
    for thred in thresholds:
        all_result_dicts[str(thred)] = calculate_metrics_based_on_confidence_threshold(gold, all_scores, intent_dict, threshold=thred)
    conf_result_output_path = os.path.join(args.output_dir, "test_result_by_threshold.json")
    with open(conf_result_output_path, 'w') as fp:
        json.dump(all_result_dicts, fp)

    # output predictions for test data
    output_test_data_file = os.path.join(args.output_dir, "msg-test-pred.jsonl")
    match_and_add_predictions_to_test_data(args, data_idx, pred, args.eval_file, output_test_data_file)
    

def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.eval_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)        # WARNING: this is a iterator, to save memory
    model.eval()
    eval_epoch_acc(args, dataloader, model, tokenizer)


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    if args.log_file is not None:
        log_path = "/".join(args.log_file.split("/")[:-1])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        fh = logging.FileHandler(args.log_file, encoding='utf-8')
        logger.addHandler(fh)

    logger.info(args)
    main(args)
    logger.info("Test finished in {} mins.\n".format((time.time() - start_time)/60))
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())

    os.system("kill $(ps aux | grep run_test_intcls.py | grep -v grep | awk '{print $2}') ")
