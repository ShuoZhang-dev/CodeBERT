import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import (
    RobertaTokenizer,
    T5Config,
)
from tqdm import tqdm

from models import ReviewerModel, load_model, get_model_size


def set_seed(seed=2233):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_codet5_model(model_path, num_cls):
    config_class, model_class, tokenizer_class = T5Config, ReviewerModel, RobertaTokenizer

    config = config_class.from_pretrained(model_path)
    config.num_cls = num_cls

    model = model_class.from_pretrained(model_path, num_cls)
    config, model, tokenizer = load_model(
        config,
        model,
        tokenizer_class,
        add_lang_ids=True
    )
    print(f"Finish loading model (size: {str(get_model_size(model))}) from {model_path}")

    model_name = os.path.join(model_path, "pytorch_model.bin")
    try:
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
    except:
        saved = model.cls_head
        model.cls_head = None
        model.load_state_dict(torch.load(model_name, map_location="cpu"))
        model.cls_head = saved
    model.to("cuda")

    return config, model, tokenizer

def encode_diff(diff ,tokenizer, max_source_length=512):
    difflines = diff.split("\n")
    # search for @@..@@ from first five line
    st = 0
    for i, fline in enumerate(difflines[:5]):
        if fline.strip().startswith("@@") and fline.strip().endswith("@@"):
            st = i+1
            break
    difflines = [line for line in difflines[st:] if len(line.strip()) > 0]
    map_dic = {"-": 0, "+": 1, " ": 2}
    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2
    labels = [f(line[0]) for line in difflines]
    difflines = [line[1:].strip() for line in difflines]
    inputstr = ""
    for label, line in zip(labels, difflines):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<del>" + line
        else:
            inputstr += "<keep>" + line

    source_ids = tokenizer.encode(inputstr, max_length=max_source_length, truncation=True)
    source_ids = source_ids[1:-1]
    source_ids = source_ids[:max_source_length - 2]
    source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
    pad_len = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_id] * pad_len

    return source_ids

def infer(code_diff, model, inverse_intent_dict):
    source_ids = torch.tensor([encode_diff(code_diff, tokenizer)], dtype=torch.long).to("cuda")
    source_mask = source_ids.ne(tokenizer.pad_id)
    logits = model(
        cls=True,
        input_ids=source_ids,
        labels=None,
        attention_mask=source_mask
    )
    prediction = torch.argmax(logits, dim=-1).cpu().numpy()
    return inverse_intent_dict[str(prediction[0])]


if __name__ == "__main__":
    num_cls = 14   # number of intent classes including OTHERS class
    model_path = "/datadisk/shuo/CodeReview/CodeReviewer/finetuned_model/intent_cls_14_classes/checkpoints-7000-0.470"

    set_seed()
    config, model, tokenizer = load_codet5_model(model_path, num_cls)
    model.to("cuda")
    model.eval()

    intent_dict = json.load(open("intent_dict.json"))
    inverse_intent_dict = {str(index): intent for intent, index in intent_dict.items()}

    code_diff = "@@ -1,8 +1,6 @@\n /*exported DqElement */\n \n function truncate(str, maxLength) {\n-\t'use strict';\n-\n \tmaxLength = maxLength || 300;\n \n \tif (str.length > maxLength) {"
    predicted_intent_class = infer(code_diff, model, inverse_intent_dict)
    print(f"Predicted intent class: {predicted_intent_class}")
    