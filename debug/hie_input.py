import numpy as np
import torch
import random
import os
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/user/Documents/projects/nlp/lmeval')
sys.path.insert(0, '/srv/home/zxu444/nlp/lmeval')

import argparse
from datasets import load_dataset
from dataclasses import dataclass, asdict
import json

from pprint import pprint
import itertools
import src
from src import tasks_repo as tasks_mapping
from src import Evaluator
from src.fuzzycopy import generator
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--n_shot", type=int, default=8)
    parser.add_argument("--masked", action="store_true")
    return parser.parse_args()

# tasks_mapping = {
#     "a_level": a_level,
#     "b_level": b_level,
#     "ab_level": ab_level,
#     "ab_level_compose_incontext": ab_level_compose_incontext,
#     "a_level_symbol": a_level_symbol,
#     "ab_level_symbol": ab_level_symbol,
#     "ab_level_compose_incontext_symbol": ab_level_compose_incontext_symbol

# }

def t4(item, label=None):
    if label is None:
        return f"{item} is"
    return f"{item} is {label}, "

def main():

    pprint(tasks_mapping)


    args = parse_args()
    task_name = "a_level"
    task_name = "b_level"
    task_name = "ab_level"
    #task_name = "ab_level_compose_incontext"
    #task_name = "a_level_symbol"
    #task_name = "ab_level_symbol"
    #task_name = "ab_level_compose_incontext_symbol"
    task_name = "proofwriter"
    
    task = tasks_mapping[task_name]()
    # task = b_level()
    # task = ab_level()
    # task = ab_level_compose_incontext()

    print(task.dataset)
    dataset = task.dataset
    train_examples = task.training_docs()
    test_examples = task.validation_docs()
    # print(len(train_examples))
    # print(test_examples)

    

    doc = test_examples[0]
    # print(task.doc_to_text(doc))
    # print(task.doc_to_target(doc))

    # print(type(task.doc_to_target(doc)))


    # assert False

    print("============random============")

    ### sample ids
    rnd = random.Random()
    rnd.seed(3407)

    description = "Below you are given examples showing how items fit into categories and larger groups. If an item is in a category that's part of a larger group, remember the connection. Use what you've learned to determine the larger group for any new item."
    
    #description = "Below you are given examples showing how items fit into categories. Use what you've learned to determine the category for any new item."
    #description = ""

    random.seed(42)  # Set the seed for reproducibility
    # Sample 100 random indices from 1 to 200
    sample_indices = random.sample(range(1, 5), 1)

    test_examples = list(test_examples)
    rnd_test_shuffle = random.Random()
    rnd_test_shuffle.seed(3407)
    rnd_test_shuffle.shuffle(test_examples) # TODO: tem zhuoyan
    print("bbb",len(test_examples))

    kwangs_fewshot_context = {}

    limit = 20
    accs = 0
    results = []

    accs_ds = 0
    results_ds = []


    evaluator = Evaluator(args.model_id, device)
    
    
    n_shot = 0
    for doc_id, doc in enumerate(itertools.islice(test_examples, 0, limit)):
        # Randomly sample an example
        # doc = test_examples[idx]
        if task_name == "ab_level":
            kwangs_fewshot_context.update({"doc_id": doc_id})

        
        qn = task.fewshot_context(doc, n_shot ,  rnd, description=description, **kwangs_fewshot_context)
        # qn = doc

        # print(task.doc_to_text(doc))
        # print('===========================')
        print("prompt: ", type(qn), qn)
        print("ground truth: ", task.doc_to_target(doc))    

        # assert False

        prompt = qn
        answer = task.doc_to_target(doc)

        print("PROMPT: ", type(prompt), prompt)
        print("answer: ", answer)
        result = evaluator.eval(prompt, answer)
        accs += result.accuracy
        results.append(asdict(result))


        print("ds==")
        ds = generator(n_examples = n_shot, template = t4, symbol = True)
        prompt, answer = next(ds)

        print("PROMPT: ", type(prompt), prompt)
        print("answer: ", answer)
    
    
        result_ds = evaluator.eval(prompt, answer)
        accs_ds += result_ds.accuracy
        results_ds.append(asdict(result_ds))


        print("============= id =", doc_id)


    results = [{"acc": accs / limit}] + results
    results_ds = [{"acc": accs_ds / limit}] + results_ds

    with open(
        f"debug/newline_{task_name}.json",
        "w",
    ) as f:
        json.dump(results, f, indent = 4)
    with open(
        f"test_ds.json",
        "w",
    ) as f:
        json.dump(results_ds, f, indent = 4)







if __name__ == '__main__':
    main()
