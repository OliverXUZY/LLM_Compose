import numpy as np
import torch
import random
import os
from tqdm import tqdm

import sys
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
from src.utils import load_json, Timer, time_str
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="a_level,a_level_symbol")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_fewshot", type=int, default=10)
    parser.add_argument("--description_dict_path", default="./data/description.json")
    parser.add_argument(
        "--limit",
        type=float,
        default=100,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--output_base_path", type=str, default=None)
    return parser.parse_args()


def inference(
        task_name,
        evaluator,
        n_shot,
        description,
        limit,
        output_base_path,
    ):

    print(f"inference for {task_name}...")
    timer = Timer()
    
    task = tasks_mapping[task_name]()

    # print(task.dataset)
    dataset = task.dataset
    train_examples = task.training_docs()
    test_examples = task.validation_docs()


    ### sample ids
    rnd = random.Random()
    rnd.seed(3407)

    test_examples = list(test_examples)
    rnd_test_shuffle = random.Random()
    rnd_test_shuffle.seed(3407)
    rnd_test_shuffle.shuffle(test_examples) # TODO: tem zhuoyan

    kwangs_fewshot_context = {}

    if limit is not None:
        limit = int(len(test_examples) * limit) if limit < 1.0 else int(limit)

    accs = 0
    results = []

    
    
    for doc_id, doc in enumerate(itertools.islice(test_examples, 0, limit)):
        if task_name == "ab_level":
            kwangs_fewshot_context.update({"doc_id": doc_id})
        
        if (doc_id+1) % 10 == 0:
            print(f"===== {task_name} =========== {doc_id+1}")

        qn = task.fewshot_context(doc, n_shot , rnd, description=description, **kwangs_fewshot_context)

        prompt = qn
        answer = task.doc_to_target(doc)

        # print(qn)

        # print(answer)

        result = evaluator.eval(prompt, answer, seq_len = 20)
        accs += result.accuracy
        results.append(asdict(result))


    results = [{"acc": accs / min(len(test_examples),limit)}] + results
    if output_base_path:
        file_path = output_base_path
    else:
        file_path = f"./output/debug"
    

    file_name = f"{task_name}"
        

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print(f"save to {file_path}/{file_name}")


    with open(
        f"{file_path}/{file_name}.json",
        "w",
    ) as f:
        json.dump(results, f, indent = 4)

    print(f"{task_name} | {evaluator.get_model_id()} | use time {time_str(timer.end())}")


def main():
    args = parse_args()
    task_names = args.tasks.split(",")
    print("this run comtain tasks: ", task_names)
    description_dict = load_json(args.description_dict_path)
    timer = Timer()


    evaluator = Evaluator(args.model_id, device)
    
    for task_name in task_names:
        description = description_dict.get(task_name) or ""
        inference(
            task_name = task_name,
            evaluator = evaluator,
            n_shot = args.num_fewshot,
            description = description,
            limit = args.limit,
            output_base_path = args.output_base_path
        )
    
    print(f"{args.model_id} all done | use time {time_str(timer.end())}")

if __name__ == '__main__':
    main()
