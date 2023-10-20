import argparse
from copy import deepcopy
from dataclasses import dataclass
import glob
import json
import os

from utils import get_filename, load_dictionary, load_quadruples_for_test


@dataclass
class EmptinessMetric:
    total: int = 0
    empty: int = 0

    def dump(self):
        return self.empty / self.total


def read_jsonl(filenames: str):
    json_lines = []
    for fn in glob.glob(filenames):
        with open(fn, encoding="utf-8") as f:
            file_lines = f.read().strip().rsplit("\n")
            for line in file_lines:
                json_lines.append(json.loads(line))
    return json_lines


def load_data(args: argparse.Namespace):
    filenames = get_filename(args, is_eval=True)

    if len(glob.glob(filenames)) == 0:
        print("no input files found.")
        print()
        exit()

    entity_dictionary, relation_dictionary = None, None
    if args.text_style:
        entity_dictionary = load_dictionary("data", os.path.join(args.dataset, "entity2id.txt"))
        relation_dictionary = load_dictionary("data", os.path.join(args.dataset, "relation2id.txt"))

    output_data = read_jsonl(filenames)
    train_data = load_quadruples_for_test(
        "data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
    )
    valid_data = load_quadruples_for_test(
        "data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
    )
    test_data = load_quadruples_for_test(
        "data",
        os.path.join(args.dataset, "test.txt"),
        entity_dictionary,
        relation_dictionary,
    )

    return output_data, train_data, valid_data, test_data


def get_index(train_data, valid_data, test_data, args):
    index = {}
    if args.eval_filter == "static":
        index = {
            y: True
            for y in (
                [x[:3] for x in train_data]
                + [x[:3] for x in valid_data]
                + [x[:3] for x in test_data]
            )
        }
    elif args.eval_filter == "time-aware":
        index = {y: True for y in (train_data + valid_data + test_data)}
    elif args.eval_filter != "none":
        raise ValueError(f"unknown eval_filter: {args.eval_filter}")

    return index


def filter_predictions(x, index, args):
    filtered_predictions = []
    if args.eval_filter == "none":
        filtered_predictions = deepcopy(x["predictions"])
    elif args.eval_filter == "static":
        for y in x["predictions"]:
            if x["direction"] == "head" and (
                y in x["targets"] or (y, x["relation"], x["entity"]) not in index
            ):
                filtered_predictions.append(y)
            elif x["direction"] == "tail" and (
                y in x["targets"] or (x["entity"], x["relation"], y) not in index
            ):
                filtered_predictions.append(y)
            else:
                raise ValueError
    if args.eval_filter == "time-aware":
        for y in x["predictions"]:
            if x["direction"] == "head" and (
                y in x["targets"] or (y, x["relation"], x["entity"], x["timestamp"]) not in index
            ):
                filtered_predictions.append(y)
            elif x["direction"] == "tail" and (
                y in x["targets"] or (x["entity"], x["relation"], y, x["timestamp"]) not in index
            ):
                filtered_predictions.append(y)
            else:
                raise ValueError
    return filtered_predictions


def update_metric(x, filtered_predictions, hits_metric, empty_metric, args):
    if args.verbose:
        print(f"filtered predictions: {filtered_predictions}")
    for target in x["targets"]:
        hits_metric.total += 1
        empty_metric.total += 1
        if len(filtered_predictions) == 0:
            empty_metric.empty += 1
        index = filtered_predictions.index(target) if target in filtered_predictions else -1
        if index >= 0:
            _predictions = [y for y in filtered_predictions[:index] if y not in x["targets"]]
            rank = len(_predictions) + 1
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            hits_metric.update(rank)


def print_metrics(rank_metric, empty_metric):
    for k, v in rank_metric.dump().items():
        if k == "total":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    print(f"empty: {empty_metric.dump():.2%}")
    print()
