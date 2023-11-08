import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional

MAX_HITS = 10


@dataclass
class HitsMetric:
    total: int = 0
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0

    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1

    def dump(self):
        return {
            "total": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument(
        "--dataset",
        choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO"],
        default="ICEWS18",
        type=str,
    )
    parser.add_argument(
        "--multi_step", default=False, action="store_true"
    )  # inference in multi_step
    # History Modeling
    parser.add_argument(
        "--history_type", choices=["entity", "pair"], default="entity", type=str
    )  # history type
    parser.add_argument(
        "--history_direction", choices=["uni", "bi"], default="uni", type=str
    )  # history type
    parser.add_argument("--history_len", default=0, type=int)  # length of history
    parser.add_argument("--history_top_k", default=1, type=int)  # length of targets from history
    # Prompt Construction
    parser.add_argument("--label", default=False, action="store_true")  # express prompt with label
    parser.add_argument(
        "--text_style", default=False, action="store_true"
    )  # express prompt in text
    parser.add_argument(
        "--no_entity", default=False, action="store_true"
    )  # express prompt without entity
    parser.add_argument("--sys_instruction", default="", type=str)  # system instcution for ChatGPT
    parser.add_argument(
        "--no_time", default=False, action="store_true"
    )  # express prompt without time
    parser.add_argument("--shuffle_history", default=False, action="store_true")  # shuffle history
    # Hyperparameter
    parser.add_argument("--top_k", default=100, type=int)  # number of predictions to store
    parser.add_argument(
        "--dec_cand", default=5, type=int
    )  # number of candidates to decode at each step
    parser.add_argument("--max_length", default=1, type=int)  # max decoding length
    parser.add_argument("--world_size", default=1, type=int)  # number of chunks
    parser.add_argument("--rank", default=0, type=int)  # rankd of the executor
    parser.add_argument(
        "--tokenizer_revision", default="main", type=str
    )  # change tokenizer revision (for llama)
    parser.add_argument(
        "--fp16", default=False, action="store_true"
    )  # use float16 instead of float32
    parser.add_argument("--verbose", default=False, action="store_true")  # print extra information
    # Evaluation
    parser.add_argument(
        "--eval_filter",
        choices=["none", "static", "time-aware"],
        type=str,
        default="none",
    )

    args = parser.parse_args()
    assert args.label or not args.no_entity

    return args


# Read entity2id, relation2id
def load_dictionary(in_path: str, file_name: str) -> Dict[int, str]:
    _dict = {}
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split("\t")
            node = line_split[0]
            index = int(line_split[1])

            _dict[index] = node
    return _dict


# Read train, valid data to construct search space
def load_quadruples(
    search_dictionary: Dict[Any, Dict[Any, Dict[Any, List[Any]]]],
    in_path: str,
    file_name: str,
    entity_dictionary: Optional[Dict[int, str]] = None,
    relation_dictionary: Optional[Dict[int, str]] = None,
    query: str = "head",
):
    discard_line, total_line = 0, 0
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            total_line += 1
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                    int(line_split[0]) not in entity_dictionary
                    or int(line_split[2]) not in entity_dictionary
                    or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    discard_line += 1
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])

            time = int(line_split[3])

            if query == "head":
                if head not in search_dictionary:
                    search_dictionary[head] = {}
                if time not in search_dictionary[head]:
                    search_dictionary[head][time] = {}
                if rel not in search_dictionary[head][time]:
                    search_dictionary[head][time][rel] = []
                search_dictionary[head][time][rel].append(tail)
            elif query == "tail":
                if tail not in search_dictionary:
                    search_dictionary[tail] = {}
                if time not in search_dictionary[tail]:
                    search_dictionary[tail][time] = {}
                if rel not in search_dictionary[tail][time]:
                    search_dictionary[tail][time][rel] = []
                search_dictionary[tail][time][rel].append(head)

    print(f"# line discarded due to index issue: {discard_line} / {total_line}")


# Read test data to inferencee
def load_quadruples_for_test(
    in_path: str,
    file_name: str,
    entity_dictionary: Optional[Dict[int, str]] = None,
    relation_dictionary: Optional[Dict[int, str]] = None,
) -> List[List[Any]]:
    test_instances = []
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                    int(line_split[0]) not in entity_dictionary
                    or int(line_split[2]) not in entity_dictionary
                    or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
            time = int(line_split[3])
            test_instances.append((head, rel, tail, time))
    return test_instances


def format_data(data):
    tail_prediction, head_prediction = {}, {}
    for head, rel, tail, time in data:
        tail_key = (head, rel, time)
        if tail_key not in tail_prediction:
            tail_prediction[tail_key] = []
        tail_prediction[tail_key].append(tail)

        head_key = (tail, rel, time)
        if head_key not in head_prediction:
            head_prediction[head_key] = []
        head_prediction[head_key].append(head)

    formatted_data = list(
        sorted(
            [([k[0], k[1], list(set(v)), k[2]], "tail") for k, v in tail_prediction.items()]
            + [([k[0], k[1], list(set(v)), k[2]], "head") for k, v in head_prediction.items()],
            key=lambda x: x[0][3],
        )
    )
    return formatted_data


def load_data(args: argparse.Namespace):
    entity_dictionary, relation_dictionary = None, None
    if args.text_style:
        entity_dictionary = load_dictionary("data", os.path.join(args.dataset, "entity2id.txt"))
        relation_dictionary = load_dictionary("data", os.path.join(args.dataset, "relation2id.txt"))

    head_search_space = {}
    load_quadruples(
        head_search_space,
        "data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )
    load_quadruples(
        head_search_space,
        "data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )

    tail_search_space = {}
    load_quadruples(
        tail_search_space,
        "data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )
    load_quadruples(
        tail_search_space,
        "data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )

    if args.history_direction == "bi":
        head_search_space.update(tail_search_space)
        tail_search_space = head_search_space

    test_data = load_quadruples_for_test(
        "data",
        os.path.join(args.dataset, "test.txt"),
        entity_dictionary,
        relation_dictionary,
    )

    formatted_test_data = format_data(test_data)

    return formatted_test_data, head_search_space, tail_search_space


def adjust_top_k(test_data, args):
    max_targets_len = max([len(x[0][2]) for x in test_data])
    args.top_k = max(args.top_k, MAX_HITS, max_targets_len + MAX_HITS)
    if args.verbose:
        print(f"max targets len: {max_targets_len}")
        print(f"adjusted top k: {args.top_k}")


def get_filename(args: argparse.Namespace, is_eval: bool = False):
    model_name = args.model.split("/")[-1]
    filename_args = "_".join(
        [
            model_name,
            args.dataset,
            f"multi_step_{args.multi_step}",
            f"history_len_{args.history_len}",
            f"history_type_{args.history_type}",
            f"history_direction_{args.history_direction}",
            f"no_time_{args.no_time}",
            f"shuffle_history_{args.shuffle_history}",
            f"label_{args.label}",
            f"text_style_{args.text_style}",
            f"no_entity_{args.no_entity}",
            f'world_size_{"*" if is_eval else args.world_size}',
            f'rank_{"*" if is_eval else args.rank}',
        ]
    )
    filename = f"outputs/{filename_args}.jsonl"
    print(f"output file: {filename}")
    return filename


def construct_history_by_search(
    search_space: Dict[str, Any], entity: str, relation: str, history_type: str
):
    if entity not in search_space:
        return {}

    search_graph = {entity: {}}

    if history_type == "entity":
        search_graph[entity] = search_space[entity]
    elif history_type == "pair":
        search_graph[entity] = {
            k: {relation: v[relation]} for k, v in search_space[entity].items() if relation in v
        }

    return search_graph


def format_history(
    history_graph: Dict[str, Any],
    history_len: int,
    question: List[str],
    args: argparse.Namespace,
    return_prompt: bool = True,
):
    quadruples = []
    for entity in history_graph:
        for time in history_graph[entity]:
            if time >= question[0]:
                continue
            for relation in history_graph[entity][time]:
                for target in history_graph[entity][time][relation]:
                    quadruples.append([entity, relation, target, time])

    candidates_stats = {}
    if args.model == "recency":
        for x in quadruples[-history_len:]:
            if x[2] not in candidates_stats:
                candidates_stats[x[2]] = -1
            candidates_stats[x[2]] = max(candidates_stats[x[2]], x[3])
    else:
        for x in quadruples[-history_len:]:
            if x[2] not in candidates_stats:
                candidates_stats[x[2]] = 0
            candidates_stats[x[2]] += 1

    candidates_stats_sorted = list(
        sorted(candidates_stats.items(), key=lambda item: item[1], reverse=True)
    )

    candidates_mapping = {}
    for i, (entity, _) in enumerate(candidates_stats_sorted):
        candidates_mapping[entity] = i

    if (args.label or args.no_entity) and args.model not in ["recency", "frequency"]:
        candidates = {v: k for k, v in candidates_mapping.items()}  # label --> entity
    else:
        candidates = {k: k for k, _ in candidates_mapping.items()}  # entity --> entity

    if return_prompt:
        prompt = ""
        history = quadruples[-history_len:]
        if args.shuffle_history:
            random.shuffle(history)
        for x in history:
            entity, relation, target, time = x[0], x[1], x[2], x[3]
            if not args.no_time:
                prompt += f"{time}:"
            if args.no_entity:
                prompt += f"[{entity},{relation},{candidates_mapping[target]}]\n"
            elif args.label:
                prompt += f"[{entity},{relation},{candidates_mapping[target]}.{target}]\n"
            else:
                prompt += f"[{entity},{relation},{target}]\n"
        if not args.no_time:
            prompt += f"{question[0]}:"
        prompt += f"[{question[1]},{question[2]},"

        return prompt, candidates
    else:
        return candidates_stats_sorted, candidates


def prepare_input(x, entity_search_space, args, return_prompt: bool = True):
    entity, relation, time = x[0], x[1], x[3]
    entity_history = construct_history_by_search(
        entity_search_space,
        entity=entity,
        relation=relation,
        history_type=args.history_type,
    )
    history_input, candidates = format_history(
        entity_history,
        args.history_len,
        [time, entity, relation],
        args=args,
        return_prompt=return_prompt,
    )
    if args.verbose:
        print(f"input:\n{history_input}\ncandidates:\n{candidates}")

    if entity not in entity_search_space:
        entity_search_space[entity] = {}
    if time not in entity_search_space[entity]:
        entity_search_space[entity][time] = {}
    if relation not in entity_search_space[entity][time]:
        entity_search_space[entity][time][relation] = []

    return history_input, candidates


def update_history(x, entity_search_space, predictions, candidates, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    if args.verbose:
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )
    if args.multi_step:
        filtered_predictions = [candidates[x[0]] for x in predictions if x[0] in candidates]
        targets = filtered_predictions[: args.history_top_k]
    entity_search_space[entity][time][relation] += targets
    if args.verbose:
        print(f"history:\n{entity},{relation},{time} --> {targets}")
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )


def write_results(x, predictions, candidates, direction, writer, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation": relation,
        "targets": targets,
        "direction": direction,
        "predictions": [candidates[x[0]] for x in predictions if x[0] in candidates],
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args):
    if args.verbose:
        print(f'predictions: {example["predictions"]}')
    for target in example["targets"]:
        metric.total += 1
        index = example["predictions"].index(target) if target in example["predictions"] else -1
        if index >= 0:
            _predictions = [
                x for x in example["predictions"][:index] if x not in example["targets"]
            ]
            rank = len(_predictions) + 1
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            metric.update(rank)
