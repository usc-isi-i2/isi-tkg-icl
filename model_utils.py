import json
import math

import numpy as np
import torch


def permute(tokenizer, scores, cur_step, max_step, cur_seq, seqs, dec_cand, end_char):
    if cur_step == max_step or (len(cur_seq) > 0 and end_char in cur_seq[-1]["token"]):
        _cur_seq = cur_seq[:-1].copy() if end_char in cur_seq[-1]["token"] else cur_seq.copy()
        normalized_logit = (
            sum([x["logit"] for x in _cur_seq]) / len(_cur_seq) if len(_cur_seq) > 0 else -math.inf
        )
        seqs.append(
            {
                "tokens": [x["token"] for x in _cur_seq],
                "text": "".join([x["token"] for x in _cur_seq]).strip(),
                "probability": normalized_logit,
            }
        )
        return
    logits = scores[cur_step]
    logits_indices = torch.argsort(logits, dim=-1, descending=True)
    for tok in logits_indices[0][:dec_cand]:
        cur_seq.append({"token": tokenizer.decode(tok), "logit": logits[0][tok].item()})
        permute(tokenizer, scores, cur_step + 1, max_step, cur_seq, seqs, dec_cand, end_char)
        cur_seq.pop()


def deduplicate(x):  # NOTE: assumes a sorted list based on probability
    f = {}
    z = []
    for y in x:
        if y[0] in f:
            continue
        f[y[0]] = True
        z.append(y)
    return z


def parse_results(results):
    logprobs = [(int(x["text"]), x["probability"]) for x in results if x["text"].isdecimal()]
    sorted_logprobs = sorted(logprobs, key=lambda tup: tup[1], reverse=True)
    dedup_sorted_logprobs = deduplicate(sorted_logprobs)

    probs = [x[1] for x in dedup_sorted_logprobs]
    softmax_probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)

    to_return = [(x[0], p) for x, p in zip(dedup_sorted_logprobs, softmax_probs)]
    return to_return


def predict(tokenizer, model, prompt, args):
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_length,
        return_dict_in_generate=True,
        output_scores=True,
        renormalize_logits=True,
    )

    if args.verbose:
        print("outputs:\n")
    if args.label and "llama" not in args.model:
        probs = outputs.scores[0]  # (1, num_vocab)
        probs_indices = torch.argsort(probs, dim=-1, descending=True)  # sort probs and get index

        results = []
        for tok in probs_indices[0][: args.top_k]:
            if args.verbose:
                print(
                    f"| {tok:5d} | {tokenizer.decode(tok):8s} | {probs[0][tok].item():.4f} | {np.exp(probs[0][tok].item()):.2%}"
                )
            results.append(
                {
                    "text": tokenizer.decode(tok).strip(),
                    "probability": probs[0][tok].item(),
                }
            )
    else:
        results = []
        permute(
            tokenizer,
            outputs.scores,
            0,
            args.max_length,
            [],
            results,
            args.dec_cand,
            "." if args.label and not args.no_entity else "]",
        )
        results = list(sorted(results, key=lambda x: x["probability"], reverse=True))[: args.top_k]
        if args.verbose:
            for x in results:
                print(
                    f'| {json.dumps(x["tokens"]):30s} | {x["text"]:10s} | {x["probability"]:.4f} | {np.exp(x["probability"]):.2%}'
                )

    parsed_results = parse_results(results)
    return parsed_results
