import json
from time import sleep

import numpy as np
import openai

with open("openai_config.json", encoding="utf-8") as f:
    config = json.load(f)

openai.api_key = config["openai_api_key"]


def parse_results(result):
    raw_logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
    logprobs = [(int(x.strip()), raw_logprobs[x]) for x in raw_logprobs if x.strip().isdecimal()]
    sorted_logprobs = sorted(logprobs, key=lambda tup: tup[1], reverse=True)

    probs = [x[1] for x in sorted_logprobs]
    softmax_probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)

    to_return = [(x[0], p) for x, p in zip(sorted_logprobs, softmax_probs)]
    return to_return


def parse_results_chatgpt(result):
    return_text = result["choices"][0]["message"]["content"]
    to_return = [(int(return_text), 1)] if return_text.isdecimal() else []
    return to_return


def predict(prompt, args):
    got_result = False
    while not got_result:
        try:
            results = openai.Completion.create(
                engine=args.model,
                prompt=prompt,
                max_tokens=64,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]", "."],
                logprobs=10,
            )
            got_result = True
        except Exception:  # pylint: disable=broad-exception-caught
            sleep(3)

    parsed_results = parse_results(results)  # type: ignore
    return parsed_results


def predict_chatgpt(prompt, args):
    if args.sys_instruction == "":
        prompt = [{"role": "user", "content": prompt}]
    else:
        prompt = [
            {"role": "system", "content": args.sys_instruction},
            {"role": "user", "content": prompt},
        ]

    got_result = False
    while not got_result:
        try:
            results = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=prompt,
                max_tokens=64,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]", "."],
            )
            got_result = True
        except Exception:  # pylint: disable=broad-exception-caught
            sleep(3)

    parsed_results = parse_results_chatgpt(results)  # type: ignore
    return parsed_results
