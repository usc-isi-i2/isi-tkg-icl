import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging

from model_utils import predict
from utils import (
    HitsMetric,
    adjust_top_k,
    get_args,
    get_filename,
    load_data,
    prepare_input,
    update_history,
    update_metric,
    write_results,
)

tf_logging.set_verbosity_error()


if __name__ == "__main__":
    args = get_args()

    test_data, head_search_space, tail_search_space = load_data(args)

    adjust_top_k(test_data, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )
    model.eval()
    print(f"model is loaded on device {model.device.type}")

    metric = HitsMetric()
    filename = get_filename(args)
    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_data) as pbar:
        for i, (x, direction) in enumerate(pbar):
            if i % args.world_size != args.rank:
                continue

            if direction == "tail":
                search_space = head_search_space
            elif direction == "head":
                search_space = tail_search_space
            else:
                raise ValueError

            model_input, candidates = prepare_input(x, search_space, args, return_prompt=True)

            predictions = predict(tokenizer, model, model_input, args)

            update_history(x, search_space, predictions, candidates, args)

            example = write_results(x, predictions, candidates, direction, writer, args)

            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())
