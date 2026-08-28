"""Load a HF tokenizer and dataset, apply the chat template to the messages column.

Example:
    uv run python scripts/apply_chat_template_demo.py \\
        --model_id Qwen/Qwen3.5-4B \\
        --dataset_id hamishivi/tmax-sft-full-20260317 \\
        --split nvidia__Nemotron_Terminal_Corpus__dataset_adapters \\
        --num_examples 2
"""

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="HF model id to load the tokenizer from.")
    parser.add_argument("--dataset_id", required=True, help="HF dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split or config to load.")
    parser.add_argument("--messages_column", default="messages", help="Column containing the chat messages.")
    parser.add_argument("--tools_column", default="tools", help="Optional column containing tool schemas.")
    parser.add_argument("--num_examples", type=int, default=1, help="How many examples to render.")
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="If set, append the assistant generation prompt at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ds = load_dataset(args.dataset_id, split=args.split)

    n = min(args.num_examples, len(ds))
    for i in range(n):
        example = ds[i]
        messages = example[args.messages_column]
        if isinstance(messages, str):
            messages = json.loads(messages)

        tools = example.get(args.tools_column) if args.tools_column in ds.column_names else None
        # Some datasets nest the tool schemas inside the system message instead of
        # a top-level column. Lift them out so the template gets them via `tools=`.
        if tools is None and messages and messages[0].get("role") == "system" and "tools" in messages[0]:
            tools = messages[0].pop("tools")

        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=args.add_generation_prompt,
        )
        print(f"===== example {i} =====")
        print(rendered)


if __name__ == "__main__":
    main()
