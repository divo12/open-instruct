"""
Convert allenai/sera-4.6-lite-t2-qwen35 to the unified SFT format.

Source format quirks:
  - Tool definitions live in the system message's `tools` key, not a top-level column
  - Each message has extra `function_calls` and `functions` keys that should be dropped
  - `messages` may be stored as a JSON character array (same artifact as dr-tulu-sft-qwen35)

Output format:
  - `messages`: list of {role, content, tool_calls?, tool_call_id?}
  - `tools`: top-level list of tool definitions (extracted from the system message)

Usage:
    python scripts/data/convert_swe_sft_to_unified_format.py \\
        --input_dataset allenai/sera-4.6-lite-t2-qwen35 \\
        --output_dataset <your-org/your-dataset-name>
"""

import argparse
import json

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

KEEP_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}


def parse_messages(msgs_raw: list) -> list:
    if msgs_raw and isinstance(msgs_raw[0], str) and len(msgs_raw[0]) == 1:
        try:
            return json.loads("".join(msgs_raw))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse messages as JSON character array, using as-is")
    return msgs_raw


def convert_sample(sample: dict) -> dict:
    msgs_raw = sample.get("messages", [])
    msgs = parse_messages(msgs_raw)

    tools = []
    for m in msgs:
        if m.get("role") == "system" and m.get("tools"):
            tools = m["tools"]
            break

    clean_messages = []
    for m in msgs:
        clean = {k: v for k, v in m.items() if k in KEEP_FIELDS and v is not None}
        if "tool_calls" in clean and not clean["tool_calls"]:
            del clean["tool_calls"]
        if clean:
            clean_messages.append(clean)

    return {"messages": clean_messages, "tools": json.dumps(tools)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="allenai/sera-4.6-lite-t2-qwen35")
    parser.add_argument("--output_dataset", required=True, help="HuggingFace dataset repo to push to")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logger.info(f"Loading {args.input_dataset} ...")
    raw = load_dataset(args.input_dataset, split=args.split)
    logger.info(f"Loaded {len(raw)} samples")

    out_ds = raw.map(convert_sample, remove_columns=["messages"])
    logger.info(f"Pushing to {args.output_dataset} ...")
    out_ds.push_to_hub(args.output_dataset, split=args.split)
    logger.info("Done")


if __name__ == "__main__":
    main()
