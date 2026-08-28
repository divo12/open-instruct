"""
Extract all unique function names called across the HF dataset:
  allenai/sera-4.6-lite-t2-qwen35

The 'messages' column is a JSON-encoded string. Each decoded message dict
has a 'tool_calls' list where each entry looks like:
  {"id": "...", "type": "function", "function": {"name": "...", "arguments": {...}}}
"""

import json
from collections import Counter
from datasets import load_dataset


DATASET_NAME = "allenai/sera-4.6-lite-t2-qwen35"


def extract_names_from_message(msg) -> list[str]:
    names = []
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return names
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(name)
    return names


def main():
    print(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train")

    print(f"Total rows: {len(ds)}")
    print(f"Columns: {ds.column_names}\n")

    counter: Counter = Counter()

    for row in ds:
        raw = row.get("messages") or row.get("conversations") or ""
        if not raw:
            continue
        messages = json.loads(raw) if isinstance(raw, str) else raw
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for name in extract_names_from_message(msg):
                counter[name] += 1

    print(f"Found {len(counter)} unique function names:\n")
    print(f"{'Function name':<40} {'Call count':>10}")
    print("-" * 52)
    for name, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"{name:<40} {count:>10}")

    unique_names = sorted(counter.keys())
    print(f"\nUnique names (sorted): {unique_names}")


if __name__ == "__main__":
    main()
