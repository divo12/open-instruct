"""
Convert allenai/dr-tulu-sft-qwen35 to the standard tulu tool-calling SFT format
(compatible with hamishivi/tmax-sft-full-* and rl-rag/browsecomp-gptoss-clean-qwen35-sft).

Source format quirks:
  - `messages` is stored as a JSON string, not a structured list
  - Tool definitions live in the system message's `tools` key, not a top-level `tools` column
  - Each message has extra `function_calls` and `functions` keys that should be dropped

Output format:
  - `messages`: structured list of {role, content, tool_calls?, tool_call_id?}
  - `tools`: top-level list of tool definitions (extracted from the system message)
  - System prompt updated to remove <think> tag instructions and the workflow example

Usage:
    python scripts/data/convert_dr_tulu_sft_to_tulu_format.py \\
        --input_dataset allenai/dr-tulu-sft-qwen35 \\
        --output_dataset <your-org/your-dataset-name>
"""

import argparse
import json

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

KEEP_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}

NEW_SYSTEM_PROMPT = (
    "You are a research assistant who answers questions through iterative reasoning and research.\n"
    "\n"
    "## Process\n"
    "- Use the provided tools when you need information.\n"
    "- You can alternate between thinking and searching multiple times.\n"
    "- Only provide <answer></answer> tags when you have enough information for a complete response."
    " If the problem asks for a specific, short-form answer, you can also put the answer string in"
    " the \\boxed{} format. \n"
    "- Support every non-trivial claim with retrieved evidence. Wrap the exact claim span in"
    ' <cite id="ID1,ID2">...</cite>, where id are snippet IDs from searched results'
    " (comma-separated if multiple). Use only returned snippets; never invent IDs."
    " Avoid citing filler text - cite just the factual claim.\n"
    "\n"
    "## Answer and Citation Format\n"
    "\n"
    "- Once you collect all of the necessary information, generate the final answer, and mark your"
    " answer with answer tags: <answer></answer>. \n"
    "- If your answer is short (e.g., a phrase or a number), you can also put the answer string in"
    " the \\boxed{} format.\n"
    '- In your answer, wrap the supported text in <cite id="SNIPPET_ID"> ... </cite>. You have to'
    " use the exact ID from a returned <snippet id=...>...</snippet>.\n"
    "- If multiple sources support a passage, use multiple <cite> tags around the relevant clauses/sentences.\n"
    "- Examples \n"
    '<cite id="S17">LLMs often hallucinate on long-tail facts.</cite>\n'
    "<answer>Based on the search results,"
    ' <cite id="S23">the first Harry Potter movie was released on November 16, 2001.</cite>'
    "Therefore, the final answer is \\boxed{November 16, 2001}.</answer>\n"
    "\n"
    "## REQUIREMENTS\n"
    "- Think and search iteratively until you have sufficient information\n"
    "- Only provide the final answer when ready\n"
    "- Cite all claims from search results using exact snippet IDs"
)


def convert_sample(sample: dict) -> dict:
    msgs_raw = sample.get("messages", [])

    # messages is stored as a JSON string split into individual characters
    try:
        msgs = json.loads("".join(msgs_raw))
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse messages JSON: {e}") from e

    # Extract tools from the system message and replace its content
    tools = []
    for m in msgs:
        if m.get("role") == "system" and m.get("tools"):
            tools = m["tools"]
            break

    # Clean each message: keep only the fields the training pipeline uses
    clean_messages = []
    for m in msgs:
        clean = {k: v for k, v in m.items() if k in KEEP_FIELDS and v is not None}
        if m.get("role") == "system":
            clean["content"] = NEW_SYSTEM_PROMPT
        # tool_calls: drop if empty list
        if "tool_calls" in clean and not clean["tool_calls"]:
            del clean["tool_calls"]
        if clean:
            clean_messages.append(clean)

    return {"messages": clean_messages, "tools": json.dumps(tools)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="allenai/dr-tulu-sft-qwen35")
    parser.add_argument("--output_dataset", required=True, help="HuggingFace dataset repo to push to")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    logger.info(f"Loading {args.input_dataset} ...")
    raw = load_dataset(args.input_dataset, split=args.split)
    logger.info(f"Loaded {len(raw)} samples")

    # remove_columns drops the old character-array "messages" so map() can add
    # the new structured-list version without a schema conflict
    out_ds = raw.map(convert_sample, remove_columns=["messages"])
    logger.info(f"Pushing to {args.output_dataset} ...")
    out_ds.push_to_hub(args.output_dataset, split=args.split)
    logger.info("Done")


if __name__ == "__main__":
    main()
