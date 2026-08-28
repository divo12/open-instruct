"""
Update hamishivi/tmax-sft-full-20260317 with a revised system prompt.

The original system prompt embeds tool function signatures in a <tools> XML block.
Since tools are already tracked in a separate top-level `tools` column, the new
system prompt removes that block and just keeps the high-level instruction.

Extra message fields (reasoning_content, tool_call_ids) are also dropped so the
output matches the unified {role, content, tool_calls?, tool_call_id?} schema.

Usage:
    python scripts/data/convert_terminal_sft_to_unified_format.py \\
        --input_dataset hamishivi/tmax-sft-full-20260317 \\
        --output_dataset <your-org/your-dataset-name>
"""

import argparse
import json

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

NEW_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You have access to a persistent bash terminal.\n"
    "Use it to explore the codebase, understand the problem, implement a solution, and verify it works.\n"
    "\n"
    "IMPORTANT RULES:\n"
    "- Every response must include a THOUGHT section explaining your reasoning, followed by exactly one bash command.\n"
    "- Your working directory and environment variables persist between commands."
    " You can `cd` into a directory and subsequent commands will run there."
    " You can `export` variables and they will be available in later commands.\n"
    "- Edit files using bash commands like `sed`, `cat > file << 'EOF'`, etc.\n"
    "- Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.\n"
    "- Interactive commands are not possible. Use `yes`/`no`, etc. as appropriate.\n"
    "- Output may be truncated. Use `head`/`tail`/`grep` to filter large outputs.\n"
    "- When you are confident your solution is correct, submit by running:"
    " `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    "- After submitting you cannot continue working on the task.\n"
    "\n"
    "# Tools\n"
    "You may call one or more functions to assist with the user query.\n"
)

KEEP_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}


def convert_sample(sample: dict) -> dict:
    msgs = sample.get("messages", [])
    clean_messages = []
    for m in msgs:
        clean = {k: v for k, v in m.items() if k in KEEP_FIELDS and v is not None}
        if m.get("role") == "system":
            clean["content"] = NEW_SYSTEM_PROMPT
        if "tool_calls" in clean and not clean["tool_calls"]:
            del clean["tool_calls"]
        if clean:
            clean_messages.append(clean)
    result = {"messages": clean_messages}
    if sample.get("tools") is not None:
        result["tools"] = json.dumps(sample["tools"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", default="hamishivi/tmax-sft-full-20260317")
    parser.add_argument("--output_dataset", required=True, help="HuggingFace dataset repo to push to")
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Dataset split to process. hamishivi/tmax-sft-full-20260317 uses non-standard names:"
            " nvidia__Nemotron_Terminal_Corpus__dataset_adapters,"
            " nvidia__Nemotron_Terminal_Corpus__skill_based_easy,"
            " nvidia__Nemotron_Terminal_Corpus__skill_based_medium,"
            " nvidia__Nemotron_Terminal_Corpus__skill_based_mixed,"
            " open_thoughts__OpenThoughts_Agent_v1_SFT"
        ),
    )
    args = parser.parse_args()

    logger.info(f"Loading {args.input_dataset} ...")
    raw = load_dataset(args.input_dataset, split=args.split)
    logger.info(f"Loaded {len(raw)} samples")

    # remove_columns drops the old structured-list "tools" column so map() can
    # add the new Value('string') version without a schema conflict
    out_ds = raw.map(convert_sample, remove_columns=["tools"])
    logger.info(f"Pushing to {args.output_dataset} ...")
    out_ds.push_to_hub(args.output_dataset, split=args.split)
    logger.info("Done")


if __name__ == "__main__":
    main()
