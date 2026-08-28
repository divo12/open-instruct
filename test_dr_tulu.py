import sys, json
sys.path.insert(0, '.')
from datasets import load_dataset
from scripts.data.convert_dr_tulu_sft_to_tulu_format import convert_sample as drtulu_convert
from scripts.data.convert_swe_sft_to_unified_format import convert_sample as swe_convert

for name, ds_id, convert_fn in [
    ("Dr Tulu", "allenai/dr-tulu-sft-qwen35", drtulu_convert),
    ("SWE",     "allenai/sera-4.6-lite-t2-qwen35", swe_convert),
]:
    raw = load_dataset(ds_id, split="train[:1]")
    out = raw.map(convert_fn, remove_columns=["messages"], load_from_cache_file=False)
    print(f"=== {name} ===")
    print(f"  tools column type: {out.features['tools']}")
    tools = json.loads(out[0]["tools"])
    for t in tools:
        fn = t["function"]
        params = list(fn["parameters"]["properties"].keys())
        assert all(p is not None for p in params), f"null param in {fn['name']}: {params}"
        print(f"  {fn['name']}: params={params}")
    print()
    print(json.dumps(out[0], indent=2))
    input("Press Enter to continue...")