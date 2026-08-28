# Terminal RL on nvidia/Nemotron-Terminal-Synthetic-Tasks

Goal: train with our Terminal RL setup (`grpo_fast.py` + `--tools swerl_vanillux_sandbox`)
on tasks from [`nvidia/Nemotron-Terminal-Synthetic-Tasks`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Synthetic-Tasks).

Status (2026-06-17): **de-risking spike complete and passing.** Full conversion +
image-build pipeline not yet built.

## How our Terminal RL consumes tasks

`SWERLVanilluxSandboxEnv` ([open_instruct/environments/swerl_vanillux_sandbox.py](../../open_instruct/environments/swerl_vanillux_sandbox.py))
needs two artifacts, produced today for tmax by [convert_tmax_tasks.py](convert_tmax_tasks.py):

1. **Prompt HF dataset** — columns `messages`, `ground_truth`, `dataset`, `env_config`, `source`.
   `env_config.image` (or a per-task `image.txt` in the tarball) sets the Docker image.
2. **`task-data.tar.gz`** (passed via `--tool_configs '{"task_data_hf_repo": ...}'`) laid out as
   `{task_id}/instruction.md`, `{task_id}/tests/test.sh`, `{task_id}/environment/seeds/` (optional),
   `{task_id}/image.txt` (optional).

Runtime flow per rollout: pull image → seed files → show `instruction.md` → agent runs `bash`
→ on `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, upload `tests/` to `/tests` and run
`test.sh` → read reward from **`/logs/verifier/reward.txt`** (clamped to 0–1).

## Nemotron task format (terminal-bench / Harbor "t-bench")

Tarballs: `skill_based/{easy,medium_shard1,medium_shard2}.tar.gz` and
`skill_based/mixed/{data_processing,data_science,debugging,file_operations,scientific_computing,security}.tar.gz`.
Each task dir:

```
<task_id>/
  instruction.md            # task prompt (references absolute /app paths)
  task.toml                 # metadata: docker_image (PRIVATE nvidia registry), timeouts, cpus, mem
  environment/
    Dockerfile              # FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest (PUBLIC) + apt/pip
    files/                  # COPY files/ /app/  (WORKDIR /app)
  tests/
    test.sh                 # writes 0/1 to /logs/verifier/reward.txt  <-- matches our contract
    test_outputs.py         # pytest assertions
    test_requirements.txt   # pip-installed by test.sh at runtime (needs container network)
  solution/                 # EMPTY in all inspected tasks (no reference solutions)
```

`security` split alone has 987 tasks; total corpus is 100K–1M rows.

## Compatibility

| Piece | Status |
|---|---|
| Reward contract (`/logs/verifier/reward.txt`, binary) | ✅ identical — no change |
| `tests/` multi-file pytest upload | ✅ env uploads whole dir; `test.sh` unchanged |
| `instruction.md` → prompt | ✅ maps to vanillux instance template |
| Docker images | ⚠️ private NVIDIA registry refs are unpullable; **must rebuild from shipped Dockerfiles** |
| `/app` file paths | ✅ with per-task images (see decision below) |

## Image strategy decision: per-task images (validated)

`environment/Dockerfile` does `WORKDIR /app` + `COPY files/ /app/`, so task files are
**task-specific**. We build **one image per task** from the shipped Dockerfile (public GHCR base,
fully cacheable apt/pip layers), push to a registry the Beaker workers reach
(Docker Hub `shashankg209/...` or the ai2 mirror `jupiter-cs-aus-193.reviz.ai2.in:5000`), and
point `image.txt`/`env_config.image` at it.

This avoids the per-category alternative's path mismatch: our env seeds to `/workspace` and only
symlinks `/app`→cwd when `/app` is absent, but Nemotron images have a real `/app`. Per-task images
bake files into `/app` directly — **no runtime seeding, no symlink, no path rewrite.**

## Spike results (2026-06-17) — PASSED

One task (`security_task_0732` from `skill_based/mixed/security.tar.gz`) taken end-to-end:

1. `docker build` from `environment/Dockerfile` → `nemotron-spike/security_task_0732:latest` (714 MB). ✅
2. Packaged as task-data layout (`instruction.md`, `tests/`, `image.txt`). ✅
3. Driven through `SWERLVanilluxSandboxEnv(backend="docker")`: reset → bash → submit.
   - `cwd=/app`, COPY'd files present at `/app/webapp/`, baked deps present (nmap, cryptography, scapy, paramiko). ✅
   - On submit, `test.sh` pip-installed `pytest`+`flask`, ran 6 real pytest tests, wrote `reward.txt`. ✅
   - `_parse_reward()` returned `reward=0.0`, `done=True` (0 expected — no solution provided; reward *plumbing* proven by a clean pytest run, not a crash). ✅

A reward=1 path requires a real solution (model-capability, not infra) — not part of the spike.

Spike driver: `/tmp/nemotron_spike_driver.py` (throwaway).

## Remaining work

1. **`convert_nemotron_tasks.py`** (sibling of `convert_tmax_tasks.py`): untar each split → per task,
   emit the prompt row (`messages` = system + `instruction.md`, `env_config.image` = built tag,
   `ground_truth` = task_id) and copy `tests/` **verbatim** into the tarball (do NOT synthesize
   `test.sh` — Nemotron's is already correct). Write `image.txt`.
2. **Image build/push pipeline**: build per-task images from `environment/Dockerfile`, tag, push to
   the chosen registry. Builds share base+apt+pip layers → heavily cacheable; can be lazy/on-demand.
3. **Verify easy/medium splits** share this Dockerfile+task.toml layout (only `security`/mixed was
   inspected empirically).
4. **Launch script**: clone
   [scripts/general_agent/terminal/rl/qwen35_4b_base_tmax_10k_8_podman_services_dppo.sh](../general_agent/terminal/rl/qwen35_4b_base_tmax_10k_8_podman_services_dppo.sh),
   swap `DATASET` and `task_data_hf_repo`. Everything else (`--tools swerl_vanillux_sandbox`, etc.) unchanged.

## Note on SFT

The Nemotron corpus is **already in SFT** here — `hamishivi/tmax-sft-full-20260317` has splits
`nvidia__Nemotron_Terminal_Corpus__skill_based_{easy,medium,mixed}`. Only the RL/environment side
(images + task-data packaging) is missing.
