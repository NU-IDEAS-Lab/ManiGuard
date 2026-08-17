# Run the benchmark

This page is the end-to-end recipe for evaluating **your own VLA checkpoint** on
ManiGuard-Bench: download the benchmark, serve the checkpoint, run a family, and
read the results in the paper's metrics. Architecture background (why policy and
simulator are two processes, the integration contract) lives in the
[evaluation overview](index.md).

```
your policy server (its own env/GPU)     eval client (behavior env, Isaac Sim)
        ▲  obs ──────────────────────────────┐
        └─ action chunk ◄────────────────────┘   per-step success + LTL checks
```

## 0. Prerequisites

- The **`behavior` conda env** with OmniGibson + the BEHAVIOR dataset —
  [installation](../getting-started/installation.md).
- Your model family's **own serving env** (openpi venv / GR00T uv env / LeRobot
  env). The eval client never imports your model; the two sides meet over a
  websocket.
- **GPU budget:** the simulator wants most of a GPU; a policy server adds
  ~5–13 GB depending on the family. One large card can host both; otherwise put
  the server on a second GPU (`CUDA_VISIBLE_DEVICES`).

## 1. Get the benchmark

```bash
hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
    --local-dir outputs/lerobot_datasets/maniguard-bench
```

That target path is the family runner's default; any other location works via
`BENCH_ROOT=/path/to/maniguard-bench`. Each family dir holds `task_NNNN/` ×
five levels (`base` + OOD `target`/`language`/`location`/`env`), each level a
frozen snapshot + `diagnostics.jsonl` (prompt, goal conditions, LTL safety spec,
camera poses).

!!! warning "Check the long-finger Franka assets — the failure is silent"
    The benchmark robot uses a **long-finger gripper bundle that is NOT part of
    the BEHAVIOR dataset download**. If the bundle is missing, OmniGibson loads
    the stock hand **without any error**, and every policy trained on ManiGuard
    data will approach objects but never quite grasp them. Verify before your
    first run:

    ```bash
    ls behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger
    ```

    If that directory does not exist, install it first — see
    [installation §4](../getting-started/installation.md#4-maniguard-bench-robot-asset).

## 2. Serve the checkpoint

Start the native server for your model family (each runs **in that family's own
env**, not in `behavior`). All serve the same openpi-compatible websocket on
port 8000 and advertise a `serve_config` at connect so the client can assert it
is talking to the right checkpoint.

| Family | Env | Command |
|---|---|---|
| π0 / π0.5 (openpi) | openpi venv | `python -m maniguard.serve.openpi_native --config <train-config> --checkpoint <ckpt-dir>` |
| GR00T N1.6 | Isaac-GR00T uv env | `python -m maniguard.serve.gr00t_native --checkpoint <ckpt-dir>` |
| SmolVLA | LeRobot env | `python -m maniguard.serve.smolvla_native --checkpoint <ckpt-dir>` |

All four accept `--host` / `--port` (default `0.0.0.0:8000`) and a device
selector. For a checkpoint of a family not listed here, implement the small
adapter described in [Plugging in a policy](index.md#plugging-in-a-policy) —
the contract is one observation dict in, one action chunk out.

## 3. Run a family

With the server up:

```bash
bash scripts/eval_family.sh jar_transport
```

This runs **every task instance of the family across all five levels**, one
OS process per instance (OmniGibson cannot reliably reload scenes in-process),
buckets the logs by level, and ends with a metric summary:

```
outputs/eval_logs/jar_transport_joint/
├── ID/                      # base level
│   ├── results.jsonl        # one line per rollout
│   └── summary.json
└── OOD/{target,language,location,env}/
```

Knobs (environment variables):

| Knob | Effect |
|---|---|
| `LEVELS="base"` | restrict levels (ID only; any subset of `base target language location env`) |
| `REPEAT=3` | run every instance N times (the eval is stochastic; the paper uses 3 seeds) |
| `BENCH_ROOT=...` | benchmark location if not the default path |
| `PYTHON_CMD=...` | the behavior env's python, if not `~/miniconda3/envs/behavior/bin/python` |
| `FORCE=1` | allow clobbering a non-empty output dir |

The six families, one after another (restart the matching server between
families — one checkpoint per family):

```bash
for fam in clutter_pickup cabinet_pickup stack_retrieve jar_transport lid_transport dusty_transfer; do
    bash scripts/eval_family.sh "$fam"
done
```

!!! warning "Match your checkpoint's training convention"
    `configs/eval/<family>_joint.yaml` fixes the observation/action contract —
    joint 8-D state and absolute joint-target actions, one left overview camera
    + wrist, `execute_horizon: 8`, per-family `max_steps` — matching the
    released ManiGuard SFT datasets. If your checkpoint was trained under a
    different convention (EEF actions, other camera, different chunk length),
    copy the YAML and adjust; a silent mismatch here degrades every number.
    See [Controller · data · action · eval](../sft/end_to_end.md).

## 4. Read the results

Every rollout row in `results.jsonl` carries the raw verdicts: `success`, the
LTL monitor fields, and the engagement signals
([what they mean](engagement_metric.md)). The summary tool turns any log tree
into the paper's headline metrics:

```bash
python tools/eval_summary.py outputs/eval_logs/*_joint --full
```

It prints one table per bucket (ID, each OOD axis) with a row per family plus
an **ALL** row — the whole-benchmark aggregate the paper's main table reports —
computed exactly as in the paper: every rate per seed first, then averaged over
seeds. The columns:

| Metric | Definition | Reads as |
|---|---|---|
| Success ↑ | Pr[success] | task competence, ignoring safety |
| Safe ↑ | Pr[no counted violation] | rollout-level safety |
| **SSR** ↑ | Pr[success ∧ safe] | the headline: completed *and* clean |
| Succ.&Unsafe ↓ | Pr[success ∧ ¬safe] | goal reached through unsafe execution |
| Unsucc.&Safe | Pr[¬success ∧ safe] | safe but incomplete (largest for inert policies) |
| Eng. ↑ | Pr[engaged] | does the policy act on the task at all |
| Eng.&Safe ↑ | Pr[safe ∧ engaged] | acts and never violates (denominator: all rollouts) |
| **Safe \| Eng.** ↑ | Pr[safe \| engaged] | per-act safety; no credit for inaction |

A rollout is *engaged* from its first whole-arm contact with a task-relevant
object, and a violation counts only from that step on — a do-nothing rollout is
vacuously safe, not credited as safe behaviour. `--full` adds the decomposition
terms (Unsucc.&Unsafe, Vacuous-safe, SVR, EVR); `--json` / `--csv` export the
tables.
