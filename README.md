<h1 align="center">ManiGuard</h1>

<p align="center">
  <img src="docs/index_gallery/overview.png" alt="ManiGuard-Bench task overview" width="760">
</p>

<p align="center">
  A safety-aware benchmark &amp; toolkit for vision-language-action manipulation,
  built on <a href="https://github.com/StanfordVL/BEHAVIOR-1K">BEHAVIOR-1K</a> / OmniGibson.
</p>

<p align="center">
  <a href="https://nu-ideas-lab.github.io/ManiGuard/"><b>📖 Documentation</b></a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/IDEAS-Lab-Northwestern/ManiGuard-Bench">🤗 Benchmark</a>
  &nbsp;·&nbsp;
  <a href="#quick-start">🚀 Quick start</a>
  &nbsp;·&nbsp;
  <a href="#citation">📝 Citation</a>
</p>

## Overview

**ManiGuard** is a safety-aware benchmark and toolkit for vision-language-action
(VLA) manipulation policies. It pairs a **frozen evaluation benchmark** — six tabletop
task families, each with an in-distribution base task and **four out-of-distribution
perturbation levels** (target / language / location / env) — with **LTL-based safety
monitoring**, so a policy is scored on **two axes: task success *and* safety**, not task
completion alone. A capable-but-reckless policy that finishes the task while knocking a
glass off the table is *not* a pass.

Around the benchmark, ManiGuard provides the full **data-to-eval pipeline**:
LTL-monitored **task generation**, **teleop + scripted data collection**, model-agnostic
**supervised fine-tuning** (openpi / GR00T / SmolVLA), and websocket **policy
evaluation** with contact-gated engagement metrics. It is built on BEHAVIOR-1K /
OmniGibson (NVIDIA Isaac Sim). Reinforcement-learning training is under development.

## Documentation & resources

Everything below is covered in depth on the **[documentation site](https://nu-ideas-lab.github.io/ManiGuard/)**:

| Topic | What's there |
|---|---|
| [Getting started](https://nu-ideas-lab.github.io/ManiGuard/getting-started/installation/) | Install the `behavior` env + BEHAVIOR datasets; download the benchmark + robot asset |
| [Concepts & foundations](https://nu-ideas-lab.github.io/ManiGuard/architecture/overview/) | Architecture, the env layer, the [LTL safety system](https://nu-ideas-lab.github.io/ManiGuard/foundations/ltl_safety/), OmniGibson patches |
| [Task generation](https://nu-ideas-lab.github.io/ManiGuard/pipelines/) | The 6 bench families and how to add your own pipeline |
| [Data collection](https://nu-ideas-lab.github.io/ManiGuard/data_collection/) | SO-101 / GELLO teleop and the scripted datagen pipeline for SFT demos |
| [Fine-Tuning (SFT)](https://nu-ideas-lab.github.io/ManiGuard/fine_tuning/) | The model-agnostic joint dataset + per-model recipes (openpi / GR00T / SmolVLA) |
| [Evaluation](https://nu-ideas-lab.github.io/ManiGuard/evaluation/) | Websocket policy eval, goal + [contact-gated safety](https://nu-ideas-lab.github.io/ManiGuard/evaluation/engagement_metric/) scoring |

**Datasets & checkpoints** (Hugging Face collections):
[**Benchmark & Datasets**](https://huggingface.co/collections/IDEAS-Lab-Northwestern/maniguard-benchmark-and-datasets-6a83d488178bcba81688cd4e)
— the ManiGuard-Bench evaluation scenes, the long-finger Franka robot asset, and the
six-family SFT demonstration datasets; and
[**Evaluated VLA Checkpoints**](https://huggingface.co/collections/IDEAS-Lab-Northwestern/maniguard-evaluated-vla-checkpoints-6a83d481b7af91f0daaf8221)
— every fine-tuned checkpoint behind the paper's rollout evaluations.

## Quick start

```bash
# 1. clone with submodules
git clone --recursive https://github.com/NU-IDEAS-Lab/ManiGuard.git && cd ManiGuard

# 2. create the `behavior` conda env + download the BEHAVIOR-1K assets
cd behavior-1k
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
cd ..

# 3. install maniguard (editable, with the websocket policy-server extras)
conda activate behavior
pip install -e ".[serve]"
```

Then grab the benchmark's **robot asset + scenes** and run a policy — see
[**Getting started**](https://nu-ideas-lab.github.io/ManiGuard/getting-started/installation/)
for the downloads and [**Evaluation**](https://nu-ideas-lab.github.io/ManiGuard/evaluation/)
to serve a checkpoint and score it.

## Repository layout

```
.
├── maniguard/            # ManiGuard Python package (all maniguard-owned code)
│   ├── _omnigibson_patches.py   # runtime patches on vanilla OmniGibson
│   ├── object_states/   #   Dropped, Upright
│   ├── utils/           #   ltl_utils, safety_monitor, task_spec, geometry
│   ├── task_generation/ #   clutter / cabinet / stack / jar / lid / dusty / transfer / liquid pipelines
│   ├── envs/            #   scene registry + frozen-snapshot runtime (no live env class)
│   ├── data/            #   datagen (scripted SFT demos), bench_builder, teleop, lerobot, real_teleop, scene + playback
│   ├── eval/            #   benchmark runner, goal checker, scene discovery
│   ├── {openpi,gr00t,smolvla}_sft/  # per-model SFT configs / embodiment
│   └── serve/           #   websocket VLA policy server (openpi_native)
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K
├── docs/                # documentation site (mkdocs sources)
├── configs/             # eval / SFT training configs
├── tools/               # SFT drivers + per-family bench-surgery utilities
├── scripts/             # shell entry points
├── tests/               # maniguard-side tests
├── teleop_bridge/       # ZMQ bridge for SO-101 teleop
└── vla_models/          # VLA checkpoints (user-downloaded, .gitignore)
```

## Citation

A paper describing ManiGuard is in preparation — **citation coming soon**.

```bibtex
@misc{maniguard2026,
  title  = {ManiGuard},
  author = {NU-IDEAS Lab},
  year   = {2026},
  note   = {Preprint coming soon}
}
```

## License

ManiGuard is released under the **Apache License 2.0** — see [LICENSE](LICENSE).
The BEHAVIOR-1K assets it builds on are separately licensed by Stanford and are not
redistributed here.

## Acknowledgements

Built on [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson
(StanfordVL, NVIDIA Isaac Sim), and integrates the
[openpi](https://github.com/Physical-Intelligence/openpi),
[GR00T](https://github.com/NVIDIA/Isaac-GR00T), and
[LeRobot](https://github.com/huggingface/lerobot) VLA stacks for SFT and serving.
