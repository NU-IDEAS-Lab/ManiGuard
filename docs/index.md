# ManiGuard

**ManiGuard** is a Python package on top of
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson that adds
**LTL safety checking**, **task-generation pipelines**, **teleop + scripted data
collection**, **VLA supervised fine-tuning**, **policy evaluation**, and
**reinforcement learning** for robotic manipulation in simulated households.

<div class="mg-gallery">
<img src="index_gallery/img_000.jpg" loading="lazy" alt="">
<img src="index_gallery/img_001.jpg" loading="lazy" alt="">
<img src="index_gallery/img_002.jpg" loading="lazy" alt="">
<img src="index_gallery/img_003.jpg" loading="lazy" alt="">
<img src="index_gallery/img_004.jpg" loading="lazy" alt="">
<img src="index_gallery/img_005.jpg" loading="lazy" alt="">
<img src="index_gallery/img_006.jpg" loading="lazy" alt="">
<img src="index_gallery/img_007.jpg" loading="lazy" alt="">
<img src="index_gallery/img_008.jpg" loading="lazy" alt="">
<img src="index_gallery/img_009.jpg" loading="lazy" alt="">
<img src="index_gallery/img_010.jpg" loading="lazy" alt="">
<img src="index_gallery/img_011.jpg" loading="lazy" alt="">
<img src="index_gallery/img_012.jpg" loading="lazy" alt="">
<img src="index_gallery/img_013.jpg" loading="lazy" alt="">
<img src="index_gallery/img_014.jpg" loading="lazy" alt="">
<img src="index_gallery/img_015.jpg" loading="lazy" alt="">
<img src="index_gallery/img_016.jpg" loading="lazy" alt="">
<img src="index_gallery/img_017.jpg" loading="lazy" alt="">
<img src="index_gallery/img_018.jpg" loading="lazy" alt="">
<img src="index_gallery/img_019.jpg" loading="lazy" alt="">
<img src="index_gallery/img_020.jpg" loading="lazy" alt="">
<img src="index_gallery/img_021.jpg" loading="lazy" alt="">
<img src="index_gallery/img_022.jpg" loading="lazy" alt="">
<img src="index_gallery/img_023.jpg" loading="lazy" alt="">
<img src="index_gallery/img_024.jpg" loading="lazy" alt="">
<img src="index_gallery/img_025.jpg" loading="lazy" alt="">
<img src="index_gallery/img_026.jpg" loading="lazy" alt="">
<img src="index_gallery/img_027.jpg" loading="lazy" alt="">
<img src="index_gallery/img_028.jpg" loading="lazy" alt="">
<img src="index_gallery/img_029.jpg" loading="lazy" alt="">
<img src="index_gallery/img_030.jpg" loading="lazy" alt="">
<img src="index_gallery/img_031.jpg" loading="lazy" alt="">
<img src="index_gallery/img_032.jpg" loading="lazy" alt="">
<img src="index_gallery/img_033.jpg" loading="lazy" alt="">
<img src="index_gallery/img_034.jpg" loading="lazy" alt="">
<img src="index_gallery/img_035.jpg" loading="lazy" alt="">
<img src="index_gallery/img_036.jpg" loading="lazy" alt="">
<img src="index_gallery/img_037.jpg" loading="lazy" alt="">
<img src="index_gallery/img_038.jpg" loading="lazy" alt="">
<img src="index_gallery/img_039.jpg" loading="lazy" alt="">
<img src="index_gallery/img_040.jpg" loading="lazy" alt="">
<img src="index_gallery/img_041.jpg" loading="lazy" alt="">
<img src="index_gallery/img_042.jpg" loading="lazy" alt="">
<img src="index_gallery/img_043.jpg" loading="lazy" alt="">
<img src="index_gallery/img_044.jpg" loading="lazy" alt="">
<img src="index_gallery/img_045.jpg" loading="lazy" alt="">
<img src="index_gallery/img_046.jpg" loading="lazy" alt="">
<img src="index_gallery/img_047.jpg" loading="lazy" alt="">
<img src="index_gallery/img_048.jpg" loading="lazy" alt="">
<img src="index_gallery/img_049.jpg" loading="lazy" alt="">
<img src="index_gallery/img_050.jpg" loading="lazy" alt="">
<img src="index_gallery/img_051.jpg" loading="lazy" alt="">
<img src="index_gallery/img_052.jpg" loading="lazy" alt="">
<img src="index_gallery/img_053.jpg" loading="lazy" alt="">
<img src="index_gallery/img_054.jpg" loading="lazy" alt="">
<img src="index_gallery/img_055.jpg" loading="lazy" alt="">
<img src="index_gallery/img_056.jpg" loading="lazy" alt="">
<img src="index_gallery/img_057.jpg" loading="lazy" alt="">
<img src="index_gallery/img_058.jpg" loading="lazy" alt="">
<img src="index_gallery/img_059.jpg" loading="lazy" alt="">
<img src="index_gallery/img_060.jpg" loading="lazy" alt="">
<img src="index_gallery/img_061.jpg" loading="lazy" alt="">
<img src="index_gallery/img_062.jpg" loading="lazy" alt="">
<img src="index_gallery/img_063.jpg" loading="lazy" alt="">
<img src="index_gallery/img_064.jpg" loading="lazy" alt="">
<img src="index_gallery/img_065.jpg" loading="lazy" alt="">
<img src="index_gallery/img_066.jpg" loading="lazy" alt="">
<img src="index_gallery/img_067.jpg" loading="lazy" alt="">
<img src="index_gallery/img_068.jpg" loading="lazy" alt="">
<img src="index_gallery/img_069.jpg" loading="lazy" alt="">
<img src="index_gallery/img_070.jpg" loading="lazy" alt="">
<img src="index_gallery/img_071.jpg" loading="lazy" alt="">
<img src="index_gallery/img_072.jpg" loading="lazy" alt="">
<img src="index_gallery/img_073.jpg" loading="lazy" alt="">
<img src="index_gallery/img_074.jpg" loading="lazy" alt="">
<img src="index_gallery/img_075.jpg" loading="lazy" alt="">
<img src="index_gallery/img_076.jpg" loading="lazy" alt="">
<img src="index_gallery/img_077.jpg" loading="lazy" alt="">
<img src="index_gallery/img_078.jpg" loading="lazy" alt="">
<img src="index_gallery/img_079.jpg" loading="lazy" alt="">
<img src="index_gallery/img_080.jpg" loading="lazy" alt="">
<img src="index_gallery/img_081.jpg" loading="lazy" alt="">
<img src="index_gallery/img_082.jpg" loading="lazy" alt="">
<img src="index_gallery/img_083.jpg" loading="lazy" alt="">
<img src="index_gallery/img_084.jpg" loading="lazy" alt="">
<img src="index_gallery/img_085.jpg" loading="lazy" alt="">
<img src="index_gallery/img_086.jpg" loading="lazy" alt="">
<img src="index_gallery/img_087.jpg" loading="lazy" alt="">
<img src="index_gallery/img_088.jpg" loading="lazy" alt="">
<img src="index_gallery/img_089.jpg" loading="lazy" alt="">
<img src="index_gallery/img_090.jpg" loading="lazy" alt="">
<img src="index_gallery/img_091.jpg" loading="lazy" alt="">
<img src="index_gallery/img_092.jpg" loading="lazy" alt="">
<img src="index_gallery/img_093.jpg" loading="lazy" alt="">
<img src="index_gallery/img_094.jpg" loading="lazy" alt="">
<img src="index_gallery/img_095.jpg" loading="lazy" alt="">
<img src="index_gallery/img_096.jpg" loading="lazy" alt="">
<img src="index_gallery/img_097.jpg" loading="lazy" alt="">
<img src="index_gallery/img_098.jpg" loading="lazy" alt="">
<img src="index_gallery/img_099.jpg" loading="lazy" alt="">
</div>
<p class="mg-gallery-cap"><em>100 task instances generated by the ManiGuard pipelines (sampled from the 6fam-base set).</em></p>

## Explore the docs

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started/installation.md)**

    Install the `behavior` conda env and the BEHAVIOR datasets.

-   :material-cube-outline: **[Concepts & foundations](architecture/overview.md)**

    Architecture, the env layer, the LTL safety system, and OmniGibson patches.

-   :material-factory: **[Task generation](pipelines/index.md)**

    Clutter, stack, transfer, lid / liquid / wet, cabinet, jar — and how to add your own.

-   :material-robot-industrial: **[Data collection](data_collection/index.md)**

    SO-101 / GELLO teleop and scripted cuRobo demo collection.

-   :material-brain: **[SFT](sft/end_to_end.md)**

    Controller / action / state / eval consistency, plus the OpenPI recipes.

-   :material-broadcast: **[Evaluation](evaluation/index.md)**

    Websocket policy eval, goal checking, and benchmark preparation.

-   :material-school: **[Reinforcement learning](rl/index.md)**

    SB3 PPO grasp training + the GraspGen / cuRobo grasp-reset pipeline.

</div>

## Repository layout

```
.
├── maniguard/            # ManiGuard package (LTL, task-gen, envs, data, eval, serve, rl)
├── behavior-1k/         # submodule → StanfordVL/BEHAVIOR-1K @ v3.7.2
├── tests/               # maniguard-side pytest suites
├── configs/             # eval / RL / SFT training configs
├── scripts/             # shell entrypoints
├── tools/               # one-off utilities
└── docs/                # this site
```

**Upstream boundary:** anything under `behavior-1k/` is upstream. Don't modify
that tree — patch behaviors via `maniguard._omnigibson_patches` instead.

## Building these docs locally

```bash
pip install mkdocs-material
mkdocs serve            # preview at http://127.0.0.1:8000
mkdocs build --strict   # build to ./site, fail on warnings
```
