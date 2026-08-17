# ManiGuard

**ManiGuard** is a Python package on top of
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson that adds
**LTL safety checking**, **task-generation pipelines**, **teleop + scripted data
collection**, **VLA supervised fine-tuning**, and **policy evaluation** for robotic
manipulation in simulated households.

!!! tip "Evaluate your VLA on ManiGuard-Bench → [Run the benchmark](evaluation/run_benchmark.md)"
    Download the benchmark, serve your checkpoint, run a family across ID + OOD,
    and read the results in the paper's success × safety metrics.

<img class="mg-hero" src="index_gallery/overall_pipeline.webp" alt="ManiGuard framework: the six benchmark families, the LTL runtime monitor, and the safety-annotated dataset suite">
<p class="mg-gallery-cap"><em>The ManiGuard framework: 200 tasks across 6 families (spatial-invariance and temporal/ordering constraints, three skill levels), every rollout runtime-checked by a compiled LTL<sub>f</sub> monitor, and a safety-annotated demonstration suite (200 tasks × 40 trajectories).</em></p>

<img class="mg-hero" src="index_gallery/overview.png" alt="ManiGuard-Bench task overview">
<p class="mg-gallery-cap"><em>A glimpse of 60 base tasks (opposite-camera view), sampled across the 6 ManiGuard-Bench families.</em></p>

## Explore the docs

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started/installation.md)**

    Install the `behavior` conda env and the BEHAVIOR datasets.

-   :material-cube-outline: **[Concepts & foundations](architecture/overview.md)**

    Architecture, the env layer, the LTL safety system, and OmniGibson patches.

-   :material-factory: **[Task generation](pipelines/index.md)**

    The 6 bench families (clutter, cabinet, stack, jar, dusty, lid) — and how to add your own.

-   :material-robot-industrial: **[Data collection](data_collection/index.md)**

    SO-101 / GELLO teleop, plus the scripted datagen pipeline for SFT demos.

-   :material-brain: **[SFT](sft/index.md)**

    The model-agnostic joint dataset + per-model recipes (openpi / GR00T / SmolVLA), and collection↔eval consistency.

-   :material-broadcast: **[Evaluation](evaluation/index.md)**

    Run your checkpoint on the benchmark; success + LTL-safety checkers, engagement-gated metrics.

-   :material-database-cog: **[Sim data generation](data_collection/index.md#scripted-datagen)**

    The scripted 6-family demo-collection pipeline + RAW → LeRobot conversion.

</div>
