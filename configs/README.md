# configs/

Experiment configs consumed by the eval and datagen entrypoints.

- **`eval/`** — per-family eval configs for the benchmark runner
  (`maniguard.eval.benchmark`), one YAML per bench family. `eval/lingbot/`
  holds the LingBot-VLA variants of the same six files (absolute-action
  contract).
- **`ablation_prompt/`** — prompt tables for the instruction-format study
  (one JSON per family: the NL / LTL / no-instruction variants per task).
  Consumed by `tools/ablation_prompt/` to build dataset variants and injected
  at eval time.
- **`firsthalf/`** — truncated-horizon task variants for the sim-to-real
  study: the cabinet task is cut to its first half (open the drawer far
  enough), with a demo-calibrated `joint_open_at_least` threshold replacing
  the full-horizon goal. Applied via `--horizon-override` in both datagen and
  eval.
