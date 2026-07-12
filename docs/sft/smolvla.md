# SmolVLA SFT

*Planned.*

SmolVLA is a candidate VLA for the same model-agnostic joint dataset used by
[openpi](openpi.md) and [GR00T](gr00t.md). No ManiGuard SmolVLA training code
exists yet.

When added, it will follow the same contract as the other tracks:

- consume a ManiGuard joint LeRobot v2.1 dataset unchanged
  (see [Dataset & data-source configs](dataset_and_config.md)),
- declare only its own state/action/camera mapping (absolute joint 8-D + one
  overview + wrist),
- eval through a joint-space `JointController` (see [end to end](end_to_end.md)).

This page is a placeholder; fill in the recipe when the SmolVLA track lands.
