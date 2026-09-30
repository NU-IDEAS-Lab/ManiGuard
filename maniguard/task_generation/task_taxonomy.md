# Task-generation catalog

The released benchmark groups tasks into six manipulation families. Generators
construct candidate layouts; the saved benchmark definitions specify each task's
prompt, goal, and safety constraints.

- **Clutter pickup**: retrieve a target from surrounding objects. The tabletop
  generator is `clutter_scene_pipeline.py`; liquid-container variants use
  `liquid_transport_pipeline.py`.
- **Cabinet pickup**: open a drawer, place the target inside, and close it while
  keeping the target and obstacle upright and above the drop threshold.
  `cabinet_pickup_pipeline.py` builds an empty scene with a selected support,
  cabinet, target, and obstacle. Drawer opening starts at the configured fraction.
- **Stack retrieval**: retrieve a target beneath stacked objects.
  `stack_scene_pipeline.py` supports same-object, flat-target, and receptacle
  selection modes, using the corresponding catalogs under `utils/stack_pipeline/`.
- **Lid transport**: place a lid or cap before lifting its container.
  `lid_transport_pipeline.py` supports food and liquid selections. Pair and food
  catalogs are under `utils/lid_transport_pipeline/`.
- **Jar transport**: close a hinged jar before lifting and moving it to a goal
  sphere. `jar_transport_pipeline.py` builds an empty scene; the default hinge
  opening fraction is 0.6. Item admission compares its longest bounding-box extent
  against the body-link opening estimate after wall inset and fit margin.
- **Dusty transfer**: clean a destination with a sponge and transfer food into it.
  `dusty_transfer_pipeline.py` extends food transfer. Success checks the transfer
  predicate and a destination without dust; this conjunction alone does not
  establish that cleaning preceded food placement.

## Additional generators

`transfer_scene_pipeline.py` constructs food-transfer layouts.
`wet_transport_pipeline.py` adds water-sensitive zones, and
`empty_invert_pipeline.py` constructs empty-before-invert tasks. Their presence
in the code does not add families to the released six-family benchmark.
`empty_scene_pipeline.py` offers catalog-based layouts without a surrounding room.

## Inputs and outputs

Object selection uses packaged JSON catalogs under `utils/`, including support
surfaces, object footprints, graspable object pools, attachment pairs, and geometric
compatibility estimates. Simulation also requires the installed BEHAVIOR assets.
Compatibility filters nominate candidates; they do not certify physical success.

Task generators produce scene snapshots and `diagnostics.jsonl`, with optional
review videos. Diagnostics include selected objects, prompts, goals, and an inline
`ltl_safety` dictionary. Safety generation does not require writing a task-level
`ltl_safety.json` file. Benchmark construction and perturbation utilities live in
`maniguard/data/bench_builder/`.

## Entry points

Invoke a generator as a Python module and consult its `--help` for parameters:

```bash
python -m maniguard.task_generation.clutter_scene_pipeline --help
python -m maniguard.task_generation.cabinet_pickup_pipeline --help
python -m maniguard.task_generation.jar_transport_pipeline --help
python -m maniguard.task_generation.empty_scene_pipeline --help
```

`run_benchmark.py` provides a subprocess driver for scene-oriented generation and
standalone cabinet/jar trials. The standalone generators also expose their own
task-directory options when invoked directly.
The builder and policy evaluator operate on saved task definitions rather than
requiring regeneration for benchmark inspection.
