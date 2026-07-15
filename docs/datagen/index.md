# Scripted datagen

The **`maniguard/data/datagen/`** pipeline turns the read-only **ManiGuard-Bench**
base tasks into large numbers of **success + safe** demonstrations — fully scripted,
no teleop, no per-trajectory human review. The only human input is the grasp source:
each grasp is **authored once** in a GUI and stored as a 6-DOF end-effector pose.

<video autoplay muted loop playsinline controls width="768" style="max-width:100%; border-radius:6px;">
  <source src="datagen_6fam_montage.mp4" type="video/mp4">
</video>
<p style="text-align:center;"><em>One representative scripted demonstration per family — collected fully autonomously.</em></p>

```
grasp annotation DB ─►┐
                      ├─►  cuRobo planning  ─►  per-family motion segments  ─►  demo
frozen bench task  ─► ┘         (execute + record every step)                   │
                                                                success gate ───┤ keep only
                                                                LTL safety gate ┘ success+safe
```

!!! abstract "Design in one line"
    A *generic executor* plans / executes / gates / records / scales every family
    identically; each *family skeleton* only declares **which motion segments make up
    the task**. They meet at one interface (`MotionSegment` + `FamilySkeleton`), so
    adding a family is "new task semantics, everything else reused."

## The six stages

Three groups — **prep** (once per object set), **collection** (per base task), and
**offline** repackaging — documented as one step per page:

```
   ┌─ one-time, per object set ──────────────────────────────────────────┐
1. extract_meshes      bench tasks ─► object-local GLB + bbox + gripper mesh   (1 OG session)
2. annotate_tool       viser GUI   ─► grasp_annotations.json (eef-target poses) (NO sim)
   └─────────────────────────────────────────────────────────────────────┘
   ┌─ collection, per base task (one OG process each) ───────────────────┐
3. driver.run_task     base task   ─► N success+safe demos (reset-reuse loop) (sim)
4. sweep               many tasks  ─► one subprocess per task, sharded/parallel
   └─────────────────────────────────────────────────────────────────────┘
   ┌─ offline, no sim ───────────────────────────────────────────────────┐
5. review              demos       ─► per-task montage MP4 (quick visual review)
6. to_lerobot          demos       ─► LeRobot v2.1 per family (video passthrough)
   └─────────────────────────────────────────────────────────────────────┘
```

| Step | Page |
|---|---|
| 1–2 · mesh extraction + grasp annotation | [Grasp annotation](annotation.md) |
| 3–4 · executor + multi-task sweep | [Collection](collection.md) |
| 5–6 · review montage + RAW → LeRobot | [Review & conversion](conversion.md) |
| per-family task semantics + gotchas | [Families & gotchas](families.md) |

## Pre-collected datasets & continued collection

The task source is
**[ManiGuard-Bench](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/ManiGuard-Bench)**
— 6 families, **200 base tasks**, each with a clean in-distribution (ID) instance plus
4 out-of-distribution (OOD) perturbations (appearance / language / location /
environment). This pipeline has already collected a large demonstration set against
**every** base task, and it is **open-ended**: point it at any task and collect as many
more demos as you want.

**Shipped v1 datasets (public on Hugging Face).** For every base task, **40
mutually-distinct, success + LTL-safe** trajectories — **8,000 episodes / ~11.6 M
frames** total — published as **LeRobot v2.1** (8-D joint state/action, 4 third-person +
1 wrist camera, 30 FPS, 256×256):

| Family (bench) | Base tasks | Episodes (×40) | Frames | HF dataset (`<org>/…`) |
|---|---:|---:|---:|---|
| [`clutter_pickup`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-clutter-v1-joint-5cam) | 55 | 2,200 | 901,520 | `datagen-clutter-v1-joint-5cam` |
| [`cabinet_pickup`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-cabinet-v1-joint-5cam) | 35 | 1,400 | 4,172,962 | `datagen-cabinet-v1-joint-5cam` |
| [`stack_retrieve`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-stack-v1-joint-5cam) | 28 | 1,120 | 2,652,083 | `datagen-stack-v1-joint-5cam` |
| [`jar_transport`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam) | 26 | 1,040 | 946,870 | `datagen-jar-v1-joint-5cam` |
| [`dusty_transfer`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-dusty-v1-joint-5cam) | 26 | 1,040 | 1,879,498 | `datagen-dusty-v1-joint-5cam` |
| [`lid_transport`](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-lid-v1-joint-5cam) | 30 | 1,200 | 1,055,142 | `datagen-lid-v1-joint-5cam` |
| **Total** | **200** | **8,000** | **11,608,075** | |

**Continued collection.** The bench tasks + this pipeline are the durable artifacts; the
shipped 40/task is a starting point, not a cap. Run the [driver or sweep](collection.md)
against any base task with a higher success target, then append with the
[converter](conversion.md) — the format is identical, so a family's dataset grows to
whatever scale training needs.

## Environment & prerequisites

Every step: `conda activate behavior`, `PYTHONPATH` = repo root. Sim steps (1, 3–5)
also need `OMNIGIBSON_HEADLESS=1` +
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` and run with `python -u`
(the `exit 139` segfault at `og.sim.stop()` is a benign teardown). Step 6 runs in a
separate `lerobot` env.

!!! warning "Install the bench tasks + robot asset first"
    The pipeline reads base tasks from a local copy of the bench, and every sim step
    loads the longfinger Franka (details in
    [Installation](../getting-started/installation.md)):

    ```bash
    # bench tasks — the source scenes the executor replays
    hf download IDEAS-Lab-Northwestern/ManiGuard-Bench --repo-type dataset \
      --local-dir outputs/lerobot_datasets/maniguard-bench
    # robot asset — required by every sim step (auto-picked up on `import maniguard`)
    hf download IDEAS-Lab-Northwestern/franka-panda-longfinger --repo-type dataset \
      --local-dir behavior-1k/datasets/omnigibson-robot-assets/models/franka/franka_panda_longfinger
    ```

**Next step →** [Grasp annotation](annotation.md)
