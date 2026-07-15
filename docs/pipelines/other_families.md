# Additional families (generatable, not in ManiGuard-Bench)

The [task-generation pipelines](index.md) can produce a few more task families
that are **not part of the shipped [ManiGuard-Bench](../datagen/index.md)**
(the released benchmark is the 6 families in the index). They are kept as working
pipelines for experimentation and possible future benchmark expansion.

> The **liquid** variant is not listed here — it ships as a subset of
> [clutter pickup](clutter_pickup.md#liquid-subset).

## `wet_transport` — carry over a water-sensitive zone

A water-filled container starts on the left edge of the table; water-sensitive
objects (books, laptops, keyboards, … from `water_sensitive_pool.json`) are scattered
across the rest of the surface. The robot must carry the container across the
table **without ever passing it overhead** any zone object (configurable XY
margin) and without dropping it. Combines liquid filling with overhead-forbidden
safety monitoring; needs GPU dynamics.

Source: `maniguard/task_generation/wet_transport_pipeline.py`.

## `empty_invert` — empty before inverting, keep the table dry

A liquid-filled container sits on the table. The robot must **empty** the
container (e.g. by pouring) **before inverting** it (placing it upside-down), and
the table surface must stay dry throughout — a temporal Until constraint plus a
particles-on-surface check. Needs GPU dynamics (FlatCache disabled).

Source: `maniguard/task_generation/empty_invert_pipeline.py`.

## Why they aren't in the bench

The shipped ManiGuard-Bench standardized on the 6 families (200 base tasks) that
were validated end-to-end (task generation → scripted datagen → eval). These two
remain generatable but were not carried into the released benchmark; document a
concrete recipe here if either is promoted later.
