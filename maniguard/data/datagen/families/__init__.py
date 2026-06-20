"""Layer-2 per-family manip skeletons (task-specific).

Each family implements ``executor.contracts.FamilySkeleton`` ONLY — it produces the
ordered ``MotionSegment`` list (subtask waypoints derived from the task diagnostics +
the grasp-annotation DB); the generic ``executor`` plans / executes / gates / records
them. Safety is baked into the waypoint/constraint choices (safe-by-construction) AND
verified by the executor's real-time LTL gate (doc §0.1 revised, §4.3).

Planned (clutter first = template; per-family high-level manip plan co-designed with user):
  - ``clutter`` : grasp(target) → transport(target→goal)            [template, first]
  - ``lid``     : grasp(lid) → place(lid,on=container) → grasp(container) → transport
  - ``stack``   : grasp(bottom,side) → extract_lateral → transport
  - ``dusty``   : grasp(sponge) → wipe(dest) → place(sponge) → grasp(source) → pour
  - ``jar``     : close_hinge(lid) → grasp(jar) → transport
  - ``cabinet`` : grasp(target) → place(target,in=drawer) → push_drawer(close)

``FAMILY`` maps a family name to its ``FamilySkeleton`` class; the driver instantiates it.
"""
from maniguard.data.datagen.families.clutter import ClutterSkeleton

FAMILY = {
    "clutter": ClutterSkeleton,
}

