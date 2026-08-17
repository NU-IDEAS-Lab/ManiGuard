"""Layer-2 per-family manip skeletons (task-specific).

Each family implements ``executor.contracts.FamilySkeleton`` ONLY — it produces the
ordered ``MotionSegment`` list (subtask waypoints derived from the task diagnostics +
the grasp-annotation DB); the generic ``executor`` plans / executes / gates / records
them. Safety is baked into the waypoint/constraint choices (safe-by-construction) AND
verified by the executor's real-time LTL gate (doc §0.1 revised, §4.3).

Planned (clutter first = template; per-family high-level manip plan co-designed with user):
  - ``clutter`` : grasp(target) → transport(target→goal)            [template, first]
  - ``lid``     : grasp(lid) → place(lid,on=container) → grasp(container) → transport
  - ``stack``   : unstack 3 identical tops → one right-side re-stack pile → retrieve bottom → goal
  - ``dusty``   : grasp(sponge) → wipe(dest) → place(sponge) → grasp(source) → pour
  - ``jar``     : close_hinge(lid) → grasp(jar_body, side) → transport
  - ``cabinet`` : grasp(target) → place(target,in=drawer) → push_drawer(close)

``FAMILY`` maps a family name to its ``FamilySkeleton`` class; the driver instantiates it.
"""
from maniguard.data.datagen.families.cabinet import CabinetSkeleton
from maniguard.data.datagen.families.cabinet_firsthalf import CabinetFirstHalfSkeleton
from maniguard.data.datagen.families.clutter import ClutterSkeleton
from maniguard.data.datagen.families.dusty import DustySkeleton
from maniguard.data.datagen.families.jar import JarSkeleton
from maniguard.data.datagen.families.lid import LidSkeleton
from maniguard.data.datagen.families.stack import StackSkeleton

FAMILY = {
    "clutter": ClutterSkeleton,
    "cabinet": CabinetSkeleton,
    # Truncated-horizon variant of `cabinet` (phases 1-2 only); needs --horizon-override.
    "cabinet_firsthalf": CabinetFirstHalfSkeleton,
    "stack": StackSkeleton,
    "jar": JarSkeleton,
    "dusty": DustySkeleton,
    "lid": LidSkeleton,
}

