"""Layer-2 per-family manip skeletons (task-specific).

Each module declares a family's subtask sequence + per-step waypoints (derived from
the task's diagnostics: object world poses + goal_region), composing the Layer-1
primitives. Safety is baked into the waypoint/constraint choices (collection data is
success+safe by construction; no LTL check during collection — see doc §0.1, §4).

Planned (filled in Step 2/3; the high-level manip plan per family is co-designed
with the user):
  - ``clutter`` : grasp(target) → transport(target→goal)            [template, first]
  - ``lid``     : grasp(lid) → place(lid,on=container) → grasp(container) → transport
  - ``stack``   : grasp(bottom,side) → extract_lateral → transport
  - ``dusty``   : grasp(sponge) → wipe(dest) → place(sponge) → grasp(source) → pour
  - ``jar``     : close_hinge(lid) → grasp(jar) → transport
  - ``cabinet`` : grasp(target) → place(target,in=drawer) → push_drawer(close)
"""
