"""Layer-1 family-agnostic reusable primitives (the reusable "skeleton template").

Planned modules (see doc §3, filled in Step 1):
  - ``scene``     : scene_from_task_dir (build the empty-scene env from a base task)
  - ``curobo_seg``: CuroboSegment.solve (one cuRobo segment + salvage + attach)
  - ``grasp``     : grasp_primitive (OBB sample + 2-stage standoff/servo + close + AG)
  - ``move``      : move_holding (carry a grasped object through waypoints)
  - ``contact``   : push_drawer / close_hinge / wipe_surface / extract_lateral
  - ``execute``   : execute_trajectory (JointController PD tracking)
  - ``obstacles`` : ObstacleWorld / Constraints (obstacle world + safety levers)
  - ``cameras``   : reuse bench camera_setup (4 third-person) + inject wrist
  - ``record``    : Recorder (joint-native, 5 images, both actions, sim-state dump,
                    LeRobot v2.1 write)
"""
