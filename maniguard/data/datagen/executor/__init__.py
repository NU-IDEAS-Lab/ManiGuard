"""Shared execution, geometry, grasp scoring, and acceptance helpers.

Family skeletons provide motion segments and runtime hooks through contracts.py.
The engine plans and executes segments, records trajectories, and evaluates goal,
LTL, and family-specific acceptance checks. Families may also use shared planning
helpers while selecting auxiliary grasps.
"""
