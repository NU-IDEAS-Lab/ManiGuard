"""GELLO and SO-101 teleoperation of a simulated Franka.

Entry points:
    python -m maniguard.data.teleop.gello_franka_teleop --snapshot <scene_ep1.json>
    python -m maniguard.data.teleop.so101_franka_teleop --snapshot <scene_ep1.json>
    python -m maniguard.data.teleop.so101_franka_playback --input <teleop.hdf5>

The SO-101 leader bridge is ``teleop_bridge/so101_server.py``. Run it in
an environment with the LeRobot hardware dependencies; the simulation
entry points use the OmniGibson environment.
"""
