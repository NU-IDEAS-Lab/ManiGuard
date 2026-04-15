"""SO-101 leader arm / Franka teleop.

Moved out of ``OmniGibson/omnigibson/{teleop,examples/teleoperation}/``
so OmniGibson stays closer to upstream. Entry points:

    python -m sentinel.teleop.so101_franka_teleop --snapshot <scene_ep1.json>
    python -m sentinel.teleop.so101_franka_playback --input <teleop.hdf5>

The companion ZMQ server (for the real SO-101 leader arm) lives outside
this package at ``teleop_bridge/so101_server.py`` -- it runs in the
``lerobot`` Python 3.12 venv, distinct from the ``behavior`` conda env
these entry points expect.
"""
