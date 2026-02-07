# SENTINEL-Lite

## LTL Safety Checking (Local Add-On)
This fork adds optional LTL safety checking and monitoring for BehaviorTask scenes:

- Atomic propositions are generated from the BDDL object scope and predicates in `omnigibson/utils/ltl_utils.py` (`AtomicPropositionGenerator`).
- Safety constraints are loaded at task init in `omnigibson/tasks/behavior_task.py`, validated with Spot if available, and combined with `&`.
- An `LTLMonitor` (in `omnigibson/utils/ltl_utils.py`) converts LTL to an LDBA and tracks automaton state.
- Per-step LTL info is exposed via `info["ltl"]` in `omnigibson/envs/env_base.py` (also on `reset()`).
- Tests live in `tests/test_ltl_propositions.py`.

Where to add / edit constraints:

- Task-level: `bddl3/bddl/activity_definitions/<activity_name>/ltl_safety.json`
- Scene-level: `datasets/behavior-1k-assets/scenes/<scene_name>/safety/ltl_safety.json` (scene_dir resolved via `get_scene_path(scene_model)`)

Note: Spot is optional. If Spot is unavailable, safety validation and monitor init are skipped with a warning.


## RL Training (Grasping)

```bash
OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/examples/learning/grasp_policy_demo.py
```



## BEHAVIOR Server Configuration 

The current workspace has already cloned the BEHAVIOR repository, so you can directly start working on the following setup steps.

Before executing the `setup.sh`, some suggested operations:
- Put Conda in the `/data` folder, instead of the `/home` folder
- Put  `~/.cache` folders inside the `/data` folder
```bash
mkdir -p /data/<net_id>/.cache

# move or delete old cache
mv ~/.cache/* /data/<net_id>/.cache/
	# or 
rm -rf ~/.cache

ln -s /data/<net_id>/.cache ~/.cache

# Set XDG_CACHE_HOME path
echo 'export XDG_CACHE_HOME=/data/<net_id>/.cache' >> ~/.bashrc
```
- Re-assign `/tmp` folder and put it inside the `/data` folder (pip & wheel build & ... will be all saved in tmp)
```bash
# first check default tmp path and storage
echo $TMPDIR
df -h $TMPDIR # probably no enough space

# change path
mkdir -p /data/<net_id>/tmp
echo 'export TMPDIR=/data/<net_id>/tmp' >> ~/.bashrc
source ~/.bashrc

```
- Put the actual storage place for `SENTINEL-Lite/datasets` folder inside the `/data` folder, and pass it through a symbolic link
```bash
# example workflow
mkdir -p /data/<net_id>/SENTINEL-Lite
mv ~/SENTINEL-Lite/datasets /data/<net_id>/SENTINEL-Lite/datasets
ln -s /data/<net_id>/SENTINEL-Lite/datasets ~/SENTINEL-Lite/datasets
```
- Always check the available space: `df -h /home`
- **Summary**: recommend having the settings as follows:

| Type               | Suggested Path           |
| :----------------- | ------------------------ |
| TMPDIR             | `/data/<net_id>/tmp`     |
| pip / wheel build  | will follow TMPDIR       |
| OmniGibson dataset | `/data/<net_id>/...`     |
| Isaac cache        | symbolic link to `/data` |
| XDG_CACHE_HOME     | `/data/<net_id>/.cache`  |
| video / logs       | `/data/<net_id>/...`     |

Then, continue to set up BEHAVIOR (may refer to the [BEHAVIOR Installation Guide](https://behavior.stanford.edu/getting_started/installation.html#setup)).
```bash
cd SENTINEL-Lite
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
```

Normally, you should see `=== Installation Complete! ===` message from the setup script.