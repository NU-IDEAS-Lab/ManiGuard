# RL Progress Snapshot

Last updated: 2026-05-05

## Current Pipeline

当前推进的是一套独立于 RGB 路线的 proprio / privileged RL pipeline，核心入口是：

- Training: `python -m sentinel.rl.algorithms.ppo_proprio_goal`
- Eval video: `python -m sentinel.rl.algorithms.eval_video_proprio`
- Task: `PickAndLiftPrivilegedTask`

Obs 不是 RGB。当前 policy input 是：

- robot proprio
- `target_pos_robot_frame`
- `goal_pos_robot_frame`
- `obj_to_goal_vector`
- `goal_radius`

设计意图：让 policy 明确知道 target object 和 green sphere goal region 的相对位置，避免 RGB-only 或 object-center-only 的 ambiguity。

## Reward / Task Design

当前 reward 已从早期的 touching-based grasp reward 改成更严格的 grasp/carry 结构：

- `pregrasp_dist_coeff`: approach target 的 dense shaping
- `grasp_acquire_reward`: 第一次从 not holding -> holding 时给一次 reward
- `carry_dist_coeff`: 只有 holding=True 时，根据 object 到 goal 的 signed progress 给 shaping
- `drop_penalty`: 曾经 holding 后又 lost grasp 时给小 penalty
- `goal_bonus`: object 到达 goal region 时的大 sparse bonus

Holding 判定不再用简单 `Touching`，而是走 OG 的 `robot.is_grasping(..., candidate_obj=target)`，并要求 object 有最小 lift threshold。这样避免 policy 学到“碰着物体摆烂”。

## Stability Fixes

训练中遇到过 PhysX / quaternion NaN crash，例如 eef link orientation 变成 `[nan, nan, nan, nan]`。后来在 vector env wrapper 里加了 sim fault recovery：

- 捕获 recoverable `og.sim.step()` / post-step fault
- 从错误路径里解析 bad scene id
- reset 对应 env
- 当前 step 返回 zero reward / skipped info
- W&B 记录 `sim_fault/recovered_total`

这不是物理层根治，但能防止单个 env 的 NaN 直接杀死整个 32-env training run。

## Latest Training Result

最新完整训练 run：

- Output dir: `outputs/rl_ppo_proprio_goal_assisted_h600_recovery_resume6p5m_to10m`
- W&B run: `ppo_proprio_goal_assisted_h600_recovery_5090_n32_resume6p5m_to10m`
- W&B URL: `https://wandb.ai/yiyanpeng2027-northwestern-university/sentinel-grasp-reset/runs/z9084bh5`
- Grasp mode: `assisted`
- Task horizon: 600 steps
- Resume checkpoint: around 6.5M steps
- Final global step: `10,002,080`
- Final checkpoint: `outputs/rl_ppo_proprio_goal_assisted_h600_recovery_resume6p5m_to10m/ppo_proprio_goal_final.zip`
- Latest periodic checkpoint: `outputs/rl_ppo_proprio_goal_assisted_h600_recovery_resume6p5m_to10m/ckpts/ppo_proprio_goal_9999776_steps.zip`
- Sim fault recoveries: 4 total
- Training finished normally, not crashed.

Final train metrics were not good:

- `rollout/ep_len_mean`: 601
- `rollout/ep_rew_mean`: about `-0.486`
- No evidence of frequent goal success / +50 goal bonus.

## Latest Eval Result

Latest eval video:

- Video: `outputs/rl_ppo_proprio_goal_eval_assisted_h600_final_wrist/videos/episode_000.mp4`
- Eval JSON: `outputs/rl_ppo_proprio_goal_eval_assisted_h600_final_wrist/eval_video_ppo_proprio_goal_final_20260503-195218.json`
- Checkpoint: final 10M assisted h600 model
- Camera: wrist
- Max steps: 600
- Reward: `0.2168`
- Length: 600
- Success: false
- Done: false
- Truncated: true

Visual interpretation: gripper approaches near the target / pre-grasp area, then keeps hovering or circling around the object. It does not form a stable grasp, lift, or transport to the goal sphere.

## Interpretation

Current policy has learned partial approach behavior, but it is stuck before real grasp acquisition. The main issue now is probably not "goal info missing" anymore; goal region is in obs. The bigger issue is that the policy lacks a strong grasp prior / grasp affordance signal.

Object-center relative position is useful for approach, but it does not define a valid grasp pose. For mug/goblet-like objects, "go to object center" is not enough to infer finger placement, approach direction, gripper closure timing, or stable lift. Reward shaping alone may be becoming too indirect.

My current read:

- System plumbing is now usable.
- NaN crash is partially contained.
- Privileged goal obs issue is fixed.
- Reward is less misaligned than before.
- But grasp skill itself has not emerged from PPO under this setup.

## Likely Next Step

Before more blind long training, the next useful direction is to add a grasp prior / curriculum:

- use validated grasp poses from dataset/reset pipeline if available
- start episodes closer to grasp-ready states
- train close + lift as a stage before full transport
- then fine-tune full pick-and-place-to-goal

This is more principled than adding many ad hoc reward terms around object center.

## Commit Note

Current code changes are still uncommitted. `outputs/` is gitignored, so checkpoint/video artifacts will not be saved by Git commit. If shutting down Vast, commit code/docs and separately back up important artifacts if needed.
