#!/usr/bin/env python3
"""CPO: Constrained Policy Optimisation.

Algorithm design ported from omnisafe's CPO
(omnisafe/algorithms/on_policy/second_order/cpo.py, Apache-2.0).

CPO replaces PPO's clipped surrogate with a constrained natural policy
gradient update. The key steps in each ``train()`` call:

1. **Full-batch policy gradient**: g = ∇_θ E[ratio · A_r], b = ∇_θ E[ratio · A_c]
2. **Natural gradients** via conjugate-gradient + Fisher-vector product:
   x_a = F⁻¹g,  x_b = F⁻¹b
3. **QP projection** (three cases, from CPO paper §4):
   - Feasible + safe TRPO step  → reward-only natural gradient
   - Feasible + violates cost   → project onto cost constraint manifold
   - Infeasible                 → feasibility recovery step
4. **Line search** on the actor parameters to honour the KL trust region
5. **Critic update**: standard Adam minibatches on reward + cost value MSE

The value function uses a *separate* Adam optimizer (``_critic_optimizer``)
that does NOT touch actor parameters, so the natural gradient step is
preserved. The two optimizers work on disjoint parameter sets.

Compatible with ``run_training`` — keeps ``.learn()`` / ``.save()`` /
``.load()`` intact.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.distributions as td
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

from sentinel.rl.buffers.focops_rollout_buffer import FOCOPSDictRolloutBuffer
from sentinel.rl.cli.common import (
    add_cost_args,
    add_env_args,
    add_training_args,
    add_video_args,
    add_wandb_args,
    validate_env_args,
)
from sentinel.rl.safe.constrained_ppo import ConstrainedPPO
from sentinel.rl.safe.trpo_utils import (
    conjugate_gradients,
    flat_grad,
    flat_params,
    set_flat_params,
)


# -----------------------------------------------------------------------
# Parameter-set helpers
# -----------------------------------------------------------------------

def _actor_params(policy) -> list[th.Tensor]:
    """Parameters that exclusively affect the action distribution.

    Includes the features extractor (actor branch), the actor MLP, the
    action net, and log_std. When ``share_features_extractor=True``, the
    shared extractor is included here (and excluded from the critic list).
    """
    params = (
        list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    if hasattr(policy, "log_std"):
        params.append(policy.log_std)
    # features_extractor is always the actor's extractor (aliased when shared)
    params += list(policy.features_extractor.parameters())
    return params


def _critic_params(policy) -> list[th.Tensor]:
    """Parameters for the value function heads.

    Only the *separate* parts of the critic are included. When features are
    shared with the actor, we exclude the shared extractor here (it is
    already in the natural-gradient set); only the MLP value branch and the
    scalar heads are updated.
    """
    params = (
        list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters())
        + list(policy.cost_value_net.parameters())
    )
    if not policy.share_features_extractor:
        params += list(policy.vf_features_extractor.parameters())
    return params


# -----------------------------------------------------------------------
# Fisher-vector product
# -----------------------------------------------------------------------

def _fisher_vector_product(
    policy,
    obs_tensor,
    old_mu: th.Tensor,
    old_log_std: th.Tensor,
    v_flat: th.Tensor,
    actor_param_list: list[th.Tensor],
    damping: float = 0.1,
) -> th.Tensor:
    """Compute (F + damping·I)·v via Pearlmutter double-backward on KL.

    F is the Fisher information matrix of the current policy at the rollout
    data. We use KL(π_θ || π_θ_old) as a proxy; its Hessian is the Fisher.
    """
    # Get current distribution (actor forward only — skip value heads)
    dist_wrapper = policy.get_action_distribution(obs_tensor)
    curr_dist: td.Normal = dist_wrapper.distribution
    old_dist = td.Normal(loc=old_mu, scale=old_log_std.exp())

    # Mean KL (summed over action dims, meaned over batch)
    kl = td.kl_divergence(curr_dist, old_dist).sum(dim=-1).mean()

    # First backward: ∇_θ KL (with create_graph so we can diff again)
    kl_grads = th.autograd.grad(kl, actor_param_list, create_graph=True)
    flat_kl_grad = th.cat([g.contiguous().view(-1) for g in kl_grads])

    # Dot with v (the direction)
    kl_v = (flat_kl_grad * v_flat.detach()).sum()

    # Second backward: Hessian·v
    hv = th.autograd.grad(kl_v, actor_param_list)
    flat_hv = th.cat(
        [
            g.contiguous().view(-1) if g is not None else th.zeros_like(p).view(-1)
            for g, p in zip(hv, actor_param_list)
        ]
    )
    return flat_hv + damping * v_flat


# -----------------------------------------------------------------------
# CPO step-direction logic
# -----------------------------------------------------------------------

def _cpo_step_direction(
    x_a: th.Tensor,
    x_b: th.Tensor,
    p: float,
    q: float,
    r: float,
    s: float,
    c: float,
    target_kl: float,
    eps: float = 1e-8,
) -> tuple[th.Tensor, str]:
    """Compute the CPO update direction from QP dual variables.

    Parameters (scalars from the natural-gradient computations):
        x_a   = F⁻¹ g  (natural reward gradient)
        x_b   = F⁻¹ b  (natural cost gradient)
        p = g · x_a    (reward curvature)
        q = b · x_b    (cost curvature)
        r = g · x_b    (cross term)
        s = b · x_b    (== q, included separately for clarity)
        c             (current constraint violation: ep_cost - cost_limit)
        target_kl     (δ, trust-region budget)

    Returns (step_direction, case_name).
    """
    delta = target_kl

    # --- Case A: feasible, TRPO step respects cost ---
    lam_a = max(float((p / (2 * delta + eps)) ** 0.5), eps)
    cost_at_trpo = c + r / lam_a

    if c <= 0 and cost_at_trpo <= 0:
        # Unconstrained TRPO step is cost-safe
        step = x_a / lam_a
        return step, "feasible_trpo"

    # --- Case B: feasible but TRPO violates cost / Case C: infeasible ---
    # Solve the QP: maximise g·v  s.t.  v^T F v ≤ 2δ,  c + b^T v ≤ 0
    # Optimal dual: λ² = (p·s − r²) / (2δ·s − c²)
    #               ν  = (c·λ + r) / s       (≥ 0 enforced)
    denom_qp = 2.0 * delta * s - c ** 2

    if denom_qp > eps:
        lam_qp_sq = max((p * s - r ** 2) / denom_qp, 0.0)
        lam_qp = max(lam_qp_sq ** 0.5, eps)
        nu_qp = max((c * lam_qp + r) / max(s, eps), 0.0)
        step = (x_a - nu_qp * x_b) / lam_qp
        case = "projected" if c <= 0 else "projected_infeasible"
        return step, case

    # --- Case D: infeasible and QP has no solution in trust region ---
    # Take pure feasibility step: minimise cost subject to KL ≤ δ
    lam_b = max((s / (2 * delta + eps)) ** 0.5, eps)
    step = -x_b / lam_b
    return step, "feasibility"


# -----------------------------------------------------------------------
# CPO algorithm
# -----------------------------------------------------------------------

class CPO(ConstrainedPPO):
    """Constrained Policy Optimisation — natural gradient + QP projection.

    CPO does NOT use a Lagrange multiplier. Instead, it constrains the policy
    update by projecting the natural policy gradient onto the cost-constraint
    manifold, then doing a trust-region line search.

    Critic parameters (value heads) are updated separately with standard Adam
    after the natural gradient step so the two updates don't interfere.
    """

    def __init__(
        self,
        *args: Any,
        target_kl_delta: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        line_search_max_iters: int = 10,
        line_search_backtrack_coef: float = 0.8,
        n_value_epochs: int = 10,
        critic_lr: float = 3e-4,
        **kwargs: Any,
    ) -> None:
        self.target_kl_delta = float(target_kl_delta)
        self.cg_iters = int(cg_iters)
        self.cg_damping = float(cg_damping)
        self.line_search_max_iters = int(line_search_max_iters)
        self.line_search_backtrack_coef = float(line_search_backtrack_coef)
        self.n_value_epochs = int(n_value_epochs)
        self.critic_lr = float(critic_lr)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Swap to FOCOPS buffer (stores old mu/log_std for FVP)
        self.rollout_buffer = FOCOPSDictRolloutBuffer(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            cost_gamma=self.cost_gamma,
            cost_gae_lambda=self.cost_gae_lambda,
        )
        # Separate critic optimizer — updates only value heads
        self._critic_optimizer = th.optim.Adam(
            _critic_params(self.policy), lr=self.critic_lr
        )

    # ------------------------------------------------------------------
    # These two hooks are not used by CPO's custom train(), but must be
    # implemented so ConstrainedPPO's abstract structure is satisfied.
    # ------------------------------------------------------------------
    def _compute_adv_surrogate(
        self, adv_r: th.Tensor, adv_c: th.Tensor
    ) -> th.Tensor:
        # CPO handles cost in train(); only reward advantage is used here
        return adv_r

    # ------------------------------------------------------------------
    # Rollout: same as FOCOPS — stores (mu_old, log_std_old) per step
    # ------------------------------------------------------------------
    def collect_rollouts(  # type: ignore[override]
        self,
        env,
        callback: BaseCallback,
        rollout_buffer: FOCOPSDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, values_cost, log_probs = self.policy(obs_tensor)
                mu_old, log_std_old = _extract_dist_params(self.policy)

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            cost_rewards = np.array(
                [float(info.get("cost", 0.0)) for info in infos], dtype=np.float32
            )

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_value_c = self.policy.predict_values_cost(terminal_obs)[0]
                    rewards[idx] += self.gamma * float(terminal_value)
                    cost_rewards[idx] += rollout_buffer.cost_gamma * float(terminal_value_c)

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                cost_reward=cost_rewards,
                cost_value=values_cost,
                mu_old=mu_old,
                log_std_old=log_std_old,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(new_obs_tensor)
            values_cost = self.policy.predict_values_cost(new_obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones, last_cost_values=values_cost
        )

        ep_cost_mean = float(rollout_buffer.cost_rewards.sum(axis=0).mean())
        ep_cost_max = float(rollout_buffer.cost_rewards.sum(axis=0).max())
        self.logger.record("rollout/ep_cost_mean", ep_cost_mean)
        self.logger.record("rollout/ep_cost_max", ep_cost_max)
        self._last_ep_cost_mean = ep_cost_mean

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Training: CPO natural gradient update
    # ------------------------------------------------------------------
    def train(self) -> None:  # noqa: C901
        """CPO update: natural gradient + QP projection + line search.

        Policy parameters are updated ONCE (no n_epochs loop) using the
        natural gradient direction. Value functions are updated with standard
        Adam over n_value_epochs minibatches.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        ep_cost = getattr(self, "_last_ep_cost_mean", 0.0)
        c = ep_cost - self.cost_limit  # constraint violation (neg = feasible)

        clip_range = self.clip_range(self._current_progress_remaining)

        # ------ 1. Get full-batch rollout data -------------------------
        full_batch = next(iter(self.rollout_buffer.get(batch_size=None)))
        obs = full_batch.observations
        actions = full_batch.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        adv_r = full_batch.advantages
        adv_c = full_batch.cost_advantages
        if self.normalize_advantage and len(adv_r) > 1:
            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = adv_c - adv_c.mean()

        old_log_prob = full_batch.old_log_prob
        old_mu = full_batch.mu_old       # (batch, act_dim)
        old_log_std = full_batch.log_std_old

        actor_param_list = _actor_params(self.policy)

        # ------ 2. Policy gradient (reward) ----------------------------
        _, log_prob, _ = _eval_actor(self.policy, obs, actions)
        ratio = th.exp(log_prob - old_log_prob)
        pi_loss = -(ratio * adv_r).mean()
        g = flat_grad(pi_loss, actor_param_list, retain_graph=True)

        # ------ 3. Cost gradient ----------------------------------------
        cost_pi_loss = (ratio * adv_c).mean()
        b = flat_grad(cost_pi_loss, actor_param_list)

        # ------ 4. Fisher-vector product closure ------------------------
        def fvp(v: th.Tensor) -> th.Tensor:
            return _fisher_vector_product(
                self.policy, obs, old_mu, old_log_std,
                v, actor_param_list, damping=self.cg_damping
            )

        # ------ 5. Natural gradients via conjugate gradient -------------
        # NOTE: fvp uses double-backward (Pearlmutter trick) and MUST run
        # outside th.no_grad() — the computation graph is needed for the
        # second backward. Each fvp call creates and immediately frees its
        # graph, so there is no memory accumulation.
        x_a = conjugate_gradients(fvp, g.detach(), n_steps=self.cg_iters)
        x_b = conjugate_gradients(fvp, b.detach(), n_steps=self.cg_iters)
        x_a = x_a.detach()
        x_b = x_b.detach()

        # ------ 6. QP scalars -------------------------------------------
        Ax_a = fvp(x_a).detach()
        Ax_b = fvp(x_b).detach()
        p_scalar = float((g * x_a).sum().item())
        q_scalar = float((b * x_b).sum().item())
        r_scalar = float((g * x_b).sum().item())
        s_scalar = q_scalar  # = b · x_b

        # ------ 7. Determine step direction (QP projection) -------------
        step_dir, optim_case = _cpo_step_direction(
            x_a, x_b,
            p_scalar, q_scalar, r_scalar, s_scalar, c,
            target_kl=self.target_kl_delta,
        )
        self.logger.record("train/cpo_optim_case", optim_case)

        # ------ 8. Line search on actor params --------------------------
        old_flat = flat_params(actor_param_list).clone()
        accept = False
        for k in range(self.line_search_max_iters):
            alpha = self.line_search_backtrack_coef ** k
            new_flat = old_flat + alpha * step_dir
            set_flat_params(actor_param_list, new_flat)

            with th.no_grad():
                _, lp_new, _ = _eval_actor(self.policy, obs, actions)
                ratio_new = th.exp(lp_new - old_log_prob)
                log_ratio_new = lp_new - old_log_prob
                kl_approx = float(
                    th.mean((th.exp(log_ratio_new) - 1) - log_ratio_new).item()
                )

            if kl_approx <= self.target_kl_delta * 1.5:
                accept = True
                break

        if not accept:
            set_flat_params(actor_param_list, old_flat)

        # ------ 9. Value function update (separate Adam, n_value_epochs) --
        value_losses: list[float] = []
        cost_value_losses: list[float] = []
        for _vep in range(self.n_value_epochs):
            for vbatch in self.rollout_buffer.get(self.batch_size):
                v_obs = vbatch.observations
                v_actions = vbatch.actions
                if isinstance(self.action_space, spaces.Discrete):
                    v_actions = v_actions.long().flatten()

                values, values_cost, _, _ = self.policy.evaluate_actions(
                    v_obs, v_actions
                )
                values = values.flatten()
                values_cost = values_cost.flatten()

                value_loss = th.nn.functional.mse_loss(vbatch.returns, values)
                cost_value_loss = th.nn.functional.mse_loss(
                    vbatch.cost_returns, values_cost
                )
                total_vf_loss = (
                    self.vf_coef * value_loss + self.vf_cost_coef * cost_value_loss
                )
                value_losses.append(value_loss.item())
                cost_value_losses.append(cost_value_loss.item())

                self._critic_optimizer.zero_grad()
                total_vf_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    _critic_params(self.policy), self.max_grad_norm
                )
                self._critic_optimizer.step()

        # ------ 10. Logging --------------------------------------------
        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        explained_var_cost = explained_variance(
            self.rollout_buffer.cost_values.flatten(),
            self.rollout_buffer.cost_returns.flatten(),
        )

        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/cost_value_loss", float(np.mean(cost_value_losses)))
        self.logger.record("train/cpo_accept", int(accept))
        self.logger.record("train/cpo_constraint_violation", c)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/explained_variance_cost", explained_var_cost)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str, **kwargs: Any) -> None:  # type: ignore[override]
        # Temporarily stash and restore critic optimizer to make SB3's save work
        crit_opt = self._critic_optimizer
        self._critic_optimizer = None  # type: ignore[assignment]
        try:
            super().save(path, **kwargs)
        finally:
            self._critic_optimizer = crit_opt
        th.save(
            crit_opt.state_dict(),
            str(path).replace(".zip", "") + "_critic_opt.pt",
        )

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "CPO":  # type: ignore[override]
        model: CPO = super().load(path, **kwargs)  # type: ignore[assignment]
        opt_path = str(path).replace(".zip", "") + "_critic_opt.pt"
        import os as _os

        if _os.path.exists(opt_path):
            model._critic_optimizer.load_state_dict(
                th.load(opt_path, map_location="cpu")
            )
        return model


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _eval_actor(
    policy, obs, actions
) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
    """Evaluate actions and return (values, log_prob, entropy).

    Thin wrapper so we can call from the training loop without unpacking
    the 4-tuple every time.
    """
    values, _, log_prob, entropy = policy.evaluate_actions(obs, actions)
    return values, log_prob, entropy


def _extract_dist_params(
    policy,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dist = getattr(policy, "_last_distribution", None)
    if dist is None or not hasattr(dist, "distribution"):
        return None, None
    raw = dist.distribution
    if not isinstance(raw, td.Normal):
        return None, None
    with th.no_grad():
        return raw.loc.cpu().numpy(), raw.scale.log().cpu().numpy()


# -----------------------------------------------------------------------
# CLI argument group
# -----------------------------------------------------------------------

def add_cpo_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("cpo")
    g.add_argument("--target-kl-delta", type=float, default=0.01,
                   help="KL trust-region budget δ.")
    g.add_argument("--cg-iters", type=int, default=10,
                   help="Conjugate-gradient iterations for F⁻¹g.")
    g.add_argument("--cg-damping", type=float, default=0.1,
                   help="Tikhonov damping added to the Fisher matrix.")
    g.add_argument("--line-search-max-iters", type=int, default=10,
                   help="Max backtracking steps in the line search.")
    g.add_argument("--line-search-backtrack-coef", type=float, default=0.8,
                   help="Per-step multiplicative factor for backtracking.")
    g.add_argument("--n-value-epochs", type=int, default=10,
                   help="Critic update epochs after each policy step.")
    g.add_argument("--critic-lr", type=float, default=3e-4,
                   help="Learning rate for the separate critic optimizer.")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def _build_cost_fns(args):
    from sentinel.rl.costs import ConstantCost, ZeroCost

    if args.cost_source == "constant":
        return [ConstantCost(1.0)]
    return [ZeroCost()]


def parse_args():
    p = argparse.ArgumentParser(
        description="CPO: Constrained Policy Optimisation on PickAndLiftPrivilegedTask."
    )
    add_env_args(p)
    add_training_args(p)
    add_wandb_args(p)
    add_video_args(p)
    add_cost_args(p)
    add_cpo_args(p)
    p.add_argument(
        "--grasping-mode",
        choices=["physical", "assisted", "sticky"],
        default="physical",
    )
    return p.parse_args()


def main():
    start_time = time.time()

    from sentinel.rl.algorithms.ppo_proprio import (
        _KitLogTailer,
        _log_stage,
        _make_proprio_scene_copy,
    )

    _log_stage("parsing CLI and validating env args")
    args = parse_args()
    validate_env_args(args)
    args.task_type = "PickAndLiftPrivilegedTask"

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _log_stage("creating proprio scene override")
    source_scene_file = Path(args.scene_file).resolve()
    proprio_scene_file, patched_robots = _make_proprio_scene_copy(
        source_scene_file, out, grasping_mode=args.grasping_mode
    )
    args.scene_file = proprio_scene_file
    print(f"  source scene:      {source_scene_file}", flush=True)
    print(f"  task type:         {args.task_type}", flush=True)
    print(f"  cost source:       {args.cost_source}", flush=True)
    print(f"  cost limit:        {args.cost_limit}", flush=True)
    print(f"  target_kl_delta:   {args.target_kl_delta}", flush=True)
    print(f"  cg_iters:          {args.cg_iters}", flush=True)

    _log_stage("starting Kit log watcher")
    kit_tailer = _KitLogTailer(since=start_time)
    kit_tailer.start()

    _log_stage("importing OmniGibson and setting macros")
    from omnigibson.macros import gm

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = True
    if os.environ.get("OMNIGIBSON_HEADLESS", "0") == "1":
        gm.HEADLESS = True

    _log_stage("importing CPO / SB3 helpers")
    from stable_baselines3.common.utils import set_random_seed

    from sentinel.rl.envs.cost_wrapper import CostInjectingVecEnvWrapper
    from sentinel.rl.envs.wrappers import build_vec_env
    from sentinel.rl.training.trainer import run_training

    _log_stage(f"building OmniGibson vec env (num_envs={args.num_envs})")
    vec_env = build_vec_env(args, out_dir=out)
    cost_fns = _build_cost_fns(args)
    vec_env = CostInjectingVecEnvWrapper(vec_env, cost_fns)
    _log_stage("vec env + cost wrapper ready")
    set_random_seed(args.seed)

    tensorboard_log_dir = out / f"tb_{time.strftime('%Y%m%d-%H%M%S')}"

    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise SystemExit(f"--resume-from does not exist: {args.resume_from}")
        _log_stage("loading CPO checkpoint")
        model = CPO.load(
            str(args.resume_from),
            env=vec_env,
            tensorboard_log=str(tensorboard_log_dir),
            device="cuda",
        )
    else:
        _log_stage("constructing fresh CPO model")
        model = CPO(
            env=vec_env,
            verbose=1,
            tensorboard_log=str(tensorboard_log_dir),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            cost_limit=args.cost_limit,
            cost_gamma=args.cost_gamma,
            cost_gae_lambda=args.cost_gae_lambda,
            vf_cost_coef=args.vf_cost_coef,
            target_kl_delta=args.target_kl_delta,
            cg_iters=args.cg_iters,
            cg_damping=args.cg_damping,
            line_search_max_iters=args.line_search_max_iters,
            line_search_backtrack_coef=args.line_search_backtrack_coef,
            n_value_epochs=args.n_value_epochs,
            critic_lr=args.critic_lr,
            device="cuda",
            seed=args.seed,
        )

    try:
        _log_stage("entering shared training loop")
        run_training(
            model,
            args,
            algo_name="cpo",
            extra_wandb_config={
                "obs_mode": "proprio_target_goal_state",
                "task_type": args.task_type,
                "grasping_mode": args.grasping_mode,
                "cost_source": args.cost_source,
                "cost_limit": args.cost_limit,
                "target_kl_delta": args.target_kl_delta,
                "cg_iters": args.cg_iters,
                "cg_damping": args.cg_damping,
                "n_value_epochs": args.n_value_epochs,
                "source_scene_file": str(source_scene_file),
            },
        )
    finally:
        kit_tailer.stop()


if __name__ == "__main__":
    main()
