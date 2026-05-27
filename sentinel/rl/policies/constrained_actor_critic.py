"""SB3 Dict-obs policy with parallel reward + cost value heads.

Mirrors omnisafe's ``ConstraintActorCritic``: one actor distribution, two
critics. The cost critic shares the features extractor + MLP extractor with
the reward critic by default (matches omnisafe's published configs and avoids
doubling parameter count); switch off via the policy_kwarg
``separate_cost_value_extractor=True`` when you suspect the shared trunk is
causing reward-critic regressions under constrained training.

The four public points of contact with SB3's training loop are:

* ``forward(obs) -> (actions, values_r, values_c, log_prob)`` — replaces the
  parent's 3-tuple return; ConstrainedPPO.collect_rollouts unpacks four.
* ``evaluate_actions(obs, actions) -> (values_r, values_c, log_prob, entropy)``
  — replaces the parent's 3-tuple; ConstrainedPPO.train unpacks four.
* ``predict_values(obs) -> values_r`` — unchanged signature so any SB3 code
  that only cares about reward (e.g. eval_video_proprio) keeps working.
* ``predict_values_cost(obs) -> values_c`` — new; used for cost-value bootstrap
  when an episode is truncated mid-rollout and at rollout end.

Save/load round-trips via ``_get_constructor_parameters``: the
``separate_cost_value_extractor`` flag is in the policy kwargs so
``PPO.load(...)`` reconstructs the same architecture.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


class MultiInputConstrainedActorCriticPolicy(MultiInputActorCriticPolicy):
    """Dict-obs actor-critic policy with reward + cost value heads."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        separate_cost_value_extractor: bool = False,
    ):
        self.separate_cost_value_extractor = bool(separate_cost_value_extractor)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Add the cost-value head and (optionally) its own features extractor.

        ``MultiInputActorCriticPolicy._build`` builds the reward critic and the
        optimizer with the parameters known at that point. We append the cost
        critic afterwards and rebuild the optimizer so its param list includes
        the new head's weights.
        """
        super()._build(lr_schedule)

        if self.separate_cost_value_extractor:
            self.vf_cost_features_extractor: BaseFeaturesExtractor = (
                self.make_features_extractor()
            )
        else:
            self.vf_cost_features_extractor = self.vf_features_extractor

        self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {self.cost_value_net: 1.0}
            if self.separate_cost_value_extractor:
                module_gains[self.vf_cost_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Rebuild optimizer so cost_value_net (and its extractor, if separate)
        # are in the param list. Use the initial LR from the schedule.
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    # --- helpers ----------------------------------------------------------
    def _cost_latent_vf(self, obs: PyTorchObs) -> th.Tensor:
        """Compute the cost-critic latent for the given obs.

        Falls back to the shared reward-critic latent unless we built a
        dedicated extractor for cost.
        """
        if self.separate_cost_value_extractor:
            features = super(MultiInputActorCriticPolicy, self).extract_features(
                obs, self.vf_cost_features_extractor
            )
            return self.mlp_extractor.forward_critic(features)
        # Shared trunk path: reuse parent's vf latent via predict_values.
        # We compute it directly to avoid double-running the actor head.
        features = super(MultiInputActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )
        return self.mlp_extractor.forward_critic(features)

    # --- overrides --------------------------------------------------------
    def forward(  # type: ignore[override]
        self, obs: PyTorchObs, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Same as parent but also returns the cost-value estimate."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
            latent_vf_cost = (
                self.mlp_extractor.forward_critic(
                    super(MultiInputActorCriticPolicy, self).extract_features(
                        obs, self.vf_cost_features_extractor
                    )
                )
                if self.separate_cost_value_extractor
                else latent_vf
            )
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            if self.separate_cost_value_extractor:
                vf_cost_features = super(
                    MultiInputActorCriticPolicy, self
                ).extract_features(obs, self.vf_cost_features_extractor)
                latent_vf_cost = self.mlp_extractor.forward_critic(vf_cost_features)
            else:
                latent_vf_cost = latent_vf

        values = self.value_net(latent_vf)
        values_cost = self.cost_value_net(latent_vf_cost)
        distribution = self._get_action_dist_from_latent(latent_pi)
        self._last_distribution = distribution  # for FOCOPS KL
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, values_cost, log_prob

    def evaluate_actions(  # type: ignore[override]
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor | None]:
        """Same as parent but also returns cost-value estimate."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
            if self.separate_cost_value_extractor:
                vf_cost_features = super(
                    MultiInputActorCriticPolicy, self
                ).extract_features(obs, self.vf_cost_features_extractor)
                latent_vf_cost = self.mlp_extractor.forward_critic(vf_cost_features)
            else:
                latent_vf_cost = latent_vf
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            if self.separate_cost_value_extractor:
                vf_cost_features = super(
                    MultiInputActorCriticPolicy, self
                ).extract_features(obs, self.vf_cost_features_extractor)
                latent_vf_cost = self.mlp_extractor.forward_critic(vf_cost_features)
            else:
                latent_vf_cost = latent_vf

        distribution = self._get_action_dist_from_latent(latent_pi)
        self._last_distribution = distribution  # for FOCOPS KL
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        values_cost = self.cost_value_net(latent_vf_cost)
        entropy = distribution.entropy()
        return values, values_cost, log_prob, entropy

    def predict_values_cost(self, obs: PyTorchObs) -> th.Tensor:
        """Estimate the cost-value V_c(s)."""
        latent_vf_cost = self._cost_latent_vf(obs)
        return self.cost_value_net(latent_vf_cost)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["separate_cost_value_extractor"] = self.separate_cost_value_extractor
        return data


__all__ = ["MultiInputConstrainedActorCriticPolicy"]
