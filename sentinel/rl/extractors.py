"""Feature extractors for SB3 policies."""

from __future__ import annotations

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import omnigibson as og


class RGBCombinedExtractor(BaseFeaturesExtractor):
    """Small CNN over every RGB subspace in a Dict observation space."""

    def __init__(self, observation_space: gym.spaces.Dict, feature_size: int = 128):
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if "rgb" in key:
                og.log.info(f"obs {key} shape: {subspace.shape}")
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 4, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with th.no_grad():
                    n_flatten = cnn(th.zeros(subspace.shape)[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        return th.cat([ext(observations[k]) for k, ext in self.extractors.items()], dim=1)
