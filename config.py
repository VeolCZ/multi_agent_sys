from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class NetworkKind(StrEnum):
    RANDOM = "random"
    BARBASI = "barbasi"
    WATTS = "watts"


@dataclass
class Config:
    n_normal_agents = 100
    n_stubborn_agents = 10
    n_manipulative_agents = 10
    n_agents = n_normal_agents + n_stubborn_agents + n_manipulative_agents
    steps = 80
    epsilon = 0.1
    seed = 1
    initial_opinion_min = -0.6
    initial_opinion_max = 0.6
    manipulative_opinion = -0.6
    manipulative_target = 1
    manipulative_speed = (manipulative_target - manipulative_opinion) / steps
    networks: tuple[NetworkKind, ...] = tuple(NetworkKind)
    p_edge = 0.5
    edges = n_agents // 2
    k_neighbors = n_agents // 2
    p_rewire = 0.1

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(x + self.manipulative_speed, self.manipulative_target)
