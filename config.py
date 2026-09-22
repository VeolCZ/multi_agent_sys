from dataclasses import dataclass
from typing import Literal

import numpy as np

NetworkKind = Literal["random", "barbasi", "watts"]


@dataclass
class Config:
    n_normal_agents = 100
    n_stubborn_agents = 10
    n_malicious_agents = 10
    n_agents = n_normal_agents + n_stubborn_agents + n_malicious_agents
    steps = 80
    epsilon = 0.1
    seed = 1
    initial_opinion_min = -0.6
    initial_opinion_max = 0.6
    malicious_opinion = -0.6
    malicious_target = 1
    malicious_speed = (malicious_target - malicious_opinion) / steps
    networks: tuple[NetworkKind, ...] = ("random", "barbasi", "watts")
    p_edge = 0.5
    edges = 3
    k_neighbors = 4
    p_rewire = 0.1

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.minimum(x + self.malicious_speed, self.malicious_target)
