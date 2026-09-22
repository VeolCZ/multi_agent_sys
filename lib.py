import itertools
from dataclasses import dataclass

# pyrefly: ignore [untyped-import]
import networkx as nx
import numpy as np

# pyrefly: ignore [untyped-import]
from networkx import barabasi_albert_graph, erdos_renyi_graph, watts_strogatz_graph

from config import Config, NetworkKind


@dataclass
class Simulation:
    kind: NetworkKind
    state: np.ndarray
    normal: np.ndarray
    stubborn: np.ndarray
    malicious: np.ndarray
    network: np.ndarray
    history: list[np.ndarray]


@dataclass
class SweepResult:
    seed: int
    mean: float
    var: float
    p_edge: float | None = None
    edges: int | None = None
    k_neighbors: int | None = None
    p_rewire: float | None = None


def init_network(
    cfg: Config,
    kind: NetworkKind,
    p_edge: float | None = None,
    edges: int | None = None,
    k_neighbors: int | None = None,
    p_rewire: float | None = None,
) -> np.ndarray:
    if p_edge is None:
        p_edge = cfg.p_edge
    if edges is None:
        edges = cfg.edges
    if k_neighbors is None:
        k_neighbors = cfg.k_neighbors
    if p_rewire is None:
        p_rewire = cfg.p_rewire

    match kind:
        case "random":
            graph = erdos_renyi_graph(cfg.n_agents, p_edge, seed=cfg.seed)
        case "barbasi":
            graph = barabasi_albert_graph(cfg.n_agents, edges, seed=cfg.seed)
        case "watts":
            graph = watts_strogatz_graph(
                cfg.n_agents, k_neighbors, p_rewire, seed=cfg.seed
            )
        case _:
            raise ValueError(f"unknown network kind: {kind}")

    network = nx.to_numpy_array(graph)
    np.fill_diagonal(network, 1)
    return network


def init_sim(cfg: Config, kind: NetworkKind) -> Simulation:
    np.random.seed(cfg.seed)
    state = np.concatenate(
        [
            np.linspace(
                cfg.initial_opinion_min, cfg.initial_opinion_max, cfg.n_normal_agents
            ),
            np.random.uniform(
                cfg.initial_opinion_min,
                cfg.initial_opinion_max,
                cfg.n_stubborn_agents,
            ),
            np.full(cfg.n_malicious_agents, cfg.malicious_opinion),
        ]
    )

    normal = np.zeros(cfg.n_agents, dtype=bool)
    stubborn = np.zeros(cfg.n_agents, dtype=bool)
    malicious = np.zeros(cfg.n_agents, dtype=bool)
    normal[: cfg.n_normal_agents] = True
    stubborn[cfg.n_normal_agents : cfg.n_normal_agents + cfg.n_stubborn_agents] = True
    malicious[cfg.n_normal_agents + cfg.n_stubborn_agents :] = True
    history = [state.copy()]
    network = init_network(cfg, kind)
    return Simulation(kind, state, normal, stubborn, malicious, network, history)


def run_one(cfg: Config, sim: Simulation, draw: bool = False) -> tuple[float, float]:
    state = sim.state
    init_state = state.copy()

    for step in range(cfg.steps + 1):
        network_copy = sim.network.copy()
        differences = np.abs(state[:, None] - state[None, :])
        network_copy[differences > cfg.epsilon] = 0

        next_state = (network_copy @ state) / network_copy.sum(axis=1)
        next_state[sim.malicious] = cfg.transform(state[sim.malicious])
        next_state[sim.stubborn] = state[sim.stubborn]
        state = next_state
        sim.history.append(state.copy())

    sim.state = state
    if draw:
        from plot import draw as draw_figure

        draw_figure(cfg, sim, network_copy, step)
    return float(np.mean(state - init_state)), float(np.var(state - init_state))


def sweep(
    kind: NetworkKind,
    grid: dict[str, list[float | int]],
    seeds: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
) -> list[SweepResult]:
    keys = tuple(grid)
    results: list[SweepResult] = []
    for combo in itertools.product(*(grid[key] for key in keys)):
        values = dict(zip(keys, combo))
        for seed in seeds:
            cfg = Config()
            cfg.seed = seed
            for key, value in values.items():
                setattr(cfg, key, value)
            mean, var = run_one(cfg, init_sim(cfg, kind))
            results.append(SweepResult(seed, mean, var, **values))
    return results
