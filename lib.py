from dataclasses import dataclass
from statistics import fmean, pvariance

# pyrefly: ignore [untyped-import]
import networkx as nx
import numpy as np

# pyrefly: ignore [untyped-import]
from networkx import barabasi_albert_graph, erdos_renyi_graph, watts_strogatz_graph

# pyrefly: ignore [untyped-import]
from config import Config, NetworkKind


@dataclass
class Simulation:
    kind: NetworkKind
    state: np.ndarray
    normal: np.ndarray
    stubborn: np.ndarray
    manipulative: np.ndarray
    network: np.ndarray
    history: list[np.ndarray]


@dataclass
class GridResult:
    mean: float
    var: float
    p_edge: float | None = None
    edges: int | None = None
    k_neighbors: int | None = None
    p_rewire: float | None = None


@dataclass
class SweepPoint:
    p_edge: float | None = None
    edges: int | None = None
    k_neighbors: int | None = None
    p_rewire: float | None = None


@dataclass
class SweepResult:
    highest_delta_mean: float
    average_delta_mean: float
    variance_delta_mean: float
    average_delta_variance: float
    seed_means: list[float]
    grid_results: list[GridResult]


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
        case NetworkKind.RANDOM:
            graph = erdos_renyi_graph(cfg.n_agents, p_edge, seed=cfg.seed)
        case NetworkKind.BARBASI:
            graph = barabasi_albert_graph(
                cfg.n_agents,
                edges,
                seed=cfg.seed,
                initial_graph=nx.complete_graph(edges + 1),
            )
        case NetworkKind.WATTS:
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
            np.full(cfg.n_manipulative_agents, cfg.manipulative_opinion),
        ]
    )

    normal = np.zeros(cfg.n_agents, dtype=bool)
    stubborn = np.zeros(cfg.n_agents, dtype=bool)
    manipulative = np.zeros(cfg.n_agents, dtype=bool)
    normal[: cfg.n_normal_agents] = True
    stubborn[cfg.n_normal_agents : cfg.n_normal_agents + cfg.n_stubborn_agents] = True
    manipulative[cfg.n_normal_agents + cfg.n_stubborn_agents :] = True
    permutation = np.random.permutation(cfg.n_agents)
    state = state[permutation]
    normal = normal[permutation]
    stubborn = stubborn[permutation]
    manipulative = manipulative[permutation]
    history = [state.copy()]
    network = init_network(cfg, kind)
    return Simulation(kind, state, normal, stubborn, manipulative, network, history)


def run_one(cfg: Config, sim: Simulation, draw: bool = False) -> tuple[float, float]:
    state = sim.state
    init_state = state.copy()
    norm_init_state = init_state[sim.normal]

    for step in range(cfg.steps):
        network_copy = sim.network.copy()
        differences = np.abs(state[:, None] - state[None, :])
        network_copy[differences > cfg.epsilon] = 0

        next_state = (network_copy @ state) / network_copy.sum(axis=1)
        next_state[sim.manipulative] = cfg.transform(state[sim.manipulative])
        next_state[sim.stubborn] = state[sim.stubborn]
        state = next_state
        norm_state = state[sim.normal]
        sim.history.append(state.copy())

    sim.state = state
    if draw:
        from plot import draw as draw_figure

        draw_figure(cfg, sim, network_copy, step)

    return float(np.mean(norm_state - norm_init_state)), float(
        np.var(init_state) - np.var(state)
    )


def sweep(
    kind: NetworkKind,
    points: list[SweepPoint],
    seeds: list[int],
) -> SweepResult:
    grid_results: list[GridResult] = []
    means_by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}

    for point in points:
        means: list[float] = []
        variances: list[float] = []

        for seed in seeds:
            cfg = Config()
            cfg.seed = seed

            if point.p_edge is not None:
                cfg.p_edge = point.p_edge
            if point.edges is not None:
                cfg.edges = point.edges
            if point.k_neighbors is not None:
                cfg.k_neighbors = point.k_neighbors
            if point.p_rewire is not None:
                cfg.p_rewire = point.p_rewire

            mean, var = run_one(cfg, init_sim(cfg, kind))
            means.append(mean)
            variances.append(var)
            means_by_seed[seed].append(mean)

        grid_results.append(
            GridResult(
                mean=fmean(means),
                var=fmean(variances),
                p_edge=point.p_edge,
                edges=point.edges,
                k_neighbors=point.k_neighbors,
                p_rewire=point.p_rewire,
            )
        )

    grid_means = [result.mean for result in grid_results]
    return SweepResult(
        highest_delta_mean=max(grid_means),
        average_delta_mean=fmean(grid_means),
        variance_delta_mean=pvariance(grid_means),
        average_delta_variance=fmean(result.var for result in grid_results),
        seed_means=[fmean(means_by_seed[seed]) for seed in seeds],
        grid_results=grid_results,
    )
