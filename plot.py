from pathlib import Path

import matplotlib.pyplot as plt

# pyrefly: ignore [untyped-import]
import networkx as nx
import numpy as np
from matplotlib.patches import Patch

from config import HYPERPARAMETER_NAMES, NETWORK_NAMES, Config
from lib import Simulation, SweepResult

COLORS: dict[str, str] = {
    "normal": "tab:blue",
    "stubborn": "tab:orange",
    "manipulative": "tab:red",
}


def layout(graph: nx.Graph, seed: int) -> dict[int, tuple[float, float]]:
    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = list(nx.community.greedy_modularity_communities(graph))

    centers = nx.circular_layout(nx.complete_graph(len(communities)), scale=3)
    positions: dict[int, tuple[float, float]] = {}

    for i, nodes in enumerate(communities):
        local_positions = nx.spring_layout(graph.subgraph(nodes), seed=seed, scale=0.8)
        positions.update(
            {node: point + centers[i] for node, point in local_positions.items()}
        )

    return positions


def draw(
    cfg: Config,
    sim: Simulation,
    active_network: np.ndarray,
    step: int,
) -> None:
    figure, (opinion_axes, network_axes) = plt.subplots(1, 2, figsize=(16, 6))
    figure.suptitle(f"Network: {NETWORK_NAMES[sim.kind]}")
    values = np.asarray(sim.history)
    agent_types = np.where(
        sim.stubborn, "stubborn", np.where(sim.manipulative, "manipulative", "normal")
    )

    for index in range(cfg.n_agents):
        kind = agent_types[index]
        opinion_axes.plot(
            values[:, index],
            color=COLORS[kind],
            alpha=0.3,
            linewidth=0.8,
            zorder=1 if kind == "normal" else 2,
        )

    mean_opinion = np.mean(values[:, sim.normal], axis=1)
    opinion_variance = np.var(values, axis=1)
    (mean_line,) = opinion_axes.plot(
        mean_opinion,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Mean",
        zorder=4,
    )
    (variance_line,) = opinion_axes.plot(
        opinion_variance,
        color="tab:purple",
        linewidth=2.5,
        linestyle=":",
        label="Variance",
        zorder=4,
    )

    opinion_axes.set(
        title=f"Opinion step {step + 1}",
        xlabel="Step",
        ylabel="Opinion",
    )
    opinion_axes.legend(
        handles=[
            *[
                Patch(color=color, label=name.capitalize())
                for name, color in COLORS.items()
            ],
            mean_line,
            variance_line,
        ]
    )

    active_network_graph = nx.from_numpy_array(active_network)
    active_network_graph.remove_edges_from(nx.selfloop_edges(active_network_graph))
    full_network_graph = nx.from_numpy_array(sim.network)
    full_network_graph.remove_edges_from(nx.selfloop_edges(full_network_graph))
    full_network_graph.remove_edges_from(active_network_graph.edges)
    positions = layout(active_network_graph, cfg.seed)
    nx.draw_networkx_edges(
        full_network_graph,
        positions,
        ax=network_axes,
        edge_color="gray",
        style="dashed",
        alpha=0.1,
    )
    nx.draw_networkx_edges(
        active_network_graph,
        positions,
        ax=network_axes,
        edge_color="black",
        alpha=0.45,
    )
    nx.draw_networkx_nodes(
        active_network_graph,
        positions,
        ax=network_axes,
        node_color=[COLORS[kind] for kind in agent_types],
        node_size=300,
    )
    nx.draw_networkx_labels(
        active_network_graph,
        positions,
        ax=network_axes,
        labels={i: f"{round(sim.state[i], 2)}" for i in range(cfg.n_agents)},
        font_color="white",
        font_size=6,
    )
    network_axes.set_title("Network")
    network_axes.axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    res_path = Path("./figs/")

    if not res_path.exists():
        res_path.mkdir()
    plt.savefig(res_path / (sim.kind.name + ".png"))
    plt.show()


def plot_grid_search(
    results: SweepResult,
    x: str,
    y: str | None = None,
) -> None:
    x_label = HYPERPARAMETER_NAMES[x]
    title = NETWORK_NAMES[results.kind]
    if y is not None:
        y_label = HYPERPARAMETER_NAMES[y]

    if y is None:
        figure, ax = plt.subplots(figsize=(10, 6))
        figure.suptitle(title)
        groups = {}
        for row in results.grid_results:
            value = getattr(row, x)
            groups.setdefault(value, []).append(row)

        values = sorted(groups)
        mean_avgs = []
        var_avgs = []
        for value in values:
            rows = groups[value]
            mean_avgs.append(sum(row.mean for row in rows) / len(rows))
            var_avgs.append(sum(row.var for row in rows) / len(rows))

        ax.plot(values, mean_avgs, label="Delta mean")
        ax.plot(values, var_avgs, label="Delta variance")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Seed-averaged delta")
        ax.legend()
    else:
        figure, (mean_ax, var_ax) = plt.subplots(1, 2, figsize=(16, 6))
        figure.suptitle(title)
        groups = {}
        for row in results.grid_results:
            point = (row.k_neighbors, row.p_rewire)
            groups.setdefault(point, []).append(row)

        x_values = sorted({point[0] for point in groups})
        y_values = sorted({point[1] for point in groups})
        mean_grid = np.empty((len(y_values), len(x_values)))
        var_grid = np.empty((len(y_values), len(x_values)))
        for y_index, y_value in enumerate(y_values):
            for x_index, x_value in enumerate(x_values):
                rows = groups[(x_value, y_value)]
                mean_grid[y_index, x_index] = sum(row.mean for row in rows) / len(rows)
                var_grid[y_index, x_index] = sum(row.var for row in rows) / len(rows)

        x_grid, y_grid = np.meshgrid(x_values, y_values)
        mean_limit = float(np.max(np.abs(mean_grid))) or 1.0
        var_limit = float(np.max(np.abs(var_grid))) or 1.0
        mean_heatmap = mean_ax.pcolormesh(
            x_grid, y_grid, mean_grid, cmap="RdBu_r", vmin=-mean_limit, vmax=mean_limit
        )
        figure.colorbar(mean_heatmap, ax=mean_ax, label="Seed-averaged delta mean")
        mean_ax.set(title="Delta mean", xlabel=x_label, ylabel=y_label)

        var_heatmap = var_ax.pcolormesh(
            x_grid, y_grid, var_grid, vmin=-var_limit, vmax=var_limit
        )
        figure.colorbar(var_heatmap, ax=var_ax, label="Seed-averaged delta variance")
        var_ax.set(title="Delta variance", xlabel=x_label, ylabel=y_label)

    figure.tight_layout()
    plt.show()
