import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from typing import List, Tuple
import time
import tracemalloc


class InitError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
NEXT_AVAILABLE_ID = 0


def get_next_id():
    global NEXT_AVAILABLE_ID
    next_id = NEXT_AVAILABLE_ID
    NEXT_AVAILABLE_ID += 1
    # return "N" + format(next_id, "X").zfill(5)
    return f"N{NEXT_AVAILABLE_ID}"


def convert_to_iterable(input):
    if isinstance(input, str):
        return (input,)

    try:
        result = tuple(input)
    except TypeError:
        result = (input,)
    return result


def all_dict1_vals_in_dict2_vals(dict1_val, dict2_val) -> bool:
    dict1_val_itr = convert_to_iterable(dict1_val)
    dict2_val_itr = convert_to_iterable(dict2_val)
    return all(val in dict2_val_itr for val in dict1_val_itr)


def any_dict1_vals_in_dict2_vals(dict1_val, dict2_val) -> bool:
    dict1_val_itr = convert_to_iterable(dict1_val)
    dict2_val_itr = convert_to_iterable(dict2_val)
    return any(val in dict2_val_itr for val in dict1_val_itr)


def adjacent_edges_from_edge(nxGraph: nx.Graph, vertex1, vertex2) -> List[Tuple]:
    adjacent_sorted_edges = []
    for edge in nxGraph.edges(vertex1):
        if edge == (vertex1, vertex2) or edge == (vertex2, vertex1):
            continue
        # edge = list(edge)
        # edge.sort()
        adjacent_sorted_edges.append(edge)

    for edge in nxGraph.edges(vertex2):
        if edge == (vertex1, vertex2) or edge == (vertex2, vertex1):
            continue
        # edge = list(edge)
        # edge.sort()
        adjacent_sorted_edges.append(edge)

    return adjacent_sorted_edges


def adjacent_edge_idxs_from_edge(nxGraph: nx.Graph, vertex1, vertex2) -> List[int]:
    adjacent_edge_idxs = []
    edge_list = list(nxGraph.edges)
    for edge in nxGraph.edges(vertex1):
        if edge == (vertex1, vertex2) or edge == (vertex2, vertex1):
            continue
        edge_idx = find_edge_index(edge[0], edge[1], edge_list)
        adjacent_edge_idxs.append(edge_idx)

    for edge in nxGraph.edges(vertex2):
        if edge == (vertex1, vertex2) or edge == (vertex2, vertex1):
            continue
        edge_idx = find_edge_index(edge[0], edge[1], edge_list)
        adjacent_edge_idxs.append(edge_idx)

    return adjacent_edge_idxs


def find_edge_index(edge_qubit1, edge_qubit2, edge_list: List):
    try:
        edge_idx = edge_list.index((edge_qubit1, edge_qubit2))
    except ValueError:
        edge_idx = edge_list.index((edge_qubit2, edge_qubit1))

    return edge_idx


def compute_resources(func):
    def inner(self, *args, **kwargs):
        begin = time.time_ns()
        tracemalloc.start()
        current, _ = tracemalloc.get_traced_memory()

        func(self, *args, **kwargs)

        _, max = tracemalloc.get_traced_memory()
        end = time.time_ns()
        tracemalloc.stop()

        self.memory = max - current
        self.time = (end - begin) / 1e9

    return inner


def publish_gantt(
    dataframe: pd.DataFrame,
    figsize=(16, 6),
    alpha=None,
    cmap=None,
    frame_off=True,
    ax=None,
    save=None,
    show=True,
):
    """
    Creates a Gantt plot from the given DataFrame tabulating results from <Network>.run

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    assert all(
        col in dataframe.columns
        for col in [
            "Node Idx",
            "Process",
            "Start Time [s]",
            "End Time [s]",
            "Memory Cost [B]",
        ]
    ), "Ensure the DataFrame has the necessary columns"

    dataframe["Duration"] = dataframe["End Time [s]"] - dataframe["Start Time [s]"]

    # apply a colormap to the processes based on the Memory Cost (untested)
    if cmap:
        norm = plt.Normalize(
            dataframe["Memory Cost [B]"].min(), dataframe["Memory Cost [B]"].max()
        )
        colors = plt.cm.get_cmap(cmap)(norm(dataframe["Memory Cost [B]"]))
    else:
        colors = "skyblue"

    ax.barh(
        dataframe["Node Idx"] + ": " + dataframe["Process"],
        dataframe["Duration"],
        left=dataframe["Start Time [s]"],
        color=colors,
        edgecolor="k",
        alpha=alpha,
    )

    ax.xaxis.grid(True)
    if frame_off:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.set_xlabel('Time [s]')

    if save:
        plt.savefig(save)
    if show and not save:
        plt.show()

    return fig, ax

def publish_gantt2(
    dataframe: pd.DataFrame,
    figsize=(16, 6),
    alpha=None,
    cmap=None,
    frame_off=True,
    ax=None,
    save=None,
    show=True,
    top=5,
    key='Duration',
    ascending=False,
):
    """
    Creates a Gantt plot from the given DataFrame tabulating results from <Network>.run

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    assert all(
        col in dataframe.columns
        for col in [
            "Node Idx",
            "Process",
            "Start Time [s]",
            "End Time [s]",
            "Memory Cost [B]",
        ]
    ), "Ensure the DataFrame has the necessary columns"

    dataframe["Duration [s]"] = dataframe["End Time [s]"] - dataframe["Start Time [s]"]

    dataframe = dataframe.sort_values(by=key, ascending=ascending)

    # apply a colormap to the processes based on the Memory Cost (untested)
    if cmap:
        norm = plt.Normalize(
            dataframe[key].min(), dataframe[key].max()
        )
        colors = plt.cm.get_cmap(cmap)(norm(dataframe[key]))
    else:
        colors = "skyblue"

    ax.barh(
        dataframe["Node Idx"] + ": " + dataframe["Process"],
        #dataframe["Duration"],
        dataframe[key],
        #left=dataframe["Start Time [s]"],
        color=colors,
        edgecolor="k",
        alpha=alpha,
    )

    ax.xaxis.grid(True)
    if frame_off:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.set_xlabel(key)

    if save:
        plt.savefig(save)
    if show and not save:
        plt.show()

    return fig, ax