import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import pandas as pd
import time
import tracemalloc

class InitError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
NEXT_AVAILABLE_ID = 0

def get_next_id():
    global NEXT_AVAILABLE_ID
    next_id = NEXT_AVAILABLE_ID
    NEXT_AVAILABLE_ID +=1
    # return 'N' + format(next_id,"X").zfill(7)
    return f'N{NEXT_AVAILABLE_ID}'
def convert_to_iterable(val):
    """
    Converts the input to an iterable.
    - If the input is a string, returns a tuple containing the string.
    - If the input is a NumPy array, returns a tuple of its flattened elements.
    - Otherwise, attempts to convert the input to a tuple.
    """
    if isinstance(val, str):
        return (val,)
    if isinstance(val, np.ndarray):
        # Flatten the array and convert to a tuple
        return tuple(val.flatten().tolist())
    try:
        result = tuple(val)
    except TypeError:
        result = (val,)
    return result


def all_dict1_vals_in_dict2_vals(dict1_val, dict2_val) -> bool:
    """
    Checks that every element in dict1_val (converted to an iterable) is found
    in dict2_val (also converted to an iterable).

    If any element is a NumPy array, it is converted via convert_to_iterable so that
    membership tests work without ambiguity.
    """
    dict1_val_itr = convert_to_iterable(dict1_val)
    dict2_val_itr = convert_to_iterable(dict2_val)
    for val in dict1_val_itr:
        found = False
        for candidate in dict2_val_itr:
            # If both values are numpy arrays, use array_equal for a proper comparison.
            if isinstance(val, np.ndarray) and isinstance(candidate, np.ndarray):
                if np.array_equal(val, candidate):
                    found = True
                    break
            else:
                try:
                    if val == candidate:
                        found = True
                        break
                except Exception:
                    pass
        if not found:
            return False
    return True
# def convert_to_iterable(input):
#     if isinstance(input, str):
#         return (input,)

#     try:
#         result = tuple(input)
#     except TypeError:
#         result = (input,)
#     return result


# def all_dict1_vals_in_dict2_vals(dict1_val, dict2_val) -> bool:
#     dict1_val_itr = convert_to_iterable(dict1_val)
#     dict2_val_itr = convert_to_iterable(dict2_val)
#     return all(val in dict2_val_itr for val in dict1_val_itr)


def any_dict1_vals_in_dict2_vals(dict1_val, dict2_val) -> bool:
    dict1_val_itr = convert_to_iterable(dict1_val)
    dict2_val_itr = convert_to_iterable(dict2_val)
    return any(val in dict2_val_itr for val in dict1_val_itr)
from collections.abc import Iterable
import numpy as np

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
    else:
        fig = ax.figure


    assert all(
        col in dataframe.columns
        for col in [
            "Node Idx",
            "Process",
            "Start Time [s]",
            "End Time [s]",
            "Memory [B]",
        ]
    ), "Ensure the DataFrame has the necessary columns"

    dataframe["Duration"] = dataframe["End Time [s]"] - dataframe["Start Time [s]"]

    # apply a colormap to the processes based on the Memory Cost (untested)
    if cmap:
        norm = plt.Normalize(
            dataframe["Memory [B]"].min(), dataframe["Memory [B]"].max()
        )
        colors = plt.cm.get_cmap(cmap)(norm(dataframe["Memory [B]"]))
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
    
    title_fs = ax.title.get_fontsize()
    ax.xaxis.label.set_size(round(title_fs * 0.9))
    tick_labels_fs = round(title_fs * 0.75)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick_labels_fs)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick_labels_fs)
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
    # print(type(ax))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    assert all(
        col in dataframe.columns
        for col in [
            "Node Idx",
            "Process",
            "Start Time [s]",
            "End Time [s]",
            "Memory [B]",
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
    title_fs = ax.title.get_fontsize()
    ax.xaxis.label.set_size(round(title_fs * 0.9))
    tick_labels_fs = round(title_fs * 0.75)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick_labels_fs)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick_labels_fs)
    if save:
        plt.savefig(save)
    if show and not save:
        plt.show()

    return fig, ax