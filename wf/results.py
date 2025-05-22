import os

import networkx as nx
from typing import Callable
import numpy as np
import pygraphviz as pgv
import pandas as pd

from pyvis.network import Network as PyvisNetwork
from .simulation.refactor.graph import Node

import matplotlib.pyplot as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from .util.log import get_logger,logging
logger = get_logger(__name__)
# print(pgv.__version__) 
from PIL import Image


def get_default_source_path():

    return __file__

class Results:
    """Results object representation.
    This object provides methods that allow a user to rapidly publish studies, results, and visualizations of data collected by a Workflow or resource estimation process.
    Examples of usage:
    Creating a results object:
    ›› results = Results("/test-results")
    ›› os.path.exists("./test-results")
    True
    ›› import time
    ›› os.rmdir("/test-results")
    """
    def __init__(self, path=None, title=None, delay_creation=False, is_created=False, source_path=None):
        """
        Args:
            path (str or None): If provided, the output path shall be created (overwrites any existing files).
            title (str or None): The title of the results.
            delay_creation (bool): If True, generation of the output path shall be delayed.
            is_created (bool): Flag for indicating if output directory already exists.
        """

        self.path = path
        self.source_path = source_path or get_default_source_path()
        self.delay_creation = delay_creation
        self.is_created = is_created
        self.title = title
        self.published_items = {}
        if not self.delay_creation and self.path:
            self.create_output_directory(self.path)
            
    def create_output_directory(self,path):
        os.makedirs(path,exist_ok=True)
        self.is_created = True
    
    
    def publish_logical_graph_plot(self, graphable, level=None, pos=None, label_type="label", data_nodelist=None, proc_nodelist=None, edgelist=None, axes_offset=0.1, with_labels=True, label_offset=(0.1, 0.15), data_node_size=300, data_node_color='#3D89DE', proc_node_size=300, proc_node_color='#81C6DD', alpha=1, frame_off=True, ax=None, show=True, save=None):
        if not ax:
            fig, ax = mpl.subplots()
        if not isinstance(graphable, nx.DiGraph):
            g = graphable.nx_graph(level=level)
        else:
            g = graphable
        
        data_nodelist = data_nodelist or [node for node in g.nodes if g.nodes[node]["node_type"] == "data"]
        proc_nodelist = proc_nodelist or [node for node in g.nodes if g.nodes[node]["node_type"] == "process"]
        
        labels = {node: g.nodes[node][label_type] for node in g.nodes}
        # nx.draw(g, with_labels=with_labels, ax=ax)
        pos = pos or nx.shell_layout(g)
        # pos = pos or nx.planar_layout(g)
        data_nodes = nx.draw_networkx_nodes(g, pos, nodelist=data_nodelist, node_shape="^", node_size=data_node_size, node_color=data_node_color, alpha=alpha, ax=ax)
        data_nodes.set_edgecolor("#000000")
        proc_nodes = nx.draw_networkx_nodes(g, pos, nodelist=proc_nodelist, node_shape="s", node_size=proc_node_size, node_color=proc_node_color, alpha=alpha, ax=ax)
        proc_nodes.set_edgecolor("#000000")
        nx.draw_networkx_edges(g, pos, edgelist=edgelist, alpha=alpha, ax=ax)
        if with_labels:
            label_pos = {node: (x + label_offset[0], y + label_offset[1]) for node, (x, y) in pos.items()}
            for node, (x, y) in pos.items():
                if y > 0:
                    label_pos[node] = (x, y + abs(label_offset[1]))
                else:
                    label_pos[node] = (x, y - abs(label_offset[1]))
            nx.draw_networkx_labels(g, label_pos, labels=labels, ax=ax,
                                    font_size=8, font_family="Arial",font_weight ='bold', alpha=alpha)
        if axes_offset:
            if type(axes_offset) is int or type(axes_offset) is float:
                axes_offset = (-axes_offset, axes_offset, -axes_offset, axes_offset)
            if len(axes_offset) == 2:
                axes_offset = (*axes_offset, *axes_offset)
            xlims = ax.get_xlim()
            ax.set_xlim((xlims[0] + axes_offset[0], xlims[1] + axes_offset[1]))
            ylims = ax.get_ylim()
            ax.set_ylim((ylims[0] + axes_offset[2], ylims[1] + axes_offset[3]))
        if frame_off:
            ax.axis('off')
        if show and not save:
            mpl.show()
        elif save:
            save = self.path(save)
            mpl.savefig(save)
            self.add_published_item(save, "A plot of the logical graph network structure.")
    def publish_logical_graph_plot_pygraphviz(self,domain, level=None, save=None, show=True):
        data_nodelist, proc_nodelist, edges = domain.extract_graph_details(level=level)
        print("edges: ",edges)
        print("data_nodelist: ",data_nodelist)
        print("proc_nodelist: ",proc_nodelist)
        G = pgv.AGraph(strict=True, directed=True)
        # Add nodes and styles
        for node in data_nodelist:
            
            G.add_node(node.label, shape="triangle", style="filled", fillcolor="#3D89DE", fontcolor="white")
        for node in proc_nodelist:
            G.add_node(node.label, shape="box", style="filled", fillcolor="#81C6DD", fontcolor="white")

        # Add edges
        for edge in edges:
            G.add_edge(edge[0], edge[1], color="black")

        

        G.layout(prog='dot')
        if save:
            save_path = f"{self.path}/{save}"
            G.draw(save_path, format='png')

        if show:
            img = Image.open(save_path)
            img.show()
    def publish_gantt(self,ganttable,figsize=(16,6), alpha = None, cmap = None, frame_off = True,ax=None,save = None,show = True):
        if not ax:
            fig,ax = mpl.subplots(figsize=figsize)

        mpl.subplots_adjust(left=0.15)
        df = ganttable.to_dataframe()

        ax.barh(df.Process,df.Duration,left=df.Start,color =df.Color, edgecolor = 'k', joinstyle = 'round')
        ax.xaxis.grid(True)
        if frame_off:
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if show and not save:
            mpl.show()

    def publish_resource_usage_history(self,simulation_results, key, cmap=None, ax=None, show=True, save=None):
        if not ax:
            fig, ax = mpl.subplots()
        
        df = simulation_results.get_resource_usage(key)
        #df = results.get_resource_usage(key)
        cmap = cmap or "jet"
        if type(cmap) is str:
            cmap = mpl.get_cmap(cmap)
        ax.barh(df.Resource, df.End - df.Start, left=df.Start, color=cmap(df.Usage))
        if show and not save:
            mpl.show()
        elif save:
            save_path = f"resource_usage_{key}.png"
            mpl.savefig(save_path)
            print(f"Saved resource usage plot as {save_path}")
        
        
import matplotlib.pyplot as plt
import pandas as pd

def publish_gantt(dataframe: pd.DataFrame, figsize=(16, 6), alpha=None, cmap=None, frame_off=True, ax=None, save=None, show=True):
    """
    Creates a Gantt plot from the given DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the task information.
        figsize (tuple): Size of the figure.
        alpha (float): Transparency level for the bars.
        cmap (str): Colormap for the bars.
        frame_off (bool): Whether to turn off the frame around the plot.
        ax (matplotlib.axes.Axes): Axes to plot on. Creates new if None.
        save (str): Path to save the plot. Doesn't save if None.
        show (bool): Whether to display the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Ensure the DataFrame has the necessary columns
    assert all(col in dataframe.columns for col in ["Process", "Start Time [s]", "End Time [s]", "Memory [B]"]), \
        "DataFrame must contain 'Process', 'Start Time [s]', 'End Time [s]', and 'Memory [B]' columns."

    dataframe['Duration'] = dataframe['End Time [s]'] - dataframe['Start Time [s]']
    
    # Optional: Apply a colormap to the processes based on the Memory Cost
    if cmap:
        norm = plt.Normalize(
            dataframe['Memory [B]'].min(), dataframe['Memory [B]'].max())
        colors = plt.cm.get_cmap(cmap)(norm(dataframe['Memory [B]']))
    else:
        colors = 'skyblue'  # Default color
    
    # Plotting the Gantt chart
    ax.barh(dataframe['Process'], dataframe['Duration'], left=dataframe['Start Time [s]'], color=colors, edgecolor='k', alpha=alpha)
    # ax.barh(
    #     dataframe["Node Idx"] + ":" + dataframe["Process"],
    #     dataframe["Duration"],
    #     left = dataframe["Start Time [s]"],
    #     color =colors,
    #     edgecolor = "k",
    #     alpha=alpha
    # )
    # Grid and frame settings
    ax.xaxis.grid(True)
    if frame_off:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Show or save the plot
    if save:
        plt.savefig(save)
    if show and not save:
        plt.show()
def visualize_graph(data_nodes: list,
                    process_nodes: list,
                    edges: list,
                    input_nodes: list | None = None,
                    output_nodes: list | None = None,
                    labels: dict | None = None,
                    default_data_node_shape: str = 'dot',
                    default_data_node_color: str = '#a6e2f7',
                    default_data_node_size: int = 10,
                    data_node_shape_fn: Callable | None = None,
                    data_node_color_fn: Callable | None = None,
                    data_node_size_fn: Callable | None = None,
                    default_proc_node_shape: str = 'dot',
                    default_proc_node_color: str = '#a6e2f7',
                    default_proc_node_size: int = 10,
                    proc_node_shape_fn: Callable | None = None,
                    proc_node_color_fn: Callable | None = None,
                    proc_node_size_fn: Callable | None = None,
                    input_node_shape: str = 'dot',
                    input_node_color: str = '#a6e2f7',
                    input_node_size: int = 10,
                    output_node_shape: str = 'dot',
                    output_node_color: str = '#a6e2f7',
                    output_node_size: int = 10,
                    default_edge_width: int = 1,
                    default_edge_color: str = '#000000',
                    edge_width_fn: Callable | None = None,
                    edge_color_fn: Callable | None = None,
                    highlighted_data_nodes: list | None = None,
                    highlighted_data_node_shape: str = 'dot',
                    highlighted_data_node_color: str = '#a6e2f7',
                    highlighted_data_node_size: int = 10,
                    highlighted_proc_nodes: list | None = None,
                    highlighted_proc_node_shape: str = 'dot',
                    highlighted_proc_node_color: str = '#a6e2f7',
                    highlighted_proc_node_size: int = 10,
                    highlighted_edges: list | None = None,
                    highlighted_edge_width: int = 1,
                    highlighted_edge_color: str = '#000000',
                    ) -> PyvisNetwork:
    """
    This function converts a NetworkX graph (`nx.Graph`) with loaded
    attributes and information on nodes and edges into an equivalent
    PyVis network (`PyvisNetwork`) object, and converts the associated
    visualization features to match those prescribed.

    This is useful for interactive visualizations of the given graph
    object.

    Args:
        `highlighted_inputs` (`list` or `None`): (Optional) If provided, nodes in this set shall be colored with `highlighted_input_color`.
        `highlighted_inputs_color` (`str` or `None`): (Optional) If provided, highlighted input nodes shall use this color (If `None`, uses default color).
        `highlighted_outputs` (`list` or `None`): (Optional) If provided, nodes in this set shall be colored with `highlighted_output_color`.
        `highlighted_outputs_color` (`str` or `None`): (Optional) If provided, highlighted output nodes shall use this color (If `None`, uses default color).
        `highlighted_edges` (`list` or `None`): (Optional) If provided, edges in this set shall be colored with `highlighted_edges_color`.
        `highlighted_edges_color` (`str` or `None`): (Optional) If provided, highlighted edges shall use this color (If `None`, uses default color).

    Authors:
        Joel Thompson (richard.j.thompson3@boeing.com)
    
    """
    input_nodes = input_nodes or []
    output_nodes = output_nodes or []
    highlighted_data_nodes = highlighted_data_nodes or []
    highlighted_proc_nodes = highlighted_proc_nodes or []
    highlighted_edges = highlighted_edges or []


    g = PyvisNetwork(directed=True)
    
    id = 1
    for node in data_nodes:
        shape = default_data_node_shape
        color = default_data_node_color
        size = default_data_node_size
        if data_node_shape_fn is not None:
            shape = data_node_shape_fn(node)
        if data_node_color_fn is not None:
            color = data_node_color_fn(node)
        if data_node_size_fn is not None:
            size = data_node_size_fn(node)
        if node in input_nodes:
            shape = input_node_shape
            color = input_node_color
            size = input_node_size
        if node in output_nodes:
            shape = output_node_shape
            color = output_node_color
            size = output_node_size
        if node in highlighted_data_nodes:
            shape = highlighted_data_node_shape
            color = highlighted_data_node_color
            size = highlighted_data_node_size
        level = 3
        if node in input_nodes:
            level = 1
        if node in output_nodes:
            level = 99
        label = str(node),
        if labels is not None and str(node) in labels:
            label = labels[str(node)]
        g.add_node(str(node),
                   label=label,
                   #group=1,
                   #level=level,
                   shape=shape,
                   color=color,
                   size=size,
                   )
        id += 1

    for node in process_nodes:
        shape = default_proc_node_shape
        color = default_proc_node_color
        size = default_proc_node_size
        if proc_node_shape_fn is not None:
            shape = proc_node_shape_fn(node)
        if proc_node_color_fn is not None:
            color = proc_node_color_fn(node)
        if proc_node_size_fn is not None:
            size = proc_node_size_fn(node)
        if node in highlighted_proc_nodes:
            shape = highlighted_proc_node_shape
            color = highlighted_proc_node_color
            size = highlighted_proc_node_size
        label = str(node),
        if labels is not None and str(node) in labels:
            label = labels[str(node)]
        g.add_node(str(node),
                   label=label,
                   #group=2,
                   #level=2,
                   shape=shape,
                   color=color,
                   size=size,
                   )
        id += 1

    for edge in edges:
        width = default_edge_width
        color = default_edge_color
        if edge_width_fn is not None:
            width = edge_width_fn(edge)
        if edge_color_fn is not None:
            color = edge_color_fn(edge)
        if edge in highlighted_edges:
            width = highlighted_edge_width
            color = highlighted_edge_color
        g.add_edge(*edge,
                   width=width,
                   color=color,
                   )

    return g



def visualize_graph_from_nx(nxg: nx.Graph,
                            labels: dict | None = None,
                            default_data_node_shape: str = 'dot',
                            default_data_node_color: str = '#a6e2f7',
                            default_data_node_size: int = 10,
                            data_node_shape_fn: Callable | None = None,
                            data_node_color_fn: Callable | None = None,
                            data_node_size_fn: Callable | None = None,
                            default_proc_node_shape: str = 'dot',
                            default_proc_node_color: str = '#a6e2f7',
                            default_proc_node_size: int = 10,
                            proc_node_shape_fn: Callable | None = None,
                            proc_node_color_fn: Callable | None = None,
                            proc_node_size_fn: Callable | None = None,
                            input_node_shape: str = 'dot',
                            input_node_color: str = '#a6e2f7',
                            input_node_size: int = 10,
                            output_node_shape: str = 'dot',
                            output_node_color: str = '#a6e2f7',
                            output_node_size: int = 10,
                            default_edge_width: int = 1,
                            default_edge_color: str = '#000000',
                            edge_width_fn: Callable | None = None,
                            edge_color_fn: Callable | None = None,
                            highlighted_data_nodes: list | None = None,
                            highlighted_data_node_shape: str = 'dot',
                            highlighted_data_node_color: str = '#a6e2f7',
                            highlighted_data_node_size: int = 10,
                            highlighted_proc_nodes: list | None = None,
                            highlighted_proc_node_shape: str = 'dot',
                            highlighted_proc_node_color: str = '#a6e2f7',
                            highlighted_proc_node_size: int = 10,
                            highlighted_edges: list | None = None,
                            highlighted_edge_width: int = 1,
                            highlighted_edge_color: str = '#000000',
                            pg: PyvisNetwork | None = None,
                            ) -> PyvisNetwork:
    """
    This function converts a NetworkX graph (`nx.Graph`) with loaded
    attributes and information on nodes and edges into an equivalent
    PyVis network (`PyvisNetwork`) object, and converts the associated
    visualization features to match those prescribed.

    This is useful for interactive visualizations of the given graph
    object.

    Args:
        `highlighted_inputs` (`list` or `None`): (Optional) If provided, nodes in this set shall be colored with `highlighted_input_color`.
        `highlighted_inputs_color` (`str` or `None`): (Optional) If provided, highlighted input nodes shall use this color (If `None`, uses default color).
        `highlighted_outputs` (`list` or `None`): (Optional) If provided, nodes in this set shall be colored with `highlighted_output_color`.
        `highlighted_outputs_color` (`str` or `None`): (Optional) If provided, highlighted output nodes shall use this color (If `None`, uses default color).
        `highlighted_edges` (`list` or `None`): (Optional) If provided, edges in this set shall be colored with `highlighted_edges_color`.
        `highlighted_edges_color` (`str` or `None`): (Optional) If provided, highlighted edges shall use this color (If `None`, uses default color).

    Authors:
        Joel Thompson (richard.j.thompson3@boeing.com)
    
    """
    data_nodes = [x for x,y in nxg.nodes(data=True) if 'network_type' in y and y['network_type'] in [Node.INPUT, Node.OUTPUT]]
    process_nodes = [x for x,y in nxg.nodes(data=True) if 'network_type' in y and y['network_type'] == Node.NETWORK]
    edges = nxg.edges
    input_nodes = [x for x,y in nxg.nodes(data=True) if 'network_type' in y and y['network_type'] == Node.INPUT]
    output_nodes = [x for x,y in nxg.nodes(data=True) if 'network_type' in y and y['network_type'] == Node.OUTPUT]
    highlighted_data_nodes = highlighted_data_nodes or []
    highlighted_proc_nodes = highlighted_proc_nodes or []
    highlighted_edges = highlighted_edges or []
    # ----------------------------------------------------------------
    # Make sure we didn’t drop any nodes entirely
    all_nodes      = set(nxg.nodes())
    classified     = set(data_nodes) | set(process_nodes)
    missing_records = all_nodes - classified

    if missing_records:
        # Warn you (or use logger.warning)
        logger.debug(f"⚠️  visualize_graph_from_nx: tagging untyped nodes as PROCESS: {missing_records}")
        # process_nodes.extend(sorted(missing_records))
        # raise ValueError
    # logger.debug(f'data nodes: {data_nodes}')
    # logger.debug(f'process nodes: {process_nodes}')
    # logger.debug(f'input nodes: {input_nodes}')
    # logger.debug(f'output nodes: {output_nodes}')

    g = PyvisNetwork(directed=True) if pg is None else pg
    
    id = 1
    for node in data_nodes:
        shape = default_data_node_shape
        color = default_data_node_color
        size = default_data_node_size
        if data_node_shape_fn is not None:
            shape = data_node_shape_fn(node)
        if data_node_color_fn is not None:
            color = data_node_color_fn(node)
        if data_node_size_fn is not None:
            size = data_node_size_fn(node)
        if node in input_nodes:
            shape = input_node_shape
            color = input_node_color
            size = input_node_size
        if node in output_nodes:
            shape = output_node_shape
            color = output_node_color
            size = output_node_size
        if node in highlighted_data_nodes:
            shape = highlighted_data_node_shape
            color = highlighted_data_node_color
            size = highlighted_data_node_size
        level = 3
        if node in input_nodes:
            level = 1
        if node in output_nodes:
            level = 99
        label = str(node),
        if labels is not None and str(node) in labels:
            label = labels[str(node)]
        g.add_node(str(node),
                   label=label,
                   #group=1,
                   #level=level,
                   shape=shape,
                   color=color,
                   size=size,
                   )
        id += 1

    for node in process_nodes:
        shape = default_proc_node_shape
        color = default_proc_node_color
        size = default_proc_node_size
        if proc_node_shape_fn is not None:
            shape = proc_node_shape_fn(node)
        if proc_node_color_fn is not None:
            color = proc_node_color_fn(node)
        if proc_node_size_fn is not None:
            size = proc_node_size_fn(node)
        if node in highlighted_proc_nodes:
            shape = highlighted_proc_node_shape
            color = highlighted_proc_node_color
            size = highlighted_proc_node_size
        label = str(node),
        if labels is not None and str(node) in labels:
            label = labels[str(node)]
        
        g.add_node(str(node),
                   label=label,
                   #group=2,
                   #level=2,
                   shape=shape,
                   color=color,
                   size=size,
                   )
        id += 1

    for edge in edges:
        width = default_edge_width
        color = default_edge_color
        if edge_width_fn is not None:
            width = edge_width_fn(edge)
        if edge_color_fn is not None:
            color = edge_color_fn(edge)
        if edge in highlighted_edges:
            width = highlighted_edge_width
            color = highlighted_edge_color
        #print(f'edge={edge[0], edge[1]}')
        g.add_edge(str(edge[0]), str(edge[1]),
                   width=width,
                   color=color,
                   )

    return g

def publish_resource_usage_history(df, key, cmap=None, ax=None, show=True, save=None, broker=None, use_absolute_scale=False, fake=False):
    if not ax:
        fig, ax = mpl.subplots()
    #df = results.get_resource_usage(key)

    #df = df[key]
    cmap = cmap or "jet"
    if type(cmap) is str:
        cmap = mpl.get_cmap(cmap)
    #ax.barh(df.Resource, df.Duration, left=df.Start, color=cmap(df.Usage))
    #bh = ax.barh(df['Device Name'], df['End Time [s]'] - df['Start Time [s]'], left=df['Start Time [s]'], color=cmap(df[key]))
    if fake:
        vmin = 0.
        vmax = 127 if use_absolute_scale else fake
        et = np.max(df['End Time [s]'])
        df.loc[len(df)] = ['N020000', 'GSE_RunQuantumCircuit', et, et + 0.5, 'IBM Brisbane', 0, fake, 10, 0.80]
    if use_absolute_scale and not fake:
        vmin = np.min(df[key])
        vmax = 16.e9
    else:
        if not fake:
            vmin = 0.
            vmax = np.max(df[key])
    for row in df.iterrows():
        row = row[1]


        device_name = row['Device Name']
        start_time = row['Start Time [s]']
        end_time = row['End Time [s]']
        #mem_cost = row['Memory [B]']
        mem_cost = row[key]
        if mem_cost == 'N/A':
            mem_cost = 0.
        duration = end_time - start_time
        ax.barh(device_name, duration, left=start_time, color=cmap(mem_cost/vmax))
    #bh.cmap = cmap
    ax.set_xlabel("Time [s]")
    #fig.colorbar(bh, ax=ax, location='bottom')
    #mpl.colorbar(ax=ax, location='bottom')
    #if broker is not None:
    #    vmax = broker.get_max_value(key)
    norm = Normalize(vmin=vmin,vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    mpl.colorbar(sm, ticks=np.linspace(vmin, vmax, 5), location='bottom', ax=ax, label=key)
                #boundaries=np.arange(-0.05,2.1,.1))
    if show and not save:
        mpl.show()
    elif save:
        #save = self.subpath(save)
        mpl.savefig(save)
        #self.add_published_item(save, f"Plot of resource utilization over time of {key}.")
    return fig, ax
