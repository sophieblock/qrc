
VERSION = "1.1"
PYVIS_OUTPUT_ID = 'pyvis'


from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import matplotlib.pyplot as mpl
from matplotlib.colors import rgb2hex
import numpy as np
import asyncio
import datetime as dt
from io import StringIO, BytesIO
from tempfile import NamedTemporaryFile
# import openpyxl as xl
from pathlib import Path
import faicons as fa
#from d3graph import d3graph, vec2adjmat
from pyvis.network import Network as PyvisNetwork
import networkx as nx
import os
import logging
logging.basicConfig(level=logging.DEBUG)
import sys
from typing import Iterable
import builtins

# Set the Z3 library directory
builtins.Z3_LIB_DIRS = ["/Users/so714f/opt/anaconda3/envs/wf_env/lib/python3.12/site-packages/z3/lib"]
os.environ['Z3_LIBRARY_PATH'] = "/Users/so714f/opt/anaconda3/envs/wf_env/lib/python3.12/site-packages/z3/lib"

# Now import the necessary modules
try:
    from z3 import Optimize, Int, IntVector, And, Or, Bool, Implies, If, sat
except ImportError:
    print("Warning: z3 is not available. Some features may be disabled.")
# set up a static_assets folder for holding the Network()'s html file
DIR = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.join(DIR, "www")

if not os.path.exists(WWW):
    os.makedirs(WWW)





#cmap = mpl.colormaps['plasma']
purplesmap = mpl.colormaps['Purples']
bluesmap = mpl.colormaps['Blues']
cmap = bluesmap
#cmap = mpl.colormaps['jet']

# Import Resource Estimation package.

print("Importing resource estimation package...")
base_dir = os.path.dirname(os.path.abspath(__file__))

WORKFLOW_IMPORT_PATH2 = os.path.join(base_dir, 'workflow')
WORKFLOW_IMPORT_PATH = r'/Users/so714f/Documents/code/workflow'
WORKFLOW_TEST_IMPORT_PATH =r'/Users/so714f/Documents/code/workflow/test'
sys.path.append(WORKFLOW_IMPORT_PATH)
sys.path.append(WORKFLOW_TEST_IMPORT_PATH)
print("Current sys.path:", sys.path)
print("Base directory:", base_dir)
print("Looking for workflow in:", WORKFLOW_IMPORT_PATH)

from qiskit.transpiler.preset_passmanagers.plugin import passmanager_stage_plugins,list_stage_plugins
print(f"init: {list_stage_plugins('init')}")
print(f"routing: {list_stage_plugins('routing')}")
print(f"translation: {list_stage_plugins('translation')}")
print(f"scheduling: {list_stage_plugins('scheduling')}")
print(f"layout: {list_stage_plugins('layout')}\n")
# routing_plugins = passmanager_stage_plugins('routing')
# for plugin_name, plugin_class_type in routing_plugins.items():


#     print(f"{plugin_name} :    {plugin_class_type}")
routing_plugins = passmanager_stage_plugins('routing')
if not routing_plugins:
    raise RuntimeError(f"No routing plugins found. Please check your Qiskit installation.")

from workflow.simulation.refactor.graph import Network, Node, DirectedEdge
from workflow.simulation.refactor.process import Process
from workflow.simulation.refactor.resources.classical_resources import ClassicalDevice
from workflow.simulation.refactor.broker import Broker
from workflow.ast_dag.AstDagConverter import AstDagConverter
from test_GE import generate_Gaussian_elimination_network_random_demo
from test_chemistry_network import generate_electronic_energy_network,generate_electronic_energy_network2, gen_broker
from workflow.results import visualize_graph, visualize_graph_from_nx, visualize_graph_from_pg, publish_resource_usage_history
from workflow.simulation.refactor.utilities import publish_gantt, publish_gantt2
from workflow.simulation.refactor.devices.quantum_devices import *
n1 = Node(id=None, process_model=None, network_type=Node.INPUT)
n2 = Node(id=None, process_model=None, network_type=Node.INPUT)
out1 = Node(id=None, process_model=None, network_type=Node.OUTPUT)
p1 = Node(id=None, process_model=None, inputs=[n1], outputs=[out1])
p2 = Node(id=None, process_model=None, inputs=[n1, n2], outputs=[out1])
p3 = Node(id=None, process_model=None, inputs=[n1, p2], outputs=[out1])
device = ClassicalDevice(device_name="supercomputer", processor_type="CPU", RAM=100*10**9, properties={"Cores": 20, "Clock Speed": 3 * 10**9})
broker = Broker(classical_devices=[device])
net = Network("Test", nodes=[p1, p2, p3], input_nodes=[n1, n2], output_nodes=[out1], broker=broker)
