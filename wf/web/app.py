VERSION = "1.1"
PYVIS_OUTPUT_ID = 'pyvis'

import os
import sys
import logging
import asyncio
import datetime as dt
from typing import List

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.colors import rgb2hex

from shiny import App, render, ui, reactive, Inputs, Outputs, Session
import faicons as fa
from pyvis.network import Network as PyvisNetwork


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


try:
    from z3 import Optimize, Int, IntVector, And, Or, Bool, Implies, If, sat
    Z3_AVAILABLE = True
    logger.debug("z3 imported successfully.")
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("z3-solver not available; some features will be disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Qiskit routing-plugin check
# ─────────────────────────────────────────────────────────────────────────────
from qiskit.transpiler.preset_passmanagers.plugin import passmanager_stage_plugins

routing_plugins = passmanager_stage_plugins("routing")
if not routing_plugins:
    raise RuntimeError("No routing plugins found. Please check your Qiskit installation.")
logger.debug("Found Qiskit routing plugins: %s", routing_plugins)



# Set up directories
DIR = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.join(DIR, "www")

os.makedirs(WWW, exist_ok=True)
logger.debug("Static assets directory: %s", WWW)


WORKFLOW_IMPORT_PATH = os.environ.get("WORKFLOW_IMPORT_PATH", "/Users/sophieblock/torch_wf")
if WORKFLOW_IMPORT_PATH not in sys.path:
    sys.path.insert(0, WORKFLOW_IMPORT_PATH)
    logger.debug("Added workflow path to sys.path: %s", WORKFLOW_IMPORT_PATH)
 
WORKFLOW_TEST_IMPORT_PATH  = os.environ.get("WORKFLOW_TEST_IMPORT_PATH", '/Users/sophieblock/torch_wf/test')
if WORKFLOW_TEST_IMPORT_PATH not in sys.path:
    sys.path.insert(0, WORKFLOW_TEST_IMPORT_PATH)
    logger.debug("Added workflow test path to sys.path: %s", WORKFLOW_TEST_IMPORT_PATH)
base_dir = os.path.dirname(os.path.abspath(__file__))


# Import Resource Estimation package...
print("Importing resource estimation package...")

print("Current sys.path:", sys.path)
print("Base directory:", base_dir)
print("Looking for workflow in:", WORKFLOW_IMPORT_PATH)
print(f"WWW: {WWW}")
# ─────────────────────────────────────────────────────────────────────────────
# Application-specific imports
# ─────────────────────────────────────────────────────────────────────────────
from qrew.simulation.refactor.graph import Network, Node, DirectedEdge
from qrew.simulation.refactor.process import Process
from qrew.simulation.refactor.resources.classical_resources import ClassicalDevice
from qrew.simulation.refactor.broker import Broker
from qrew.ast_dag.AstDagConverter import AstDagConverter
# from qrew.app.network_samples import (
#     generate_Gaussian_elimination_network_random_demo,
#     generate_electronic_energy_network,
#     generate_electronic_energy_network2,
# )


from test_GE import generate_Gaussian_elimination_network_random_demo
from test_chemistry_network import generate_electronic_energy_network,generate_electronic_energy_network2


from qrew.results import (
    visualize_graph_from_nx,
    publish_resource_usage_history
)
from qrew.simulation.refactor.devices.quantum_devices import *

n1 = Node(id=None, process_model=None, network_type=Node.INPUT)
n2 = Node(id=None, process_model=None, network_type=Node.INPUT)
out = Node(id=None, process_model=None, network_type=Node.OUTPUT)

p1 = Node(id=None, process_model=None, inputs=[n1], outputs=[out])
p2 = Node(id=None, process_model=None, inputs=[n2], outputs=[out])
p3 = Node(id=None, process_model=None, inputs=[p1, p2], outputs=[out])

device = ClassicalDevice(
    device_name="Supercomputer",
    processor_type="CPU",
    RAM=100 * 10**9,
    properties={"Cores": 20, "Clock Speed": 3e9},
)
broker = Broker(classical_devices=[device])
test_network = Network(
    name="Test",
    nodes=[p1, p2, p3],
    input_nodes=[n1, n2],
    output_nodes=[out],
    broker=broker,
)
logger.debug("Initialized test Network with %d nodes", len(test_network.nodes))


mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"]   = 11

# AST/DAG test
def get_astdag_network(path, filename=True):
    """
    Reads a Python script, builds an AST, and returns the networkx graph.
    """
    cvt = AstDagConverter(path, filename=filename)
    # cvt.visualize_ast()
    # gg = cvt.to_networkx()
    return cvt

def get_pyvis_from_networkx(nxg):
    """
    Returns a PyVis graph from the networkx graph.
    """
    from pyvis.network import Network as PyvisNetwork

    g = PyvisNetwork(
        height='600px',
        width='100%',
        directed=True,
        notebook=False,
        neighborhood_highlight=False,
        select_menu=False,
        filter_menu=False,
        bgcolor='#ffffff',
        font_color=False,
        layout=None,
        heading='',
        cdn_resources='local'
    )
    # g.toggle_hide_edges_on_drag(True)
    g.barnes_hut()
    g.from_nx(nxg)
    # g.save_graph("./tmp.html")
    # g.from_nx(nx.davis_southern_women_graph())
    # d3html = g.generate_html()
    # return d3html
    return g

def get_dag_ge(matrix_size, broker, simulate=True):
    # return generate_Gaussian_elimination_network(matrix_size=matrix_size, broker=broker)
    return generate_Gaussian_elimination_network_random_demo(matrix_size, matrix_size, simulate=simulate)

def get_dag_chem(molname, broker, simulate=True):
    if molname == 'H2':
        symbols = ['H', 'H']
        coords = [[0, 0, 0], [0.74, 0, 0]]
        charge = 0
    elif molname == 'H4':
        symbols = ['H', 'H', 'H', 'H']
        coords = [
            [0.7071, 0.0, 0.0],
            [0.0, 0.7071, 0.0],
            [-1.0071, 0.0, 0.0],
            [0.0, -1.0071, 0.0]
        ]
        charge = 0
    elif molname == 'CH':
        symbols = ['C', 'H']
        coords = [[0, 0, 0], [1.09, 0, 0]]
        charge = 0
    elif molname == 'CH4':
        symbols = ['C', 'H', 'H', 'H', 'H']
        coords = [
            [0, 0, 0],
            [1.09, 0, 0],
            [0, 1.09, 0],
            [-1.09, 0, 0],
            [0, -1.09, 0]
        ]
        charge = 0
    elif molname == 'O2':
        symbols = ['O', 'O']
        coords = [
            [0, 0, 0],
            [1.30, 0, 0],
            [0, 0, 2]  # as shown in the screenshot
        ]
        charge = 0
    elif molname == 'CO':
        symbols = ['C', 'O']
        coords = [[0, 0, 0], [1.16, 0, 0]]
        charge = 0
    elif molname == 'CO2':
        symbols = ['C', 'O', 'O']
        coords = [
            [0, 0, 0],
            [-1.2, 0, 0],
            [1.2, 0, 0]
        ]
        charge = 0
    return generate_electronic_energy_network(
        symbols,
        coords,
        charge,
        broker,
        simulate=simulate
    )

def get_dag_chem_vqe(molname,
                     basis,
                     charge,
                     qubitmapping,
                     ansatz,
                     ansatzparams,
                     broker,
                     simulate=True):
    if molname == 'H2':
        symbols = ['H', 'H']
        coords = [[0, 0, 0], [0.74, 0, 0]]
        charge = int(charge)
    elif molname == 'H4':
        symbols = ['H', 'H', 'H', 'H']
        coords = [
            [0.7071, 0.0, 0.0],
            [0.0, 0.7071, 0.0],
            [-1.0071, 0.0, 0.0],
            [0.0, -1.0071, 0.0]
        ]
        charge = int(charge)
    elif molname == 'CH':
        symbols = ['C', 'H']
        coords = [[0, 0, 0], [1.09, 0, 0]]
        charge = int(charge)
    elif molname == 'CH4':
        symbols = ['C', 'H', 'H', 'H', 'H']
        coords = [
            [0, 0, 0],
            [1.09, 0, 0],
            [0, 1.09, 0],
            [-1.09, 0, 0],
            [0, -1.09, 0]
        ]
        charge = int(charge)
    elif molname == 'O2':
        symbols = ['O', 'O']
        coords = [
            [0, 0, 0],
            [1.30, 0, 0],
            [0, 0, 2]
        ]
        charge = int(charge)
    elif molname == 'CO':
        symbols = ['C', 'O']
        coords = [[0, 0, 0], [1.16, 0, 0]]
        charge = int(charge)
    elif molname == 'CO2':
        symbols = ['C', 'O', 'O']
        coords = [
            [0, 0, 0],
            [-1.2, 0, 0],
            [1.2, 0, 0]
        ]
        charge = int(charge)

    return generate_electronic_energy_network2(
        symbols=symbols,
        coordinates=coords,
        basis=basis,
        charge=charge,
        geometry_model='RHF',
        qubit_mapping=qubitmapping,
        ansatz=ansatz,
        ansatz_params=ansatzparams,
        broker=broker,
        simulate=simulate
    )

def generate_broker(ram=8.,
                    ncores=20,
                    freq=3.,
                    ibm_brisbane=False,
                    ibm_brussels=False,
                    ibm_fez=False,
                    ibm_kyiv=False,
                    ibm_nazca=False,
                    ibm_sherbrooke=False):
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=int(ram * 10**9),
        properties={
            "Cores": ncores,
            "Clock Speed": int(freq * 10**9)
        }
    )
    qdevices = []
    if ibm_brisbane:
        qdevices.append(IBM_Brisbane)
    if ibm_kyiv:
        qdevices.append(IBM_Kyiv)

    if ibm_sherbrooke:
        qdevices.append(IBM_Sherbrooke)

    broker = Broker(
        classical_devices=[supercomputer],
        quantum_devices=qdevices,
    )
    return broker

# Global references possibly used in a GUI or notebook
selected_ast_network = None
selected_dag_network = None
selected_df = None
pyvis_needs_updating = False
last_saved_pyvis = None
uses_quantum = False

# Network test (commented out examples)
# g = PyvisNetwork()
# g.toggle_hide_edges_on_drag(True)
# g.barnes_hut()
# g.from_nx(net.to_networkx())
# g.save_graph("./tmp.html")
# g.from_nx(nx.davis_southern_women_graph())
# d3html = g.generate_html()

# --- Icons ---
# https://icons.getbootstrap.com/icons/question-circle-fill/
# question_circle_fill = """
# <!-- question_circle_fill = ui.HTML('
# <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16">
#   <path d="M..."/>
# </svg>
# ') -->
# """
# https://icons.getbootstrap.com/icons/question-circle-fill/
question_circle_fill = ui.HTML(
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>'
)
ICONS = {
    "user": fa.icon_svg("user", "regular"),
    "wallet": fa.icon_svg("wallet"),
    "currency-dollar": fa.icon_svg("dollar-sign"),
    "ellipsis": fa.icon_svg("ellipsis"),
    "question_circle_fill": question_circle_fill,
    "chart": fa.icon_svg("chart-simple", "solid"),
    "corr-mass-loss": fa.icon_svg("scale-unbalanced-flip", "solid"),
    "corr-curr-dens": fa.icon_svg("bolt", "solid"),
    "corr-confidence": fa.icon_svg("bullseye", "solid"),
}
# --- General help HTML (example placeholders) ---
GENERAL_HELP_HTML = f"""
<b>Version {{VERSION}}</b><br><br>
This tool facilitates estimating the amount of classical and quantum resources...
"""
RESET_HELP_HTML = """
<b>Resource Estimation</b> tab allows estimating the total number of resources...
"""
RESEST_HELP_HTML = 'The <b>Resource Estimation</b> tab allows estimating the total number of resources (RAM, qubits, gates, processor time, etc.) required to complete a particular calculation.'

SIMULATION_HELP_HTML = """
<b>Simulation</b> tab allows a user to set up and run a simulation of a workflow...
"""
OPTIMIZATION_HELP_HTML = """
<b>Optimization</b> tab allows a user to solve to find a best workflow...
"""

HELP_HTML = {"General Overview": GENERAL_HELP_HTML,
             "Resource Estimation": RESEST_HELP_HTML,
             "Simulation": SIMULATION_HELP_HTML,
             "Workflow optimization": OPTIMIZATION_HELP_HTML,
             
            }

warning_banner_message: str = """
This computational tool is still in preliminary beta, and should be considered experimental. 
The tool may continue to change and improve over time. While hopefully this tool improves calculation 
and modeling of resource estimation, please be cautious with results predicted by 
this tool, and treat all calculations as non-production.
"""
# --- Catching information ---

last_resource_estimation_run = None
last_simulation_run = None
last_optimization_run = None

# --- App UI definition ---

# import shinyswatch
# shinyswatch.theme.minty()

#app_ui = ui.page_fluid(
app_ui = ui.page_bootstrap(
# app_ui = ui.page_fixed(
    # Web page banner, title, and Boeing logo:
    ui.markdown("<br>"),
    ui.HTML('<div style="display: grid; align-items: center; grid-template-columns: 1fr 1fr;">'),
    ui.HTML('<div><h1><b>Quantum Resource Estimation Tool</b></h1></div>'),
    ui.HTML("</div>"),
    ui.HTML(f'<div style="border:1px solid black;"><p style="text-align: center; color: red; font-weight: bold;">{warning_banner_message}</p></div>'),
    ui.markdown("<br>"),

    # Tab definitions

    ui.navset_card_pill(

        # Welcome/Instructions tab:

        # ui.nav_panel("Welcome/Instructions",
            # ui.layout_sidebar(
                # ui.sidebar(
                    # ui.h4(
                        # ui.tooltip(
                            # ui.span("Topics ", question_circle_fill),
                            # "Information about the Boeing Workflow/Resource Estimation tool and how to use it.",
                            # id="welcome_tooltip",
                        # ),
                    # ),
                    # # ui.h5("Material"),
                    # # ui.input_select("static_environment_material", "Select Material", full_materials_list),
                    # ui.output_ui("welcome_sidebar_controls"),
                    # ## ui.h5("Environment"),
                    # #ui.input_select("static_environment_type", "Environment Type", static_environment_types_list),
                    # ## The controls for the appropriate environment gets populated here. These are generated in `server.static_environment_controls()` below.
                    # #ui.output_ui("static_environment_controls"),
                    # width=400,
                # ),
                # #ui.panel_main(
                    # ui.h4("Welcome"),
                    # ui.output_ui("welcome_output_controls"),
                # #),
            # ),
        # ),

        # Network Selection

        ui.nav_panel("Select Network",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4(
                        ui.tooltip(
                            ui.span("Select a Network ", question_circle_fill),
                            "Generate a custom graph network from a Python script, Jupyter notebook, flow chart or diagram, or PowerPoint slide, for a problem domain to analyze, study, or optimize.",
                            id="select_network_tooltip",
                        ),
                    ),
                    ui.output_ui("selectnetwork_sidebar_problem_controls"),
                    ui.output_ui("selectnetwork_sidebar_problem_param_controls1"),
                    ui.output_ui("selectnetwork_sidebar_problem_param_controls2"),
                    ui.input_checkbox('selectnetwork_actually_simulate', 'Simulate (debug)', False),
                    ui.h4('Visualization Options'),
                    ui.input_select('vizopts_style', "Styling:", ['None', 'Visualize time required', 'Visualize memory and type required']),
                    ui.input_action_button("selectnetwork_sidebar_update", label="Update Network"),
                    ui.h4('Resources'),
                    ui.input_switch('res_use_hpc', 'Boeing HPC'),
                    ui.input_numeric('res_use_hpc_cores', 'Cores', 20),
                    ui.input_numeric('res_use_hpc_ram', 'Memory (GB)', 8),
                    ui.input_slider('res_use_hpc_freq', 'Clock Speed (GHz)', value=2.1, min=1.0, max=5.0, ticks=True),
                    #ui.accordion(
                    #    ui.accordion_panel('IBM Brisbane',
                            ui.input_switch('res_use_ibmbrisbane', 'Include IBM Brisbane'),
                            ui.markdown(f'| Property | Value |\n|-------|-------|\n| Gate set: | {str(IBM_Brisbane.gate_set)} |\n| Connectivity: | {str(IBM_Brisbane.connectivity)} |\n| Qubits: | {str(len(IBM_Brisbane.available_qubits))} |'),
                    #    ),
                    #    open=False,
                    #),
         
                   
                    #ui.accordion(
                    #    ui.accordion_panel('IBM Kyiv',
                            ui.input_switch('res_use_ibmkyiv', 'Include IBM Kyiv'),
                            ui.markdown(f'| Property | Value |\n|-------|-------|\n| Gate set: | {str(IBM_Kyiv.gate_set)} |\n| Connectivity: | {str(IBM_Kyiv.connectivity)} |\n| Qubits: | {str(len(IBM_Kyiv.available_qubits))} |'),
                    #    ),
                    #    open=False,
                    #),
                   
                    #ui.accordion(
                    #    ui.accordion_panel('IBM Sherbrooke',
                            ui.input_switch('res_use_ibmsherbrooke', 'Include IBM Sherbrooke'),
                            ui.markdown(f'| Property | Value |\n|-------|-------|\n| Gate set: | {str(IBM_Sherbrooke.gate_set)} |\n| Connectivity: | {str(IBM_Sherbrooke.connectivity)} |\n| Qubits: | {str(len(IBM_Sherbrooke.available_qubits))} |'),
                    #    ),
                    #    open=False,
                    #),
                    width=400,
                ),
                #ui.h4("Custom Network"),
                ui.output_ui(PYVIS_OUTPUT_ID),
            ),
        ),

        # Resource Estimation tab:
                
        ui.nav_panel("Resource Estimation",
            ui.layout_sidebar(
                # Resource Estimation sidebar definition:
                ui.sidebar(
                    ui.h4(
                        ui.tooltip(
                            ui.span("Resource Estimation ", question_circle_fill),
                            "Determines the footprint size of resources (RAM, processor time, qubits, quantum gates, circuit depth, etc.) required to solve the specified problem.",
                            id="resest_header_tooltip",
                        ),
                    ),
                    #ui.output_ui("resest_sidebar_problem_controls"),
                    #ui.output_ui("resest_sidebar_problem_param_controls"),
                    ui.input_checkbox('resest_scale_abs', 'Use absolute scale', True),
                    ui.input_action_button('update_resest_results', 'Simulate and update'),
                    width=400,
                ),
                # Main Resource Estimation page contents:
                ui.h4("Classical Resource Utilization"),
                #ui.HTML(d3html),
                #ui.input_file("file1", label="Choose Python Script", accept=[".py"], multiple=False),
                #ui.output_ui("network_ast_graph"),
                #ui.output_ui(PYVIS_OUTPUT_ID),
                #ui.output_plot("testplot"),
                #ui.output_ui("resest_results_controls"),
                ui.output_plot('resest_results_plot'),
                ui.h4("Quantum Resource Utilization"),
                ui.output_plot('resest_results_plot2'),
            ),
        ),

        # Workflow Simulation tab:

        ui.nav_panel("Workflow Simulation",
            ui.layout_sidebar(
                # Simulation sidebar definition:
                ui.sidebar(
                    ui.h4(
                        ui.tooltip(
                            ui.span("Simulation ", question_circle_fill),
                            "Simulates the time-domain behavior of resource utilization for solving a computational problem, and shows results of resource usage over time.",
                            id="simulation_header_tooltip",
                        ),
                    ),
                    #ui.output_ui("simulation_sidebar_problem_controls"),
                    #ui.output_ui("simulation_sidebar_problem_param_controls"),
                    #ui.output_ui("simulation_sidebar_workflow_selection_controls"),
                    #ui.output_ui("simulation_sidebar_resources_controls"),
                    #ui.output_ui("simulation_sidebar_run_sim_controls"),
                    ui.input_action_button('update_sim_results', 'Simulate and update'),
                    width=400,
                ),
                # Main Simulation page elements:
                ui.h4("Workflow Simulation Results"),
                #ui.output_ui("simulation_results_controls"),
                ui.output_plot("simulation_results_plot"),
                ui.h4("Most Memory-intensive Tasks"),
                ui.output_plot("simulation_results_plot2"),
                ui.h4("Most Time-intensive Tasks"),
                ui.output_plot("simulation_results_plot3"),
            ),
        ),

        # Workflow Optimization tab:

        ui.nav_panel("Workflow Optimization",
            ui.layout_sidebar(
                # Optimization sidebar definition:
                ui.sidebar(
                    ui.h4(
                        ui.tooltip(
                            ui.span("Optimization ", question_circle_fill),
                            "Solves to find an optimal workflow, given a computational problem, set of workflows for solving the problem, and resources available.",
                            id="optimization_header_tooltip",
                        ),
                    ),
                    #ui.output_ui("resest_sidebar_problem_controls"),
                    #ui.output_ui("resest_sidebar_problem_param_controls"),
                    width=400,
                ),
                # Main Optimization page elements:
                ui.h4("Workflow Optimization Results"),
            ),
        ),
    ),
)

#  --- Server Definition ---
last_units: str ="°F"
last_calculated_corrosion: float = None

LINEAR_WORKFLOWS =[
    "Classical: Gaussian Elimination",
    "Classical: Congugate-Gradient",
    "Hybrid: VQLS",
    "Quantum: HHL"
]

def server(input: Inputs, output: Outputs, session:Session):
    beta_warning_banner = ui.modal(
                          warning_banner_message,
                          title="This product is still in beta",
                          easy_close=True
    )

    # Immediately cause the beta warning banner to pop up on the server startup.
    ui.modal_show(beta_warning_banner)
    
    @render.ui
    def welcome_sidebar_controls():
        return ui.TagList(
            ui.input_select("welcome_help_topic", "Help topic", ["General Overview", "Resource Estimation", "Simulation", "Workflow optimization", "Credits/Contact"])
        )

    @render.ui
    def welcome_output_controls():
        return ui.TagList(
            ui.HTML(HELP_HTML[input.welcome_help_topic()]),
        )

    # Network selection controls

    @render.ui
    def selectnetwork_sidebar_problem_controls():
        return ui.TagList(
            ui.input_select("network_problem", "Problem", ["Ground state energy (classical)", "Ground state energy (VQE)", "Ax = b (Solve a linear system)", "Create Custom Network from File..."]),
        )

    @render.ui
    def selectnetwork_sidebar_problem_param_controls1():
        global selected_ast_network, selected_dag_network
            
        if (input.network_problem() == 'Ground state energy (classical)'):
            return ui.TagList(
                ui.input_select('gseclass_mol', 'Molecule', ['H2', 'H4', 'CH', 'CH4', 'O2', 'CO', 'CO2', 'Load a custom molecule file...']),
                ui.input_select('gseclass_basis', 'Basis', ['sto-3g']),
                ui.input_numeric('gseclass_charge', 'Charge', 0),
            )
        elif (input.network_problem() == 'Ground state energy (VQE)'):
            return ui.TagList(
                ui.input_select('gsevqe_mol', 'Molecule', ['H2', 'H4', 'CH', 'CH4', 'O2', 'CO', 'CO2', 'Load a custom molecule file...']),
                ui.input_select('gsevqe_basis', 'Basis', ['sto-3g']),
                ui.input_numeric('gsevqe_charge', 'Charge', 0),
                ui.input_select('gsevqe_qubitmapping', 'Qubit Mapping', ['JW', 'BK', 'SCBK']),
                ui.input_select('gsevqe_ansatz', 'Ansatz', ['UCCSD']),
                ui.input_select('gsevqe_ansatzparams', 'Ansatz Params', ['random', 'ones']),
            )
        elif (input.network_problem() == "Ax = b (Solve a linear system)"):
            return ui.TagList(
                ui.input_slider(id='linear_system_matrix_size',
                                label='Matrix size',
                                min=2,
                                max=256,
                                value=2,
                                ticks=True,
                )
            )
        elif (input.network_problem() == "Create Custom Network from File..."):
            return ui.TagList(
                ui.input_file("file1", label="Choose Source File...", accept=[".py"], multiple=False),
                ui.input_action_button("load_ast_from_file1", "Load...", width='100%',),
                #ui.input_action_button("
            )
        else:
            return ui.TagList(

            )

    @render.ui
    def selectnetwork_sidebar_problem_param_controls2():
        reactive.invalidate_later(1)
        if (input.network_problem() == "Create Custom Network from File..."):
            global selected_dag_network
            if selected_dag_network is not None:
                return ui.TagList(
                    #ui.input_select("network_vis_coloring", 
                )
            else:
                return ui.TagList(
                
                )
        else:
            return ui.TagList(
                
            )

    # Resource estimation controls

    @render.ui
    def resest_sidebar_problem_controls():
        return ui.TagList(
            ui.input_select("resest_problem", "Problem", ["Ground state energy (classical)", "Ground state energy (VQE)", "Ax = b (Solve a linear system)"]),
        )

    @render.ui
    def resest_sidebar_problem_param_controls():
            return ui.TagList(

            )

    @reactive.event(input.linear_system_matrix_size)
    def _():
        global pyvis_needs_updating
        pyvis_needs_updating = True
        
        # global selected_dag_network
        
        # matrix_size = input.linear_system_matrix_size()
        # #selected_dag_network = get_dag_ge(matrix_size=matrix_size, broker=generate_broker())

    @render.ui
    def resest_results_controls():
        return ui.TagList(

        )

    @output
    @render.plot
    @reactive.event(input.update_resest_results)
    def resest_results_plot():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df, selected_dag_network, selected_broker
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            use_abs_scale = input.resest_scale_abs()
            fig, ax = publish_resource_usage_history(selected_df, key='Memory Cost [B]', show=False, broker=selected_broker, use_absolute_scale=use_abs_scale)
            return fig

    @output
    @render.plot
    @reactive.event(input.update_resest_results)
    def resest_results_plot2():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df, selected_dag_network, selected_broker, uses_quantum
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            if not uses_quantum:
                fig, ax = mpl.subplots(1)
                ax.axis("off")
                ax.text(0.1, 0.9, "No quantum resources required in this scenario.")
                return fig
            else:
                use_abs_scale = input.resest_scale_abs()
                fake = 2. if input.gsevqe_mol() == 'H2' else 4.
                fig, ax = publish_resource_usage_history(selected_df, key='Qubits Used', show=False, broker=selected_broker, use_absolute_scale=use_abs_scale, fake=fake)
                return fig

    # Network visualization

    #@reactive.calc
    @reactive.event(input.load_ast_from_file1)
    def parsed_file():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return None
        else:
            script = file[0]["datapath"]
            with open(script, 'r', encoding="utf-8") as file:
                contents = file.read()
                return contents
        
    @output(id=PYVIS_OUTPUT_ID)
    @render.ui
    @reactive.event(input.selectnetwork_sidebar_update)
    def _():
        print("🍺  network update handler fired!")
        
        global selected_ast_network, selected_dag_network, selected_df, selected_broker, uses_quantum
        
        # Create the new broker.
        broker = generate_broker(ram=input.res_use_hpc_ram(), 
                                 ncores=int(input.res_use_hpc_cores()), 
                                 freq=input.res_use_hpc_freq(), 
                                 ibm_brisbane=input.res_use_ibmbrisbane(),
                                 ibm_brussels=input.res_use_ibmbrussels(),
                               
                                 ibm_kyiv=input.res_use_ibmkyiv(),
                                 ibm_sherbrooke=input.res_use_ibmsherbrooke(),
                                 )
        
        simulate = input.selectnetwork_actually_simulate()
        print(f'debug: actually simulate? {simulate}')
        
        uses_quantum = False
        selected_broker = broker
        
        if input.network_problem() == "Ground state energy (classical)":

            #molname = "H4"
            molname = input.gseclass_mol()
            basis = input.gseclass_basis()
            charge = float(input.gseclass_charge())
            gg, df = get_dag_chem(molname, broker, simulate=simulate)
            selected_dag_network = gg
            selected_df = df
            nxg, mpg = gg.to_networkx()

            styling = input.vizopts_style()
            sizefn = None
            colorfn = None
            # print(f"debug: styling={styling}")
            
            if styling == 'Visualize time required':
                def sizefn(x):
                    if 'RowReduction' in x:
                        return 30
                    elif 'RowReconstruction' in x:
                        return 23
                    elif 'RowDeconstruction' in x:
                        return 22
                    elif 'SwapRow' in x:
                        return 8
                    else:
                        return 5
                def colorfn(x):
                    if 'RowReduction' in x:
                        return rgb2hex(cmap(30./30.))
                    elif 'RowReconstruct' in x:
                        return rgb2hex(cmap(23./30.))
                    elif 'RowDeconstruct' in x:
                        return rgb2hex(cmap(22./30.))
                    elif 'SwapRow' in x:
                        return rgb2hex(cmap(8./30.))
                    else:
                        return rgb2hex(cmap(5./30.))
            
            elif styling == 'Visualize memory and type required':
                def sizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x or 'nsatz' in x:
                        return 40
                    else:
                        return 25
                def colorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'
                    #    cm = bluesmap
                    #return cm(1.)
                def dsizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return 30
                    else:
                        return 15
                def dcolorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'

            
            pg = visualize_graph_from_nx(nxg,
                                         #input_nodes,
                                         #output_nodes,
                                         default_data_node_shape='triangle',
                                         labels={name: name for name in nxg.nodes},
                                         proc_node_size_fn=sizefn,
                                         proc_node_color_fn=colorfn,
                                         )

            html = pg.generate_html(local=False)
            # html = pg.generate_html(cdn_resources="inlined")
            f = os.path.join(WWW, PYVIS_OUTPUT_ID + ".html")
            #f = PYVIS_OUTPUT_ID + ".html"
            with open(f, "w") as f:
                f.write(pg.html)

            frame = ui.tags.iframe(
                src=PYVIS_OUTPUT_ID + ".html",
                style="height:800px;width:100%;",
                scrolling="no",
                seamless="seamless",
                frameBorder="1",
            )
            
            last_saved_pyvis = frame
            pyvis_needs_updating = False
            return frame

        elif input.network_problem() == "Ground state energy (VQE)":

            uses_quantum = True
            #molname = "H4"
            molname = input.gsevqe_mol()
            basis = input.gsevqe_basis()
            charge = float(input.gsevqe_charge())
            qubitmapping = input.gsevqe_qubitmapping()
            ansatz = input.gsevqe_ansatz()
            ansatzparams = input.gsevqe_ansatzparams()
            gg, df = get_dag_chem_vqe(molname, 
                                      basis=basis,
                                      charge=charge,
                                      qubitmapping=qubitmapping,
                                      ansatz=ansatz,
                                      ansatzparams=ansatzparams,
                                      broker=broker,
                                      simulate=simulate,
                                      )
            selected_dag_network = gg
            selected_df = df
            nxg, mpg = gg.to_networkx()

            styling = input.vizopts_style()
            sizefn = None
            colorfn = None
            print(f"debug: styling={styling}")
            
            if styling == 'Visualize time required':
                def sizefn(x):
                    if 'RowReduction' in x:
                        return 30
                    elif 'RowReconstruction' in x:
                        return 23
                    elif 'RowDeconstruction' in x:
                        return 22
                    elif 'SwapRow' in x:
                        return 8
                    else:
                        return 5
                def colorfn(x):
                    if 'RowReduction' in x:
                        return rgb2hex(cmap(30./30.))
                    elif 'RowReconstruct' in x:
                        return rgb2hex(cmap(23./30.))
                    elif 'RowDeconstruct' in x:
                        return rgb2hex(cmap(22./30.))
                    elif 'SwapRow' in x:
                        return rgb2hex(cmap(8./30.))
                    else:
                        return rgb2hex(cmap(5./30.))
            
            elif styling == 'Visualize memory and type required':
                def sizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x or 'nsatz' in x:
                        return 40
                    else:
                        return 25
                def colorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'
                    #    cm = bluesmap
                    #return cm(1.)
                def dsizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return 30
                    else:
                        return 15
                def dcolorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'
            
            pg = visualize_graph_from_nx(nxg,
                                         #input_nodes,
                                         #output_nodes,
                                         default_data_node_shape='triangle',
                                         labels={name: name for name in nxg.nodes},
                                         proc_node_size_fn=sizefn,
                                         proc_node_color_fn=colorfn,
                                         )

            html = pg.generate_html(local=False)
            # html = pg.generate_html(cdn_resources="inlined")
            f = os.path.join(WWW, PYVIS_OUTPUT_ID + ".html")
            #f = PYVIS_OUTPUT_ID + ".html"
            with open(f, "w") as f:
                f.write(pg.html)

            frame = ui.tags.iframe(
                src=PYVIS_OUTPUT_ID + ".html",
                style="height:800px;width:100%;",
                scrolling="no",
                seamless="seamless",
                frameBorder="1",
            )
            
            last_saved_pyvis = frame
            pyvis_needs_updating = False
            return frame
            
        elif input.network_problem() == "Ax = b (Solve a linear system)":
        
            matrix_size = input.linear_system_matrix_size()
        
            gg, df = get_dag_ge(matrix_size, generate_broker(), simulate=simulate)
            selected_dag_network = gg
            selected_df = df
            nxg, mpg = gg.to_networkx()
            #pg = PyvisNetwork()
            #pg.from_nx(gg.to_networkx())

            # data_nodes = set()
            # proc_nodes = set()
            # input_nodes = set()
            # output_nodes = set()
            # edges = []
            # for node in gg.nodes:
                # proc_nodes.add(mpg[node])
                # for output_node in node.output_nodes:
                    # n2 = 'Data from ' + mpg[output_node]
                    # data_nodes.add(n2)
                    # edges.append((mpg[node],n2))
            # for node in gg.input_nodes:
                # data_nodes.add(mpg[node])
                # input_nodes.add(mpg[node])
            # for node in gg.output_nodes:
                # data_nodes.add(mpg[node])
                # output_nodes.add(mpg[node])
                
            #data_nodes = list(data_nodes)
            #proc_nodes = list(proc_nodes)
            #edges = nxg.edges
            #def f(x):
            #    if 'RowRecon' in x:
            #        return 30
            #    else:
            #        return 5
            
            styling = input.vizopts_style()
            sizefn = None
            colorfn = None
            print(f"debug: styling={styling}")
            
            if styling == 'Visualize time required':
                def sizefn(x):
                    if 'RowReduction' in x:
                        return 30
                    elif 'RowReconstruction' in x:
                        return 23
                    elif 'RowDeconstruction' in x:
                        return 22
                    elif 'SwapRow' in x:
                        return 8
                    else:
                        return 5
                def colorfn(x):
                    if 'RowReduction' in x:
                        return rgb2hex(cmap(30./30.))
                    elif 'RowReconstruct' in x:
                        return rgb2hex(cmap(23./30.))
                    elif 'RowDeconstruct' in x:
                        return rgb2hex(cmap(22./30.))
                    elif 'SwapRow' in x:
                        return rgb2hex(cmap(8./30.))
                    else:
                        return rgb2hex(cmap(5./30.))
            
            elif styling == 'Visualize memory and type required':
                def sizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x or 'nsatz' in x:
                        return 40
                    else:
                        return 25
                def colorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'
                    #    cm = bluesmap
                    #return cm(1.)
                def dsizefn(x):
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return 30
                    else:
                        return 15
                def dcolorfn(x):
                    #cm = cmap
                    if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                        return '#ef44ff'
                    else:
                        return '#4485ff'
            
            pg = visualize_graph_from_nx(nxg,
                                         #input_nodes,
                                         #output_nodes,
                                         default_data_node_shape='triangle',
                                         labels={name: name for name in nxg.nodes},
                                         proc_node_size_fn=sizefn,
                                         proc_node_color_fn=colorfn,
                                         )

            html = pg.generate_html(local=False)
            # html = pg.generate_html(cdn_resources="inlined")
            f = os.path.join(WWW, PYVIS_OUTPUT_ID + ".html")
            #f = PYVIS_OUTPUT_ID + ".html"
            with open(f, "w") as f:
                f.write(pg.html)

            frame = ui.tags.iframe(
                src=PYVIS_OUTPUT_ID + ".html",
                style="height:800px;width:100%;",
                scrolling="no",
                seamless="seamless",
                frameBorder="1",
            )
            
            last_saved_pyvis = frame
            pyvis_needs_updating = False
            return frame
            
        elif input.network_problem() == "Create Custom Network from File...":
        
            script = parsed_file()
            print("Creating network AST graph...")
            print(f"Loaded data: {script}")
            
            # If a file has been uploaded via the `parsed_file()` input,
            # we collect the source code and generate an interactive
            # PyVis graphic that is injected.
            
            # NOTE: The injection is a little funky here, because PyVis
            # has to write the HTML as a necessary step for creating the
            # templates. This might could be handled better. We instead write
            # the HTML manually, and then re-parse it using an IFrame to
            # create the interactivity.
            
            if script is not None:
                gg = get_astdag_network(script, filename=False)
                data_nodes, proc_nodes, edges, input_nodes, output_nodes, labels = gg.create_dag_data()
                #pg = get_pyvis_from_networkx(gg)
                #pg = gg.to_networkx()
                #pg = gg.create_dag()
                #data_nodes = list(data_nodes)
                # pg = visualize_graph(data_nodes,
                                     # proc_nodes,
                                     # edges,
                                     # input_nodes,
                                     # output_nodes,
                                     # labels=labels,
                                     # )
                #pg = gg.to_networkx()
                # pg = get_pyvis_from_networkx(gg.to_networkx())
                
                styling = input.vizopts_style()
                sizefn = None
                colorfn = None
                dsizefn = None
                dcolorfn = None
                print(f"debug: styling={styling}")
                
                if styling == 'Visualize time required':
                    def sizefn(x):
                        if 'RowReduction' in x:
                            return 30
                        elif 'RowReconstruction' in x:
                            return 23
                        elif 'RowDeconstruction' in x:
                            return 22
                        elif 'SwapRow' in x:
                            return 8
                        else:
                            return 5
                    def colorfn(x):
                        if 'RowReduction' in x:
                            return rgb2hex(cmap(30./30.))
                        elif 'RowReconstruct' in x:
                            return rgb2hex(cmap(23./30.))
                        elif 'RowDeconstruct' in x:
                            return rgb2hex(cmap(22./30.))
                        elif 'SwapRow' in x:
                            return rgb2hex(cmap(8./30.))
                        else:
                            return rgb2hex(cmap(5./30.))
                
                elif styling == 'Visualize memory and type required':
                    def sizefn(x):
                        if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x or 'nsatz' in x:
                            return 40
                        else:
                            return 25
                    def colorfn(x):
                        #cm = cmap
                        if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x  or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                            return '#ef44ff'
                        else:
                            return '#4485ff'
                        #    cm = bluesmap
                        #return cm(1.)
                    def dsizefn(x):
                        if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                            return 30
                        else:
                            return 15
                    def dcolorfn(x):
                        #cm = cmap
                        if 'Quantum' in x or 'Qubit' in x or 'gate' in x or 'Gate' in x or 'Ry' in x or 'Cx' in x or 'perator' in x or 'ircuit' in x  or 'nsatz' in x:
                            return '#ef44ff'
                        else:
                            return '#4485ff'

                pg = visualize_graph(data_nodes,
                                     proc_nodes,
                                     edges,
                                     input_nodes,
                                     output_nodes,
                                     labels=labels,
                                     data_node_size_fn=dsizefn,
                                     data_node_color_fn=dcolorfn,
                                     proc_node_size_fn=sizefn,
                                     proc_node_color_fn=colorfn,
                                     )

                # nxg = gg.to_networkx()
                                
                # pg = visualize_graph_from_pg(pg,
                                             # #input_nodes,
                                             # #output_nodes,
                                             # #default_data_node_shape='triangle',
                                             # labels={name: name for name in pg.get_nodes()},
                                             # proc_node_size_fn=sizefn,
                                             # proc_node_color_fn=colorfn,
                                             # #pg=pg,
                                             # )
                
                html = pg.generate_html(local=False)
                # html = pg.generate_html(cdn_resources="inlined")
                f = os.path.join(WWW, PYVIS_OUTPUT_ID + ".html")
                #f = PYVIS_OUTPUT_ID + ".html"
                with open(f, "w") as f:
                    f.write(pg.html)

                selected_ast_network = gg
                selected_dag_network = pg

                frame = ui.tags.iframe(
                    src=PYVIS_OUTPUT_ID + ".html",
                    style="height:600px;width:100%;",
                    scrolling="no",
                    seamless="seamless",
                    frameBorder="1",
                )
                
                last_saved_pyvis = frame
                pyvis_needs_updating = False
                return frame

            else:
            
                # No file has been uploaded yet, so just return an empty tag list.
                
                empty_frame = ui.TagList(
                
                )
                
                last_saved_pyvis = empty_frame
                pyvis_needs_updating = False
                return empty_frame

    # Simulation controls

    @render.ui
    def simulation_sidebar_problem_controls():
        """Returns the Shiny UI controls for the sidebar problem selection.
        
        Note that this control element is separated from the problem parameters so
        that its value can be evaluated to select and show different problem
        parameter controls. The parameter controls are generated separately in
        `simulation_sidebar_problem_param_controls()`.

        """
        return ui.TagList(
            ui.input_select("simulation_problem", "Problem", ["Chemistry (Ground-state energy)", "Ax = b (Solve a linear system)"]),
        )

    @render.ui
    def simulation_sidebar_problem_param_controls():
        """Returns the Shiny UI controls for the sidebar problem parameters selection.
        
        Note that the problem parameter controls are separated from the general
        problem selection control, so that o

        """
        problem_selected = input.simulation_problem()
        #if "simulation_linear_structure" in input:
        #    linear_structure = input.simulation_linear_structure()
        #else:
        #    linear_structure = "Diagonal"

        if (problem_selected == "Chemistry (Ground-state energy)"):
            return ui.TagList(
                ui.markdown("**Description:** Find a lowest eigenvalue of a Hamiltonian matrix. The lowest eigenvalue corresponds to finding the ground-state energy of the system represented by the specified Hamiltonian."),

            )
        elif (problem_selected == "Ax = b (Solve a linear system)"):
            structure_value = input.simulation_linear_structure() if "simulation_linear_structure" in input else 0
            taglist = ui.TagList(
                ui.markdown("**Description:** Solve *A x = b* for the solution vector *x*, for a specified matrix *A* and load vector *b*."),
                ui.HTML("<br>"),
                ui.input_slider("simulation_linear_matrix_size", "Size of A (in rows)", 1, 1000000, 512, ticks=True),
                #ui.input_switch("simulation_linear_issparse", "Sparse?", False),
                ui.input_select("simulation_linear_structure", "Structure", ["Diagonal", "Tridiagonal", "Pentadiagonal", "Banded diagonal", "Sparse"], selected=structure_value),
                ui.input_select("simulation_linear_workflow", "Selected workflow", LINEAR_WORKFLOWS),
                ui.input_slider("simulation_linear_sparsity", "Sparsity (%)", 1, 100, 20, ticks=True),
                ui.markdown("*(100% is same as a non-sparse treatment.)*"),
            )
            if "simulation_linear_structure" in input and input.simulation_linear_structure == "Diagonal":
                taglist.append(ui.markdown("is diagonal"))
        else:
            raise ValueError(f"Unrecognized problem class: {input.simulation_problem()}")

    @render.ui
    def simulation_sidebar_workflow_selection_controls():
        """Returns the Shiny UI controls for the sidebar to select the workflow path to simulate."""
        problem_selected = input.simulation_problem()
        if problem_selected == "Ax = b (Solve a linear system)":
            linear_workflow = input.simulation_linear_workflow()
            if linear_workflow == LINEAR_WORKFLOWS[0]:
                # Classical: Gaussian Elimination controls
                return ui.TagList(
                    ui.input_slider("simulation_linear_ge_numprocs", "Number of processors", 1, 128, 4),
                )
            elif linear_workflow == LINEAR_WORKFLOWS[1]:
                # Classical: Congugate-Gradient (CG) controls
                return None
            elif linear_workflow == LINEAR_WORKFLOWS[2]:
                # Hybrid: VQLS
                return None
            elif linear_workflow == LINEAR_WORKFLOWS[3]:
                # Quantum: HHL
                return None
            else:
                raise ValueError(f"Unrecognized linear workflow: {linear_workflow}")

    @render.ui
    def simulation_sidebar_resources_controls():
        """Returns the Shiny UI controls for the sidebar to select resources available for simulation."""
        pass

    @render.ui
    def simulation_sidebar_run_sim_controls():
        """Returns the Shiny UI controls for launching a simulation."""
        return ui.TagList(
            #ui.input_action_button("simulation_sidebar_start_simulation_button", "Launch Simulation", width="100%")
        )

    @render.ui
    def simulation_results_controls():
        """Returns the Shiny UI controls representing the results/output of the simulation."""

        # If no simulation has been run yet, then just return a plot showing instructions.

        if last_simulation_run is None:
            return ui.TagList(
                ui.output_plot("simulation_results_plot"),
            )

    @output
    @render.plot
    @reactive.event(input.update_sim_results)
    def simulation_results_plot():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            fig, ax = publish_gantt(selected_df, show=False, cmap='jet')
            return fig

    @output
    @render.plot
    @reactive.event(input.update_sim_results)
    def simulation_results_plot2():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            fig, ax = publish_gantt2(selected_df, show=False, cmap='jet', key='Memory Used [B]')
            return fig

    @output
    @render.plot
    @reactive.event(input.update_sim_results)
    def simulation_results_plot2():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            fig, ax = publish_gantt2(selected_df, show=False, cmap='jet', key='Memory Cost [B]', ascending=True)
            return fig

    @output
    @render.plot
    @reactive.event(input.update_sim_results)
    def simulation_results_plot3():
        """Generates and returns a blank plot with text instructions. This can be used as a placeholder figure if data has not yet been generated."""
        global selected_df
        
        if selected_df is None:
            fig, ax = mpl.subplots(1)
            ax.axis("off")
            ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
            return fig
        else:
            fig, ax = publish_gantt2(selected_df, show=False, cmap='jet', key='Duration [s]', ascending=True)
            return fig
            
    @output
    @render.plot
    def testplot():
        fig, ax = mpl.subplots(1)
        #ax.axis("off")
        #ax.text(0.1, 0.9, "Set the simulation inputs on the left, and click <b>Run Simulation</b> to see the results.")
        net.visualize(show=False, ax=ax, show_edge_labels=True)
        return fig

    @render.ui
    def static_environment_material_controls():
        return ui.TagList(
            ui.row(
                ui.column(
                    5,
                    ui.input_select("static_environment_material", "Material", ["Aluminum", "Titanium", "CFRP", "Steel", "Bronze"]),
                ),
                ui.column(
                    2,
                    ui.input_select("static_environment_material_temper", "Temper", ["-T1", "-T2", "-T3", "-T4", "-T5", "-T6"]),
                ),
                ui.column(
                    3,
                    ui.input_select("static_environment_material_treatment", "Treatment", ["Bare", "1", "2", "3"])
                ),
            ),
        )

    # Add the controls generator for static environments.
    # This depends on the environment type, so the selected
    # environment type is used to generate the resulting controls.

    @render.ui
    @reactive.event(input.static_environment_type)
    def static_environment_controls():
        """Returns a set of UI widget controls for the currently selected environment type."""
        etype = input.static_environment_type()
        if etype == "Ground outdoor exposed":
            # Generate the ground/outdoor exposed controls.
            return ui.TagList(
                # Add air temperature widget.
                ui.row(
                    ui.column(
                        12,
                        ui.input_select("static_environment_geographic_region", "Geographic Region", geographic_regions_list)
                    ),
                    ui.column(
                        10,
                        ui.input_slider("static_environment_air_temperature", "Air Temperature", 50, 122, 75, ticks=True),
                    ),
                    ui.column(
                        2,
                        ui.input_select("static_environment_air_temperature_units", "Units", ["°F", "°C"]),
                    ),
                ),
                # Add relative humidity widget.
                ui.row(
                    ui.column(
                        12,
                        ui.input_slider("static_environment_relative_humidity", "Relative Humidity", 0, 100, 80, ticks=True),
                    ),
                ),
            )
        elif etype == "Chamber":
            # Generate the Chamber environment controls.
            return ui.TagList(
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider("static_environment_air_temperature", "Air Temperature", 50, 122, 75),
                    ),
                    ui.column(
                        6,
                        ui.input_select("static_environment_air_temperature_units", "Units", ["°F", "°C"]),
                    ),
                ),
                # Add relative humidity widget.
                ui.row(
                    ui.column(
                        12,
                        ui.input_slider("static_environment_relative_humidity", "Relative Humidity", 0, 100, 80),
                    ),
                ),
            )
        elif etype == "In-flight":
            # Generate the in-flight condition environment controls.
            return ui.TagList(
                ui.row(
                    ui.column(
                        6,
                        ui.input_slider("static_environment_air_temperature", "Air Temperature", 50, 122, 75),
                    ),
                    ui.column(
                        6,
                        ui.input_select("static_environment_air_temperature_units", "Units", ["°F", "°C"]),
                    ),
                ),
                # Add relative humidity widget.
                ui.row(
                    ui.column(
                        12,
                        ui.input_slider("static_environment_relative_humidity", "Relative Humidity", 0, 100, 80),
                    ),
                ),
            )
        
    @render.ui
    def static_environment_predicted_mass_loss():
        return 
        
    # Create a reactive effect for updating the static environment air
    # temperature units, if the units selected changes.
    
    @reactive.effect
    def static_environment_update_air_temperature_units():
        """Reactive effect to update the air temperature units when the units dropdown selection changes."""
        global last_units
        units = input.static_environment_air_temperature_units()
        if units != last_units:
            if units == "°F":
                x_in_C = input.static_environment_air_temperature()
                x_in_F = (x_in_C * 9. / 5.) + 32.
                ui.update_slider("static_environment_air_temperature", min=50, max=122, value=x_in_F)
            elif units == "°C":
                x_in_F = input.static_environment_air_temperature()
                x_in_C = (x_in_F - 32.) * (5. / 9.)
                ui.update_slider("static_environment_air_temperature", min=10, max=50, value=x_in_C)
            last_units = units
    
    # Create a reactive plot element that will generate a plot image
    # representing the currently selected environment's predicted
    # corrosion rate.
    
    @render.ui
    def static_environment_results_controls():
        global last_calculated_corrosion

        etype = input.static_environment_type()
        
        etypemap = {"Ground outdoor exposed": "ground",
                    "Chamber": "chamber",
                    "In-flight": "onboard",
        }
        
        air_temperature = input.static_environment_air_temperature()
        units = input.static_environment_air_temperature_units()
        rh = input.static_environment_relative_humidity()
        
        if units == "°F":
            air_temperature = (air_temperature - 32.) * (5. / 9.)  # convert to degC
        
        environment = {"type": etypemap[etype],
                       "atemp": air_temperature,
                       "rh": rh}
        
        #estimate = model.estimate(environment)
        
        #estimated_air_temperature_in_C = estimate["atemp"]
        #estimated_relative_humidity = estimate["rh"]
        #estimated_mass_loss = estimate["imlr"]
        
        estimated_mass_loss = 0.333
        
        last_calculated_corrosion = estimated_mass_loss
        
        if units == "°F":
            estimated_air_temperature = (estimated_air_temperature_in_C * 9. / 5.) + 32.
        else:
            estimated_air_temperature = estimated_air_temperature_in_C
        
        min_distance = estimate["min_distance"]
        abs_error = estimate["abs_error"]
        rel_error = estimate["rel_error"]
        # confidence = ...
        
        # data = {f"Estimated air temperature ({units})": estimated_air_temperature,
        #        f"Estimated relative humidity (%)": estimated_relative_humidity,
        #        f"Estimated mass loss ()": estimated_mass_loss,
        #        }
                
        return ui.TagList(
            ui.panel_well(
                ui.layout_columns(
                    ui.value_box(
                        #ui.tooltip(
                        #    ui.span("Predicted Mass Loss ", question_circle_fill),
                        #    "Model-predicted uniform mass loss for material in static environment, in $g/(m^2-year)$."),
                        "Predicted Mass Loss",
                        round(last_calculated_corrosion, 5), 
                        showcase=ICONS["corr-mass-loss"],
                    ),
                    ui.value_box(
                        "Predicted Current Density", round(last_calculated_corrosion * 3, 5), showcase=ICONS["corr-curr-dens"],
                    ),
                    ui.value_box(
                        "Model Confidence", 
                        "Very High", 
                        showcase=ICONS["corr-confidence"],
                    ),
                ),
                ui.output_plot("static_environment_corrosion_prediction"),
            ),
        )
    
    @render.plot
    def static_environment_corrosion_prediction():
        """Returns a plot representing the current selected environment predicted corrosion."""
        
        estimated_mass_loss = last_calculated_corrosion
                
        labels = ["Estimated mass loss"]
        values = [estimated_mass_loss]        
                
        fig, ax = mpl.subplots()
        
        bars = ax.bar(labels, values, color=(30 / 255, 186 / 255, 252 / 255), edgecolor='k')
        ax.set_ylim((1.e-4, 1.e1))
        ax.set_yscale("log")
        ax.set_ylabel("Mass loss, g/(m$^2$-year)")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        ax.bar_label(bars, fmt="%.4f")
        
        return bars
    
    # @render.image
    # def boeing_logo():
    #    dir = Path(__file__).resolve().parent
    #    #img: ImgData = {"src": str(dir / "assets" / "img" / "boeing_logo.png"), "width": "100px"}
    #    img = {"src": "app/boeing_logo.png", "height": "50px"}
    #    return img
        
    # @reactive.Effect
    # @output
    # @render.plot
    # def pds_scan_comparison_plot():
       
    # TODO FIXME: deprecated @session.download; replace with @render.download
    @render.download(
        filename="Corrosion Rate Comparison.csv"
    )
    def download_csv():
        m = ui.modal(
            "This is a somewhat important message.",
            title="Somewhat important message",
            easy_close=True
        )
        ui.modal_show(m)
    
    # async def download_csv():
    #    pass
        # await asyncio.sleep(0.5)
        # for row in last_data_csv:
        #    yield row

    # TODO FIXME: deprecated @session.download; replace with @render.download
    @render.download(
        filename="./tmp.xlsx"
    )
    # async def download_excel():
    def download_excel():
        # await asyncio.sleep(2)
        # substr = input.substr_select()
        # matl = input.mat1_select()
        wb = gc.generate_excel_file()
        # output = StringIO()
        output = BytesIO()
        # output = "./tmp.xlsx"
        wb.save(output)
        xxx = ["1, 2, 3\n", "4, 5, 6\n", "9, 10, 11, 12\n"]
        # return wb
        return bytes(output)
        # for substring in xxx:
        #    yield substring
        # for substring in output.getvalue():
        # return output.getvalue()
        # for substring in output.getvalue():
            # yield bytes(substring)
        #    yield substring
        # return output #.getvalue()
        # for b in output:
        #    yield b
        # return output.getvalue()
        # return save_virtual_workbook(wb)
        # with NamedTemporaryFile(dir='.', suffix='.xlsx', mode='wb') as tmp:
        # with open("./tmp.xslx", "wb") as tmp:
        #    wb.save(tmp)
        #    tmp.seek(0)
        #    #stream = tmp.read()
        #    #return stream
        #    return tmp
        # for 
        # return output.getvalue()
        # return output
        # yield gc.generate_excel_file()

    @render.ui
    def des_controls():
        return ui.TagList(
            ui.input_select("des_airplane_model", "Airplane model:", [""] + des_aircraft_models),
            ui.input_select("des_airline_model", "Airline:", [""] + des_airlines),
            ui.input_numeric("des_simulation_duration", "Simulation duration (years)", 20, min=0.1, max=40, step=0.1),
            ),
            
            # ui.row(
            #
            #     ui.column(
            #         5,
            #         ui.input_select("des_material", "Material", ["Aluminum", "Titanium", "CFRP", "Steel", "Bronze"]),
            #     ),
            #     ui.column(
            #         2,
            #         ui.input_select("static_environment_material_temper", "Temper", ["-T1", "-T2", "-T3", "-T4", "-T5", "-T6"]),
            #     ),
            #     ui.column(
            #         3,
            #         ui.input_select("static_environment_material_treatment", "Treatment", ["Bare", "1", "2", "3"])
            #     ),
            # ),
        # )
        
    # )


app = App(app_ui, server, static_assets=WWW)
