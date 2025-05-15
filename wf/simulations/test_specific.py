from qrew.simulation.refactor.Process_Library.Gaussian_Elim import *
from qrew.simulation.refactor.resources.classical_resources import ClassicalDevice
from qrew.simulation.refactor.broker import Broker
from qrew.simulation.refactor.resources.classical_resources import ClassicalAllocation, ClassicalResource
from qrew.simulation.refactor.utilities import InitError
from qrew.simulation.refactor.graph import DirectedEdge,AllocationError
from scipy.linalg import lu




def generate_broker():
    supercomputer = ClassicalDevice(
        device_name="Supercomputer",
        processor_type="CPU",
        RAM=100 * 10**9,
        properties={"Cores": 20, "Clock Speed": 3 * 10**9},
    )

    broker = Broker(classical_devices=[supercomputer])
    return broker


def test_FindPivot_update():
    # matrix = np.array(
    #     [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    # )
    matrix = A = np.array([
    [1.,2.,3.],
        [4.,5.,6.],
        [7.,8.,9.]
    ])
    findpivot = GE_FindPivot(
        inputs=[
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    )

    assert findpivot.validate_data()
    findpivot.update()
    results = {}

    for result in findpivot.generate_output():
        results[result.properties["Usage"]] = result.data
    print(results["Matrix"])
    assert np.allclose(results["Matrix"], matrix)
    assert results["Column Idx"] == 0
    assert results["Pivot Idx"] == 2


def test_SwapRows_update():
    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )
    # matrix = A = np.array([
    # [1.,2.,3.],
    #     [4.,5.,6.],
    #     [7.,8.,9.]
    # ])
    swaprows = GE_SwapRows(
        inputs=[
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
            Data(data=2, properties={"Usage": "Pivot Idx"}),
        ]
    )

    assert swaprows.validate_data()
    swaprows.update()
    results = {}

    for result in swaprows.generate_output():
        results[result.properties["Usage"]] = result.data

    swapped_matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )
    assert np.allclose(results["Matrix"], swapped_matrix)
    assert results["Column Idx"] == 0


def test_RowReduction_update():
    matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    rowreduction = GE_RowReduction(
        inputs=[
            Data(data=matrix[0], properties={"Usage": "Principle Row"}),
            Data(data=matrix[1], properties={"Usage": "Reduction Row", "Row Idx": 1}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ],
        row_idx=1,
    )

    assert rowreduction.validate_data()
    rowreduction.update()
    results = {}

    for result in rowreduction.generate_output():
        results[result.properties["Usage"]] = result.data

    reduction_factor = matrix[1][0] / matrix[0][0]
    for i in range(4):
        matrix[1][i] = matrix[1][i] - matrix[0][i] * reduction_factor

    assert np.allclose(results["Reduced Row"], matrix[1])


def test_RowDeconstruction_update():
    matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    rowdeconstruction = GE_RowDeconstruction(
        inputs=[
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    )
    assert rowdeconstruction.validate_data()
    rowdeconstruction.update()
    results = {}

    for result in rowdeconstruction.generate_output():
        if "Row Idx" in result.properties.keys():
            results["Reduction Row " + str(result.properties["Row Idx"])] = result.data
        else:
            results[result.properties["Usage"]] = result.data

    assert np.allclose(results["Matrix"], matrix)
    assert results["Column Idx"] == 0
    assert np.allclose(results["Principle Row"], matrix[0])
    assert np.allclose(results["Reduction Row 1"], matrix[1])
    assert np.allclose(results["Reduction Row 2"], matrix[2])


def test_RowReconstruction_update():
    matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    reduced_row_idx1 = np.array([0.0, 4.2, 2.6, 9.4])
    reduced_row_idx2 = [0.0, 3.8, -4.6, -5.4]

    rowreconstruction = GE_RowReconstruction(
        column_idx=0,
        matrix_size=3,
        inputs=[
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
            Data(
                data=reduced_row_idx1,
                properties={"Usage": "Reduced Row", "Row Idx": 1},
            ),
            Data(
                data=reduced_row_idx2,
                properties={"Usage": "Reduced Row", "Row Idx": 2},
            ),
        ],
    )

    assert rowreconstruction.validate_data()
    rowreconstruction.update()

    results = {}

    for result in rowreconstruction.generate_output():
        results[result.properties["Usage"]] = result.data

    reconstructed_matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [0.0, 4.2, 2.6, 9.4], [0.0, 3.8, -4.6, -5.4]]
    )
    assert np.allclose(results["Matrix"], reconstructed_matrix)


def test_pivot_swap_network():
    find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
    swap_rows = Node(process_model=GE_SwapRows, network_type=Node.OUTPUT)

    find_pivot.insert_output_node(swap_rows)

    network = Network(
        name="Pivot Swap",
        nodes=[find_pivot, swap_rows],
        input_nodes=[find_pivot],
        output_nodes=[swap_rows],
        broker=generate_broker(),
    )

    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )

    starting_inputs = [
        [
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    ]
    network.run(network.input_nodes, starting_inputs)
    network.describe
    swapped_matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    result = network.output_nodes[0].output_edges[0]
    print(result)
    assert np.allclose(result.data[0].data, swapped_matrix), f'Got: {result.data[0].data}'


def test_generate_row_reduction_network():
    row_deconstruction = Node(
        process_model=GE_RowDeconstruction, network_type=Node.INPUT, continue_itr=False
    )

    network = Network(
        name="Row Reduction",
        nodes=[row_deconstruction],
        input_nodes=[row_deconstruction],
        output_nodes=[],
        broker=generate_broker(),
    )

    matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    starting_inputs = [
        [
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    ]
    network.run(network.input_nodes, starting_inputs)

    result = network.output_nodes[0].output_edges[0]
    column_reduced_matrix = result.data[0].data
    print(f"\n\nresult: {result.data[0].data}\n")
    assert np.allclose(
        column_reduced_matrix,
        np.array(
            [[5.0, -3.0, 1.0, 14.0], [0.0, 4.2, 2.6, 9.4], [0.0, 3.8, -4.6, -5.4]]
        ),
    )
def generate_GE_network_random_matrix(dim1: int, dim2: int):
    matrix = np.random.rand(dim1, dim2)
    network = generate_GE_network(broker=generate_broker())

    return matrix, network
def generate_Gaussian_elimination_network_random(nr,nc,simulate=True):
    matrix,network = generate_GE_network_random_matrix(nr,nc)
    df = network.run(
        network.input_nodes,
        starting_inputs=[
            (Data(data=matrix,properties={"Usage":"Matrix"}),
            (Data(data=0, properties={"Usage":"Column Idx"})),
            )
        ],
        simulate=simulate
    )

    # network.generate_gantt_plot()
    return network,df
def test_GE_gantt_and_graph():
    import matplotlib.pyplot as plt
    import networkx as nx
    from qrew.results import visualize_graph, visualize_graph_from_nx
    from qrew.simulation.refactor.utilities import publish_gantt, publish_gantt2
    network,df = generate_Gaussian_elimination_network_random(3,4)

    # 2) Extract the NetworkX graph
    nx_graph, _ = network.to_networkx()

    # 3) Create a 2-row figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 14),gridspec_kw={"height_ratios": [1, 1.5]},
    )

    # — Top: Gantt chart —
    axes[0].set_title("Workflow Simulation Results", fontsize=20)
    publish_gantt(
        dataframe=df,
        ax=axes[0],
        show=False,
        frame_off=False,
        cmap="jet",
    )

    # — Bottom: static NetworkX drawing —
    axes[1].set_title("GE Network", fontsize=15)
    # network.visualize(show_edge_labels=True,ax=axes[1])
    # # You can choose a layout of your liking:
    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw(
        nx_graph,
        pos,
        ax=axes[1],
        with_labels=True,
        node_size=300,
        node_color="#a6e2f7",
        edge_color="#000000",
    )
    # Optional: tune margins so labels aren’t clipped
    axes[1].margins(0.1)
    plt.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.93, bottom=0.07)

    # 4) Tidy up, save and show
    # plt.tight_layout()
    fig.savefig("gantt_and_network.png", dpi=300, bbox_inches="tight")
    # plt.show()
def test_GE_gantt_and_graph2():
    import matplotlib.pyplot as plt
    import networkx as nx
    from qrew.results import visualize_graph, visualize_graph_from_nx
    from qrew.simulation.refactor.utilities import publish_gantt, publish_gantt2
    network,df = generate_Gaussian_elimination_network_random(3,4)

    # 2) Extract the NetworkX graph
    nx_graph, _ = network.to_networkx()

    # 3) Create a 2-row figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 14),gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.2},
    )

    # — Top: Gantt chart —
    axes[0].set_title("Workflow Simulation Results", fontsize=20)
    publish_gantt(
        dataframe=df,
        ax=axes[0],
        show=False,
        frame_off=False,
        cmap="jet",
    )

    # — Bottom: static NetworkX drawing —
    axes[1].set_title("GE network", fontsize=10)
    network.visualize(show_edge_labels=True,ax=axes[1])
    # final tweaks so labels/titles don’t overlap
    plt.tight_layout(pad=2.0)
    fig.subplots_adjust(top=0.93, bottom=0.07)
    # 4) Tidy up, save and show
    plt.tight_layout()
    fig.savefig("gantt_and_network2.png", dpi=300, bbox_inches="tight")
    # plt.show()
def test_GE_viz():
    import matplotlib.pyplot as plt
    from qrew.results import publish_resource_usage_history, visualize_graph
    from qrew.simulation.refactor.utilities import publish_gantt, publish_gantt2
    network,df = generate_Gaussian_elimination_network_random(3,4)
    # network.visualize(show_edge_labels=True)
    # network.generate_gantt_plot(show=True)
    # print(df.columns)
    nx_graph, mpg = network.to_networkx()

    visualize_graph

    # create a 3-row figure for our three Gantt plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 16))
    # Workflow simulation results
    axes[0].set_title("Workflow Simulation Results", fontsize=20)
    publish_gantt(
        dataframe=df,
        ax=axes[0],
        show=False,
        
        cmap="jet",
        frame_off=False,
    )
    
    # Most time-intensive tasks
    axes[1].set_title("Most Time-Intensive Tasks", fontsize=20)
    publish_gantt2(
        dataframe=df,
        ax=axes[1],
        show=False,
        cmap="jet",
        frame_off=False,
        key="Duration",
        ascending=False,
    )
    

    # Most memory-intensive tasks
    axes[2].set_title("Most Memory-Intensive Tasks", fontsize=20)

    publish_gantt2(
        dataframe=df,
        ax=axes[2],
        show=False,
        cmap="jet",
        frame_off=False,
        key="Memory [B]",
        ascending=False,
    )
    
    plt.tight_layout()
    fig.savefig("GE_gantt_plots.png", dpi=300, bbox_inches="tight")
    # plt.show()
def plot_GE_gantt(df, save_path: str = "GE_gantt.png"):
    import matplotlib.pyplot as plt

    from qrew.simulation.refactor.utilities import publish_gantt, publish_gantt2

    
    # Single‐subplot Gantt
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title("Workflow Simulation Results", fontsize=20)

    publish_gantt(
        dataframe=df,
        ax=axes[0] if False else ax,  # just ax
        show=False,
        cmap="jet",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax

def plot_GE_network(network,save_path: str = "GE_network.png"):
    import matplotlib.pyplot as plt
    
   
    # Single‐subplot network graph
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Network Graph", fontsize=20)

    network.visualize(show_edge_labels=True, ax=ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax
def plot_sep():
    network, df = generate_Gaussian_elimination_network_random(3, 4)
    fig1, ax1 = plot_GE_gantt(df,"GE_workflow_gantt.png")
    fig2, ax2 = plot_GE_network(network,"GE_network.png")



def test_Gaussian_elimination_network_4x5():
    network = generate_GE_network(broker=generate_broker())

    matrix = np.array([
    [1.,2.,3.],
        [4.,5.,6.],
        [7.,8.,9.]
    ])

    starting_inputs = [
        (
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        )
    ]
    network.run(network.input_nodes, starting_inputs)
    # assert False
    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

def test_decomposed_GE_network_run():

    matrix = np.random.rand(4, 4)
    network = generate_GE_network(broker=generate_broker())
    all_nodes = network.nodes
    for n in all_nodes:
        print(n, f" required resources: {n.process.required_resources}")
    starting_nodes = network.input_nodes
    remaining_nodes = copy.copy(starting_nodes)
    active_nodes = []
    network.status = Network.ACTIVE
    node_idx = 0
    node = remaining_nodes[node_idx]
    print(f'current node: {node}, required_resources: {node.process.required_resources}')
    while node_idx < len(remaining_nodes):
        node = remaining_nodes[node_idx]
        print(f'current node: {node}')
        try:
            # this snippet is testing the code self.__initialize_node(node)
            # Node.start method decomposed (and __initialize_process_with_inputs)
            input_data = []
            for input_edge in node.input_edges:
                print(f"input_edge: {input_edge}")
                for data in input_edge.data:
                    input_data.append(data)
            print(f"initial input data: {input_data}")
            node.process = node.model(inputs=input_data, **node.kwargs)
            assert (
                node.process.validate_data()
            ), f"Invalid input data for process {str(node.process)}"
            # Node.__allocate_resources method decomposed
            node.process.status = Process.ACTIVE
            node.allocation = Allocation()
            time_till_completion = node.process._compute_classical_process_update_time(
                node.allocation
            )
            print(f"time_till_completion: {time_till_completion}")
            node.process.time_till_completion = time_till_completion
            active_nodes.append(node)
        except (AllocationError, InitError):
            node_idx = node_idx + 1
            continue

def test_invalid_input_properties_GE():
    matrix = np.random.rand(4, 4)
    starting_inputs= (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        
    
    network = generate_GE_network(broker=generate_broker())
    starting_nodes = network.input_nodes
    remaining_nodes = copy.copy(starting_nodes)
    
    network.status = Network.ACTIVE
    node_idx = 0
    node = remaining_nodes[node_idx]
    input_edge = DirectedEdge(
                    data=starting_inputs,
                    edge_type="INPUT",
                    source_node=None,
                    dest_nodes=[node],
                )
    node.append_input_edge(input_edge)
    print(f'current node: {node}, required_resources: {node.process.required_resources}')
    input_data = []
    for input_edge in node.input_edges:
        print(f"input_edge: {input_edge}")
        for data in input_edge.data:
            input_data.append(data)
    print(f"initial input data: {input_data}")
    node.process = node.model(inputs=input_data, **node.kwargs)
    assert (
        node.process.validate_data()
    ), f"Invalid input data for process {str(node.process)}"
    # Node.__allocate_resources method decomposed
    node.process.status = Process.ACTIVE
    node.allocation = Allocation()
    time_till_completion = node.process._compute_classical_process_update_time(
        node.allocation
    )
    print(f"time_till_completion: {time_till_completion}")
    node.process.time_till_completion = time_till_completion

    # while node_idx < len(remaining_nodes):
    #     node = remaining_nodes[node_idx]
    #     print(f'current node: {node}')
    #     try:
    #         # this snippet is testing the code self.__initialize_node(node)
    #         node.start()
    #         if node.process.required_resources is None:
    #             allocation = Allocation()

    #         node.process.required_resources

            
    #         active_nodes.append(node)


    # print(network.nodes)

import pytest
def network_run_helper(network, starting_nodes,starting_inputs):
    assert all(
        node.network_type == Node.INPUT for node in starting_nodes
    ), f"Given Nodes {starting_nodes} must all be INPUT nodes of the Network"

    if isinstance(starting_inputs, list):
        assert len(starting_nodes) == len(
            starting_inputs
        ), f"Each Node in starting_nodes must have a corresponding Data object(s) in starting_inputs"

        for idx in range(len(starting_nodes)):
            inputs = starting_inputs[idx]
            node = starting_nodes[idx]
            input_edge = DirectedEdge(
                data=tuple(inputs),
                edge_type="INPUT",
                source_node=None,
                dest_nodes=[node],
            )
            node.append_input_edge(input_edge)
    else:
        assert all(
            len(node.input_edges) > 0 for node in starting_nodes
        ), f"One or more Nodes in starting_nodes is not assigned an input DirectedEdge"

    network.reset_network()
    return network, starting_nodes

def test_pivot_swap_network2():
    find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
    swap_rows = Node(process_model=GE_SwapRows, network_type=Node.OUTPUT)

    find_pivot.insert_output_node(swap_rows)

    network = Network(
        name="Pivot Swap",
        nodes=[find_pivot, swap_rows],
        input_nodes=[find_pivot],
        output_nodes=[swap_rows],
        broker=generate_broker(),
    )
    matrix = np.array([
    [1.,2.,3.],
        [4.,5.,6.],
        [7.,8.,9.]
    ])
    starting_nodes =network.input_nodes

    # break test: Data(data=0, properties={"Usage": "boop"}),
    starting_inputs = [
        [
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "boop"}),
            # Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    ]
    network, starting_nodes = network_run_helper(network, starting_nodes, starting_inputs)
    remaining_nodes = copy.copy(starting_nodes)
    active_nodes = []
    # --- Pre‐flight deadlock detection ---
     # --- PRE-FLIGHT: leverage Process.validate_data_properties to get the clean InitError ---
    for node in remaining_nodes:
        # gather the Data list exactly as Node.start would
        input_data = [
            d
            for edge in node.input_edges
            for d in edge.data
        ]
        try:
            # instantiate the process directly to trigger its own validation
            node.model(inputs=input_data)
        except InitError as e:
            pytest.fail(str(e))
    can_init = [n for n in remaining_nodes if n.ready_to_start()]
    if not can_init:
        msgs = []
        for node in remaining_nodes:
            inputs = [d for e in node.input_edges for d in e.data]
            msgs.append(f"\nBlocked Node {node.id}:{node.process.__class__.__name__}")
            msgs.append(f"  Expected: {node.expected_input_properties}")
            msgs.append(f"  Actual:   {[d.properties for d in inputs]}")
        pytest.fail("Cannot start any input node – mismatched properties:" + "".join(msgs))
    logger.debug(f"Nodes in {network} that are ready to start (inputs are validated and available)")
    
    elapsed_time = 0.0
    execution_idx = 0
    while len(remaining_nodes) != 0 or len(active_nodes) != 0:
        network.status = Network.ACTIVE
        prev_remaining = list(remaining_nodes)
        prev_active    = list(active_nodes)
        did_initialize = False
        did_complete   = False
        node_idx = 0
        while node_idx < len(remaining_nodes):
            node = remaining_nodes[node_idx]
            try:
                network._initialize_node(node)
                logger.debug(f"Attempting to init {node} with input edges {node.input_edges}")
                active_nodes.append(node)
                remaining_nodes.remove(node)
                did_initialize = True
                network.execution_order.append(node)
                node.process.update()

                assert isinstance(node.allocation, ClassicalAllocation)
                memory_usage = node.allocation.allocated_memory
            except (AllocationError, InitError) as e:
                logger.debug(f"Could not init {node}: {e}")
                network.describe
                node_idx += 1
                continue
            execution_idx = execution_idx + 1
        
        
        
        assert all(
            node.process.status == Process.ACTIVE for node in active_nodes
        ), f"One or more Nodes in {active_nodes} have not been allocated resources"
        min_timestep = np.inf

        for node in active_nodes:
            if node.process.time_till_completion < min_timestep:
                min_timestep = node.process.time_till_completion

        update_timestep = min_timestep
        node_idx = 0
        elapsed_time = elapsed_time + update_timestep
        prev_active = active_nodes.copy()
        prev_remaining  = remaining_nodes.copy()
        logger.debug(f'timestep: {update_timestep:.5e}')
        while node_idx < len(active_nodes):
            active_node = active_nodes[node_idx]
            active_node.execute_process(update_timestep) # network.__update_node(active_node, update_timestep)
            
            if active_node.process.status == Process.COMPLETED:
                
                if active_node.process.dynamic:
                    network.__extend_dynamic_node(active_node)
               
                # next 10 lines are implementing Network.__deallocate_resources(node)
                if active_node.process.required_resources is None:
                    active_node.allocation = None
                else:
                    try:
                        network.broker.request_deallocation(active_node.allocation)
                    except:  ## TODO specify expected exception error
                        raise AllocationError(f"Node {active_node.id} resource deallocation FAILED")
                    active_node.allocation = None
                active_node.complete()
                logger.debug(f"{active_node} just completed")
                did_complete = True

                active_nodes.remove(active_node)
                remaining_nodes.extend(network.get_prepared_output_nodes(active_node))
                
                continue

            node_idx = node_idx + 1
        if not (did_initialize or did_complete):
            unstarted = [node for node in network.nodes if node.process.status == Process.UNSTARTED]
            completed = [node for node in network.nodes if node.process.status == Process.COMPLETED]
            raise RuntimeError(
                f"No progress in Network.run iteration:\n"
                f"  remaining={remaining_nodes}\n"
                f"  active   ={active_nodes}\n"
                f"  unstarted   ={unstarted}\n"
                f"  completed   ={completed}\n"
                f"  all nodes   ={network.nodes}\n"
                # f"Check data-property matching or resource allocation."
            )          
    network.status = Network.INACTIVE
    network.describe

def test_pivot_swap_network():
    find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
    swap_rows = Node(process_model=GE_SwapRows, network_type=Node.OUTPUT)

    find_pivot.insert_output_node(swap_rows)

    network = Network(
        name="Pivot Swap",
        nodes=[find_pivot, swap_rows],
        input_nodes=[find_pivot],
        output_nodes=[swap_rows],
        broker=generate_broker(),
    )

    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )

    starting_inputs = [
        [
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        ]
    ]
    network.run(network.input_nodes, starting_inputs)
    network.describe
    swapped_matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    result = network.output_nodes[0].output_edges[0]
    print(result)
    assert np.allclose(result.data[0].data, swapped_matrix), f'Got: {result.data[0].data}'


if __name__ == "__main__":
    # test_GE_gantt_and_graph()
    # test_GE_gantt_and_graph2()
    # test_RowReconstruction_update()
    # test_GE_viz()
    # test_decomposed_GE_network_run()
    test_pivot_swap_network()
    # test_Gaussian_elimination_network_4x5()
    # test_invalid_input_properties_GE()
    # plot_sep()