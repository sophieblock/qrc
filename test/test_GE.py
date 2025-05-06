from workflow.simulation.refactor.Process_Library.Gaussian_elim import *
from workflow.simulation.refactor.resources.classical_resources import ClassicalDevice
from workflow.simulation.refactor.broker import Broker
from workflow.simulation.refactor.graph import DirectedEdge
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
    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )
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

    assert np.allclose(results["Matrix"], matrix)
    assert results["Column Idx"] == 0
    assert results["Pivot Idx"] == 2


def test_SwapRows_update():
    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )
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

    swapped_matrix = np.array(
        [[5.0, -3.0, 1.0, 14.0], [2.0, 3.0, 3.0, 15.0], [3.0, 2.0, -4.0, 3.0]]
    )

    result = network.output_nodes[0].output_edges[0]
    assert np.allclose(result.data[0].data, swapped_matrix)


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

    assert np.allclose(
        column_reduced_matrix,
        np.array(
            [[5.0, -3.0, 1.0, 14.0], [0.0, 4.2, 2.6, 9.4], [0.0, 3.8, -4.6, -5.4]]
        ),
    )


def test_generate_column_elimination_network():
    find_pivot = Node(process_model=GE_FindPivot, network_type=Node.INPUT)
    swap_rows = Node(process_model=GE_SwapRows, network_type=Node.NETWORK)
    row_deconstruction = Node(
        process_model=GE_RowDeconstruction,
        network_type=Node.NETWORK,
        continue_itr=False,
    )

    find_pivot.insert_output_node(swap_rows)
    swap_rows.insert_output_node(row_deconstruction)

    network = Network(
        name="Column Elim",
        nodes=[find_pivot, swap_rows, row_deconstruction],
        input_nodes=[find_pivot],
        output_nodes=[row_deconstruction],
        broker=generate_broker(),
    )

    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )

    starting_inputs = [
        (
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        )
    ]
    network.run(network.input_nodes, starting_inputs)

    result = network.output_nodes[0].output_edges[0]
    column_reduced_matrix = result.data[0].data
    assert np.allclose(
        column_reduced_matrix,
        np.array(
            [[5.0, -3.0, 1.0, 14.0], [0.0, 4.2, 2.6, 9.4], [0.0, 3.8, -4.6, -5.4]]
        ),
    )

def test_Gaussian_elimination_network_3x4():
    network = generate_GE_network(broker=generate_broker())

    matrix = np.array(
        [[3.0, 2.0, -4.0, 3.0], [2.0, 3.0, 3.0, 15.0], [5.0, -3.0, 1.0, 14.0]]
    )

    starting_inputs = [
        (
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        )
    ]
    network.run(network.input_nodes, starting_inputs)

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # network.generate_gantt_plot()


def test_Gaussian_elimination_network_4x5():
    network = generate_GE_network(broker=generate_broker())

    matrix = np.array(
        [
            [-8.0, 3.0, -1.0, -1.0, -34.0],
            [4.0, -1.0, 5.0, -3.0, -28.0],
            [-9.0, 1.0, 1.0, -3.0, -43.0],
            [3.0, -3.0, -8.0, -4.0, 16.0],
        ]
    )

    starting_inputs = [
        (
            Data(data=matrix, properties={"Usage": "Matrix"}),
            Data(data=0, properties={"Usage": "Column Idx"}),
        )
    ]
    network.run(network.input_nodes, starting_inputs)

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # network.generate_gantt_plot()


def generate_GE_network_random_matrix(dim1: int, dim2: int):
    matrix = np.random.rand(dim1, dim2)
    network = generate_GE_network(broker=generate_broker())

    return matrix, network


def test_Gaussian_elimination_network_random_5x6():
    matrix, network = generate_GE_network_random_matrix(5, 6)
    network.run(
        network.input_nodes,
        starting_inputs=[
            (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        ],
    )

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # network.generate_gantt_plot()


def test_Gaussian_elimination_network_random_9x10():
    matrix, network = generate_GE_network_random_matrix(9, 10)
    network.run(
        network.input_nodes,
        starting_inputs=[
            (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        ],
    )

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # network.generate_gantt_plot()


def test_Gaussian_elimination_network_random_5x7():
    matrix, network = generate_GE_network_random_matrix(5, 7)
    network.run(
        network.input_nodes,
        starting_inputs=[
            (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        ],
    )

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # network.generate_gantt_plot()


def test_Gaussian_elimination_network_random_8x8():
    matrix, network = generate_GE_network_random_matrix(8, 8)
    df = network.run(
        network.input_nodes,
        starting_inputs=[
            (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        ],
        simulate=False,
    )

    result = network.output_nodes[0].output_edges[0]
    pl, GE_matrix = lu(matrix, permute_l=True)
    assert np.allclose(result.data[0].data, GE_matrix)

    # print(df.to_string())
    # assert 1 == 0
    # network.generate_gantt_plot()

def generate_Gaussian_elimination_network_random_demo(nr, nc, simulate=True):
    matrix, network = generate_GE_network_random_matrix(nr, nc)
    df = network.run(
        network.input_nodes,
        starting_inputs=[
            (
                Data(data=matrix, properties={"Usage": "Matrix"}),
                Data(data=0, properties={"Usage": "Column Idx"}),
            )
        ],
        simulate=simulate,
    )

    #network.reset_network()

    #result = network.output_nodes[0].output_edges[0]
    #pl, GE_matrix = lu(matrix, permute_l=True)
    #assert np.allclose(result.data[0].data, GE_matrix)

    #starting_nodes = network.input_nodes
    #df = network.run(starting_nodes=starting_nodes)

    #network.reset_network()

    # network.generate_gantt_plot()

    return network, df
