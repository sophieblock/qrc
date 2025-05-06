from workflow.simulation.refactor.graph import DirectedEdge, Node
from workflow.simulation.refactor.data import Data
import pytest


def test_valid_input_edge():
    dest_node = Node()
    input_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[dest_node],
    )


def test_invalid_input_edge():
    source_node = Node()
    dest_node = Node()
    with pytest.raises(AssertionError, match="should not have a source node"):
        input_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.INPUT,
            source_node=source_node,
            dest_nodes=[dest_node],
        )


def test_invalid_input_edge_missing_node():
    with pytest.raises(AssertionError, match="is missing a destination node"):
        input_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.INPUT,
            source_node=None,
            dest_nodes=None,
        )


def test_valid_connected_edge():
    dest_node = Node()
    source_node = Node()
    connected_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.CONNECTED,
        source_node=source_node,
        dest_nodes=[dest_node],
    )


def test_invalid_connected_edge_missing_source():
    dest_node = Node()
    with pytest.raises(AssertionError, match="is missing a source node"):
        connected_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.CONNECTED,
            source_node=None,
            dest_nodes=[dest_node],
        )


def test_invalid_connected_edge_missing_dest():
    source_node = Node()
    with pytest.raises(AssertionError, match="is missing a destination node"):
        connected_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.CONNECTED,
            source_node=source_node,
            dest_nodes=None,
        )


def test_valid_output_edge():
    source_node = Node()
    output_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.OUTPUT,
        source_node=source_node,
        dest_nodes=None,
    )


def test_invalid_output_edge():
    source_node = Node()
    dest_node = Node()
    with pytest.raises(AssertionError, match="should not have a destination node"):
        output_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.OUTPUT,
            source_node=source_node,
            dest_nodes=[dest_node],
        )


def test_invalid_output_edge_missing_node():
    with pytest.raises(AssertionError, match="is missing a source node"):
        output_edge = DirectedEdge(
            data=None,
            edge_type=DirectedEdge.OUTPUT,
            source_node=None,
            dest_nodes=None,
        )


def test_update_data():
    dest_node = Node()
    source_node = Node()
    connected_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.CONNECTED,
        source_node=source_node,
        dest_nodes=[dest_node],
    )

    assert connected_edge.data == None
    data = Data(42, {"Usage": "Meaning of Life"})
    connected_edge.update_data(data)
    assert connected_edge.data == (data,)


def test_update_source_node():
    source_node = Node()
    output_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.OUTPUT,
        source_node=source_node,
        dest_nodes=None,
    )
    new_source_node = Node()
    output_edge.update_source_node(new_source_node)
    assert output_edge.source_node == new_source_node
    output_edge.verify_connectivity()


def test_update_source_node_connected():
    dest_node = Node()
    input_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[dest_node],
    )

    assert input_edge.source_node == None
    new_source_node = Node()
    input_edge.update_source_node(new_source_node)
    assert input_edge.source_node == new_source_node
    assert input_edge.edge_type == DirectedEdge.CONNECTED
    input_edge.verify_connectivity()


def test_update_dest_node():
    dest_node = Node()
    input_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[dest_node],
    )
    new_dest_node = Node()
    input_edge.insert_destination_node(new_dest_node)
    input_edge.remove_destination_node(dest_node)
    assert new_dest_node in input_edge.dest_nodes
    input_edge.verify_connectivity()


def test_update_dest_node_connected():
    source_node = Node()
    output_edge = DirectedEdge(
        data=None,
        edge_type=DirectedEdge.OUTPUT,
        source_node=source_node,
        dest_nodes=None,
    )

    new_dest_node = Node()
    output_edge.insert_destination_node(new_dest_node)
    assert new_dest_node in output_edge.dest_nodes
    assert output_edge.edge_type == DirectedEdge.CONNECTED
    output_edge.verify_connectivity()


def test_combine_input_edges():
    dest_node = Node()

    data_1 = Data(42, {"Usage": "Meaning of Life"})
    input_edge_1 = DirectedEdge(
        data=(data_1,),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[dest_node],
    )

    data_2 = Data("Alex", {"Usage": "Name"})
    input_edge_2 = DirectedEdge(
        data=(data_2,),
        edge_type=DirectedEdge.INPUT,
        source_node=None,
        dest_nodes=[dest_node],
    )

    combined_edge = input_edge_1 + input_edge_2
    assert combined_edge.data == (data_1, data_2)
    combined_edge.verify_connectivity()
