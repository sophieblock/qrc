import pytest
from qrew.ast_dag.AstDagConverter import AstDagConverter
import ast
TEST_PATH = "./test/ast_dag/"
# @pytest.fixture
# def converter_objs():
#     """Fixture to create and return a list of AstDagConverter instances for testing"""
#     scripts = [f'{TEST_PATH}/ast_test_script{i}.py' for i in range(1, 11)]
#     return [AstDagConverter(script, filename=True) for script in scripts]


@pytest.fixture(scope="module")
def converter_objs():
    """
    Builds and returns a list of AstDagConverter objects,
    one for each ast_test_script{i}.py for i in range 1 to 10.
    """
    scripts = []
    # NOTE: edit range to change which scripts are tested
    for i in range(1, 11):
        scripts.append(f"{TEST_PATH}/ast_test_script{i}.py")

    converters = []
    for script in scripts:
        converters.append(AstDagConverter(script, filename=True))
    return converters

def test_build_asts(converter_objs):
    """Ensure we can instantiate 10 AstDagConverter objects without error."""
    
    assert len(converter_objs) == 10, "Expected 10 converter objects (one per script)."

def test_get_id(converter_objs):
    """
    Tests that the get_id function returns strings for all nodes,
    and that an actual ID (as opposed to the object ref name) is returned
    for certain node types.
    """
    i = 0  # for printing IDs
    for obj in converter_objs:
        i += 1  # for printing IDs
        id_list = []  # for printing IDs
        for node in ast.walk(obj.tree):
            if not isinstance(node, ast.Assign):  # for printing IDs
                id_list.append(obj.get_id(node))  # for printing IDs
                # self.assertIsInstance(obj.get_id(node), str, 
                #    f"For node {node}, get_id returns a {type(obj.get_id(node))}. "
                #    f"It should return a string.")
                
              
                # If the node is one of these types, ensure we don't get the default "object ref" name
                if isinstance(node, (AstDagConverter.ast_assign_types + 
                                    (ast.Attribute, ast.Name, ast.Call, ast.arg, ast.Constant,
                                    ast.FunctionDef, ast.Starred, ast.Lambda))):
                    assert obj.get_id(node) != str(node), (
                        f"get_id() returns the object reference name for {node} "
                        f"when it should return an ID."
                    )

def test_get_id_assign(converter_objs):
    """Tests that get_id_assign returns a list of strings for Assign nodes."""
    for obj in converter_objs:
        for node in ast.walk(obj.tree):
            if isinstance(node, ast.Assign):
                assign_ids = obj.get_id_assign(node)
                assert isinstance(assign_ids, list), \
                    f"For node {node}, get_id_assign returns {type(assign_ids)}, should return a list."
                for id_string in assign_ids:
                    assert isinstance(id_string, str), \
                        f"ID {id_string} of node {node} is not a string."

def test_get_descendants(converter_objs):
    """
    Tests that get_descendants returns nodes of the type specified in the input.
    """
    node_types_lst = [
        ast.Call, ast.Constant, ast.Name, ast.Attribute,
        ast.FormattedValue, ast.List, ast.Tuple, ast.Set,
        ast.Dict, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp
    ]

    for obj in converter_objs:
        for node in ast.walk(obj.tree):
            for node_type in node_types_lst:
                desc_list = obj.get_descendants(node, (node_type,))
                for desc_node in desc_list:
                    assert isinstance(desc_node, node_type), (
                        f"Descendant node {desc_node} of parent {node} "
                        f"has type {type(desc_node)}, but get_descendants "
                        f"should only return nodes of type {node_type}."
                    )
def test_get_assign_node(converter_objs):
    """
    Tests that get_assign_node returns the correct node for
    file ast_test_script3.py (which is converter_objs[2]).
    """
    
    obj = converter_objs[2]

    # Tests that get_assign_node grabs the Assign node that's "active" at even line numbers
    for i in range(12, 25, 2):
        assign_node = obj.get_assign_node("answer", i)
        assert assign_node is not None, (
            f"get_assign_node('answer', {i}) returned None. Expected an Assign node."
        )
        msg = (f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
               f"when it should have grabbed the one at line {i-1}")
        assert assign_node.lineno == i - 1, msg

    # Tests that get_assign_node ignores non-'answer' nodes
    for i in range(25, 27):
        assign_node = obj.get_assign_node("answer", i)
        msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
               f"should have grabbed line 23")
        assert assign_node.lineno == 23, msg

    # Tests that get_assign_node can recognize other types of variable reassignment nodes
    assign_node = obj.get_assign_node("answer", 32)
    msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
           f"when it should have grabbed line 29")
    assert assign_node.lineno == 29, msg

    # Tests that get_assign_node grabs assign nodes within the appropriate function when passed a
    # FunctionDef node as an argument
    function2 = None
    for some_node in ast.walk(obj.tree):
        if isinstance(some_node, ast.FunctionDef) and obj.get_id(some_node) == "function2":
            function2 = some_node
    assign_node = obj.get_assign_node("answer", 40, funcDef_node=function2)
    msg = (f"get_assign_node grabbed line {assign_node.lineno} "
           f"when it should have grabbed line 39, in function2")
    assert assign_node.lineno == 39, msg

    # Tests that get_assign_node ignores assign nodes in function definitions when no FunctionDef
    # arg is passed
    assign_node = obj.get_assign_node("answer", 40)
    msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
           f"should have grabbed line 29.")
    assert assign_node.lineno == 29, msg

def test_generic_visit(converter_objs):
    """
    Tests that the values of dag_inputs (built using get_dependencies() + generic_visit())
    are lists, and that they don't contain their own keys (no self loops).
    """
    i = 0
    for obj in converter_objs:
        i+=1
        # print(f"\n----------------------Tree {i} Dependencies----------------------")
        # Trigger the visitor
        obj.visit(obj.tree)
        for node in ast.walk(obj.tree):
            if node in obj.dag_inputs.keys():
                inputs = obj.dag_inputs[node]
                # dependencies_ids = []
                # for dep_node in dependencies_ids:
                #     if isinstance(dep_node, ast.Assign):
                #         dependencies_ids.append(obj.get_id_assign(dep_node))
                #     else:
                #         dependencies_ids.append(obj.get_id(dep_node))
                # if isinstance(node, ast.Assign):
                #     print(f"Node '{obj.get_id_assign(node)}' of type '{node.__class__.__name__}' "
                #           f"dependency list: {dependencies_ids}")
                # else:
                #     print(f"Node '{obj.get_id(node)}' of type '{node.__class__.__name__}' "
                #           f"dependency list: {dependencies_ids}")
                # Test whether node's output list contains itself
                # Tests that node is not in its own dependency list
                # Check that node not in its own dependency
                assert node not in inputs, (
                    f"{obj.get_id(node)} of type '{node.__class__.__name__}' "
                    f"is in its own dependency list (self-loop)."
                )
                # Check type is list
                assert isinstance(inputs, list), (
                    f"get_dependencies() on '{obj.get_id(node)}' returned "
                    f"{type(inputs)}, not a list."
                )


def test_get_outputs(converter_objs):
    """Tests that get_outputs returns ['data node'] for data nodes, that 
        it always returns a list, and that it doesn't return a list that contains itself. 
        Uncomment all commented lines to print all process/data node outputs 
        across all trees """
    i = 0
    for obj in converter_objs:
        i+=1
        # print(f"\n----------------------Tree {i} Outputs----------------------") # for printing data outputs
        obj.visit(obj.tree)
        for node in obj.dag_outputs:
            outputs = obj.dag_outputs[node]
            # for printing outputs
            # name_lst = []
            # if outputs == ["DATA NODE"]:
            #     name_lst = ["DATA NODE"]
            # else:
            #     for item in outputs:
            #         name_lst.append(obj.get_id(item))
            # if isinstance(node,ast.Assign):
            #     print(f"Output of {obj.get_id_assign(node)} of type {node.__class__.__name__}: {name_lst}")
            # else:
            #     print(f"Output of {obj.get_id(node)} of type {node.__class__.__name__}: {name_lst}")

            # should not contain itself
            assert node not in outputs, (
                f"{obj.get_id(node)} of type '{node.__class__.__name__}' "
                f"should not output itself."
            )

            assert isinstance(outputs, list), (
                f"get_outputs() on '{node}' returned '{outputs}' of type "
                f"{type(outputs)}, should be a list."
            )

            # If it's a data node, we expect ["DATA NODE"]
            if isinstance(
                node,
                AstDagConverter.ast_assign_types +
                (ast.List, ast.Set, ast.Dict, ast.Tuple, ast.Constant, ast.Starred)
            ):
                assert outputs == ["DATA NODE"], (
                    f"get_outputs() on '{obj.get_id(node)}' returned {outputs}, "
                    f"but '{node}' is a data node. Should return ['DATA NODE']."
                )


# @pytest.mark.skip(reason="Visualization only")
def test_visualize_ast(converter_objs):
    """Produces AST visuals for debugging."""
    for i in range(0, 1):
        converter_objs[i].visualize_ast()


# @pytest.mark.skip(reason="Visualization only")
def test_visualize_dag(converter_objs):
    """Produces DAG visuals for debugging."""
    for i in range(0, len(converter_objs)):
        converter_objs[i].visualize_dag(see_inputs=True, see_outputs=True, filename=f'ast_test_script{i+1}', output_dir="test_outputs/test_astdag/")
