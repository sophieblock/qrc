
from unittest import skip
import ast
import unittest

from qrew.ast_dag.AstDagConverter import AstDagConverter

TEST_PATH = "./test/ast_dag/"

def load_tests(_loader,tests,_ignore):
    return tests

def test_build_asts():
    """ Set up ASTs for testing """
    scripts = []
    # NOTE: edit range to change which scripts are tested
    # (currently tests all scripts, ast_test_script1.py through 10)
    for i in range(1, 11):
        scripts.append(f'{TEST_PATH}/ast_test_script{i}.py')

    _converter_objects = []
    for script in scripts:
        _converter_objects.append(AstDagConverter(script, filename=True))
    return _converter_objects
# TODO Could use more unittests, e.g.
# - directly test remove_arg_dependencies()
# - directly test replace_name_nodes()
# - test that DAG is nonempty
# - test that the number of edges in the DAG = number of nodes - 1
# - test that no self-loops
def _build_asts():
    """Set up ASTs for testing (helper, not a pytest test)."""
    scripts = [f"{TEST_PATH}/ast_test_script{i}.py" for i in range(1, 11)]
    return [AstDagConverter(script, filename=True) for script in scripts]
def build_asts_helper():
    """ Set up ASTs for testing (helper, not a test). """
    scripts = []
    # NOTE: edit range to change which scripts are tested
    for i in range(1, 11):
        scripts.append(f'{TEST_PATH}/ast_test_script{i}.py')

    _converter_objects = []
    for script in scripts:
        _converter_objects.append(AstDagConverter(script, filename=True))
    return _converter_objects
def test_build_asts():
    """Ensure we can instantiate 10 AstDagConverter objects without error."""
    converters = build_asts_helper()
    assert len(converters) == 10, f"Expected 10 converters, got {len(converters)}"
class TestAstToDag(unittest.TestCase):
    """Tests the classes/functions of test_ast_to_dag """

    def setUp(self):
        self.converter_objs = build_asts_helper()

    def test_get_id(self):
        """Tests that the get_id function returns strings for all nodes, and that 
        an actual ID (as opposed to the object ref name) is returned for certain 
        node types. Uncomment commented print statement to print a list of all IDs 
        across all trees """
        i = 0  # for printing IDs
        for obj in self.converter_objs:
            # i += 1  # for printing IDs
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
            # print(f"\nNode IDs in Tree {i}: {id_list}")

    def test_get_id_assign(self):
        """Tests that the get_id_assign function returns a list of strings for all 
        Assign nodes. Uncomment all commented print statements to print all 
        Assign node IDs across all trees """
        i = 0  # for printing IDs
        i += 1  # for printing IDs
        # print(f"\n----------------------Tree {i} Assign Node IDs----------------------")
        for obj in self.converter_objs:
            for node in ast.walk(obj.tree):
                if isinstance(node, ast.Assign):
                    # print(f"Assign Node '{obj.get_id_assign(node)}'")
                    self.assertIsInstance(obj.get_id_assign(node), list,
                        f"For node {node}, get_id_assign returns a {type(obj.get_id_assign(node))}. "
                        f"It should return a list.")
                    for id_string in obj.get_id_assign(node):
                        self.assertIsInstance(id_string, str, f"ID {id_string} of node {node} is not a string.")

    def test_get_descendants(self):
        """Tests that get_descendants returns nodes of the type 
        specified in the input """
        node_types_lst = [ast.Call, ast.Constant, ast.Name, ast.Attribute,
                          ast.FormattedValue, ast.List, ast.Tuple, ast.Set, 
                          ast.Dict, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp]

        for node_type in node_types_lst:
            i = 0
            for obj in self.converter_objs:
                i += 1
                for node in ast.walk(obj.tree):
                    descendant_lst = obj.get_descendants(node, (node_type,))
                    for desc_node in descendant_lst:
                        self.assertTrue(isinstance(desc_node, node_type),
                            f"Descendant node '{desc_node}' of parent '{node}' in Tree {i} has type {type(desc_node)} "
                            f"when get_descendants should only return nodes of type {node_type}")

    def test_get_assign_node(self):
        """Tests that get_assign_node returns the correct node for 
        file ast_test_script3.py"""
        obj = self.converter_objs[2]

        # Tests that get_assign_node grabs the Assign node that's "active" at even line numbers
        for i in range(12, 25, 2):
            assign_node = obj.get_assign_node('answer', i)
            self.assertEqual(assign_node.lineno, i - 1,
                f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
                f"when it should have grabbed the one at line {i-1}")

        # Tests that get_assign_node ignores non-'answer' nodes
        for i in range(25, 27):
            assign_node = obj.get_assign_node('answer', i)
            self.assertEqual(assign_node.lineno, 23,
                f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
                f"when it should have grabbed the one at line 23")
        # Tests that get_assign_node can recognize other types of variable reassignment nodes
        assign_node = obj.get_assign_node('answer', 32)
        self.assertEqual(assign_node.lineno, 29,
            f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
            f"when it should have grabbed the one at line 29")

        # Tests that get_assign_node grabs assign nodes within the appropriate function when passed a
        # FunctionDef node as an argument
        for some_node in ast.walk(obj.tree):
            if isinstance(some_node, ast.FunctionDef) and obj.get_id(some_node) == "function2":
                function2 = some_node
        assign_node = obj.get_assign_node('answer', 40, funcDef_node=function2)
        self.assertEqual(assign_node.lineno, 39,
            f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
            f"when it should have grabbed the one at line 39, in function2")

        # Tests that get_assign_node ignores assign nodes in function definitions when no FunctionDef
        # arg is passed
        assign_node = obj.get_assign_node('answer', 40)
        self.assertEqual(assign_node.lineno, 29,
            f"get_assign_node grabbed the Assign node at line {assign_node.lineno} "
            f"when it should have grabbed the one at line 29")

    def test_generic_visit(self):
        """Tests that the values of the dag_inputs dict (built using get_dependencies()
        and generic_visit()) are lists, and that they don’t contain their keys
        (i.e., no self loops).
        Uncomment all commented lines to print the list of data/process dependencies
        for every relevant AST node across all trees, for verification purposes"""
        i = 0
        for obj in self.converter_objs:
            i += 1
            # print(f"\n----------------------Tree {i} Dependencies----------------------")
            obj.visit(obj.tree)
            for node in ast.walk(obj.tree):
                if node in obj.dag_inputs.keys():
                    inputs = obj.dag_inputs[node]
                    # COMMENT OUT CODE STARTING HERE to stop printing to terminal
                    # dependencies_ids = []
                    # for dep_node in dependencies:
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
                    self.assertNotIn(node, inputs, 
                        f"{obj.get_id(node)} of type '{node.__class__.__name__}' "
                        f"is in its own dependency list.")

                    # Test return is a list
                    self.assertIsInstance(inputs, list,
                        f"get_dependencies() on '{obj.get_id(node)}' returns type "
                        f"{type(inputs)} when it should return a list.")
                    

                    
    def test_get_outputs(self):
        """Tests that get_outputs returns ['data node'] for data nodes, that 
        it always returns a list, and that it doesn't return a list that contains itself. 
        Uncomment all commented lines to print all process/data node outputs 
        across all trees """
        i = 0
        for obj in self.converter_objs:
            i += 1
            # print(f"\n----------------------Tree {i} Outputs----------------------") # for printing data outputs
            obj.visit(obj.tree)
            for node in obj.dag_outputs.keys():
                # for printing outputs
                name_lst = []
                if obj.dag_outputs[node] == ["DATA NODE"]:
                    name_lst = ["DATA NODE"]
                else:
                    for item in obj.dag_outputs[node]:
                        name_lst.append(obj.get_id(item))
                if isinstance(node,ast.Assign):
                    print(f"Output of {obj.get_id_assign(node)} of type {node.__class__.__name__}: {name_lst}")
                else:
                    print(f"Output of {obj.get_id(node)} of type {node.__class__.__name__}: {name_lst}")

                self.assertNotIn(node, obj.dag_outputs[node], 
                    f"{obj.get_id(node)} of type '{node.__class__.__name__}' should not output itself.")

                self.assertIsInstance(obj.dag_outputs[node], list,
                    f"get_outputs() on '{node}' returns '{obj.dag_outputs[node]}' of type "
                    f"{type(obj.dag_outputs[node])} when it should return a list.")

                if isinstance(node, AstDagConverter.ast_assign_types + 
                    (ast.List, ast.Set, ast.Dict, ast.Tuple, ast.Constant, ast.Starred)):

                    self.assertEqual(obj.dag_outputs[node], ["DATA NODE"],
                        f"get_outputs() on '{obj.get_id(node)}' returns '{obj.dag_outputs[node]}' even though "
                        f"'{node}' is a data node. Should return the list ['data node'].")

	# Run this test to visualize the ASTs used for testing
    @skip
    def test_visualize_ast(self):
        """Doesn't test anything. Produces visuals of the ASTs used for testing, for 
        verification purposes. To choose which ASTs to view, change the parameters of 
        the range function. 
        e.g. to view ASTs for scripts 1-3, set range(0, 3).
        To see all ASTs, set range(0, len(self.converter_objs)) 
        """
        for i in range(0, 1):
            self.converter_objs[i].visualize_ast()

    # # Run this test to visualize the DAGs produced
    @skip
    def test_visualize_dag(self):
        """Doesn't test anything. Produces visuals of the DAGs produced by the class, 
        for verification purposes. To choose which DAGs to view, change the parameters 
        of the range function.
        e.g. to view DAGs for scripts 1-3, set range(0, 3).
        Set "see_inputs/see_outputs = False" if you do not want to include edges+nodes 
        from self.dag_inputs/self.dag_outputs """
        for i in range(0, len(self.converter_objs)):
            self.converter_objs[i].visualize_dag(see_inputs=True, see_outputs=True)
# @pytest.fixture(scope="module")
# def converter_objs():
#     """
#     Builds and returns a list of AstDagConverter objects,
#     one for each ast_test_script#.py from 1 to 10.
#     """
#     scripts = []
#     # NOTE: edit range to change which scripts are tested
#     for i in range(1, 11):
#         scripts.append(f"{TEST_PATH}/ast_test_script{i}.py")

#     converters = []
#     for script in scripts:
#         converters.append(AstDagConverter(script, filename=True))
#     return converters

# def test_build_asts(converter_objs):
#     """
#     A basic test that ensures we can build ASTs from each script
#     (ast_test_script1.py ... ast_test_script10.py) without error.
#     If there's a parsing or import error, it would fail here.
#     """
#     # Just confirm we actually got 10 converter objects
#     assert len(converter_objs) == 10, "Expected 10 converter objects (one per script)."

# def test_get_id2(converter_objs):
#     """
#     Tests that the get_id function returns strings for all nodes,
#     and that an actual ID (as opposed to the object ref name) is returned
#     for certain node types.
#     """
#     for obj in converter_objs:
#         for node in ast.walk(obj.tree):
#             val = obj.get_id(node)
#             # Check that the returned value is a string
#             assert isinstance(val, str), (
#                 f"For node {node}, get_id() returned a {type(val)}. "
#                 f"It should return a string."
#             )

#             # If the node is one of these types, ensure we don't get the default "object ref" name
#             if isinstance(node, (AstDagConverter.ast_assign_types + 
#                                  (ast.Attribute, ast.Name, ast.Call, ast.arg, ast.Constant,
#                                   ast.FunctionDef, ast.Starred, ast.Lambda))):
#                 assert val != str(node), (
#                     f"get_id() returns the object reference name for {node} "
#                     f"when it should return an ID."
#                 )

# def test_get_id_assign(converter_objs):
#     """Tests that get_id_assign returns a list of strings for Assign nodes."""
#     for obj in converter_objs:
#         for node in ast.walk(obj.tree):
#             if isinstance(node, ast.Assign):
#                 assign_ids = obj.get_id_assign(node)
#                 assert isinstance(assign_ids, list), \
#                     f"For node {node}, get_id_assign returns {type(assign_ids)}, should return a list."
#                 for id_string in assign_ids:
#                     assert isinstance(id_string, str), \
#                         f"ID {id_string} of node {node} is not a string."

# def test_get_descendants(converter_objs):
#     """Tests that get_descendants returns nodes of the expected type."""
#     node_types_lst = [ast.Call, ast.Constant, ast.Name, ast.Attribute, ast.FormattedValue, 
#                       ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, 
#                       ast.GeneratorExp, ast.DictComp]

#     for node_type in node_types_lst:
#         for obj in converter_objs:
#             for node in ast.walk(obj.tree):
#                 descendant_lst = obj.get_descendants(node, (node_type,))
#                 for desc_node in descendant_lst:
#                     assert isinstance(desc_node, node_type), \
#                         f"Descendant {desc_node} of {node} is {type(desc_node)}, should be {node_type}"
# def test_get_descendants2(converter_objs):
#     """
#     Tests that get_descendants returns nodes of the type specified in the input.
#     """
#     node_types_lst = [
#         ast.Call, ast.Constant, ast.Name, ast.Attribute,
#         ast.FormattedValue, ast.List, ast.Tuple, ast.Set,
#         ast.Dict, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp
#     ]

#     for obj in converter_objs:
#         for node in ast.walk(obj.tree):
#             for node_type in node_types_lst:
#                 desc_list = obj.get_descendants(node, (node_type,))
#                 for desc_node in desc_list:
#                     assert isinstance(desc_node, node_type), (
#                         f"Descendant node {desc_node} of parent {node} "
#                         f"has type {type(desc_node)}, but get_descendants "
#                         f"should only return nodes of type {node_type}."
#                     )
# def test_get_assign_node(converter_objs):
#     """
#     Tests that get_assign_node returns the correct node for
#     file ast_test_script3.py (which is converter_objs[2]).
#     """
#     # The 3rd script (index=2) is ast_test_script3.py
#     obj = converter_objs[2]

#     #
#     # 1) Test: Even line numbers 12..24 => expect the "active" assignment at line (i-1)
#     #
#     for i in range(12, 25, 2):
#         assign_node = obj.get_assign_node("answer", i)
#         assert assign_node is not None, (
#             f"get_assign_node('answer', {i}) returned None. Expected an Assign node."
#         )
#         msg = (f"get_assign_node grabbed line {assign_node.lineno} "
#                f"when it should have grabbed line {i-1}")
#         assert assign_node.lineno == i - 1, msg

#     #
#     # 2) Test: For lines 25..26, we want line 23
#     #
#     for i in range(25, 27):
#         assign_node = obj.get_assign_node("answer", i)
#         msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
#                f"should have grabbed line 23")
#         assert assign_node.lineno == 23, msg

#     #
#     # 3) Test: "answer" at line 32 => expect line 29
#     #
#     assign_node = obj.get_assign_node("answer", 32)
#     msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
#            f"should have grabbed line 29")
#     assert assign_node.lineno == 29, msg

#     #
#     # 4) Test scoping. If we pass function2, we want the assignment at line 39
#     #
#     function2 = None
#     for some_node in ast.walk(obj.tree):
#         if isinstance(some_node, ast.FunctionDef) and obj.get_id(some_node) == "function2":
#             function2 = some_node
#     assign_node = obj.get_assign_node("answer", 40, funcDef_node=function2)
#     msg = (f"get_assign_node grabbed line {assign_node.lineno} "
#            f"when it should have grabbed line 39, in function2")
#     assert assign_node.lineno == 39, msg

#     #
#     # 5) Without passing funcDef_node, we expect line 29 at line 40
#     #
#     assign_node = obj.get_assign_node("answer", 40)
#     msg = (f"get_assign_node grabbed line {assign_node.lineno}, "
#            f"should have grabbed line 29.")
#     assert assign_node.lineno == 29, msg

# def test_generic_visit(converter_objs):
#     """
#     Tests that the values of dag_inputs (built using get_dependencies() + generic_visit())
#     are lists, and that they don't contain their own keys (no self loops).
#     """
#     for obj in converter_objs:
#         # Trigger the visitor
#         obj.visit(obj.tree)
#         for node in ast.walk(obj.tree):
#             if node in obj.dag_inputs:
#                 inputs = obj.dag_inputs[node]
#                 # Check that node not in its own dependency
#                 assert node not in inputs, (
#                     f"{obj.get_id(node)} of type '{node.__class__.__name__}' "
#                     f"is in its own dependency list (self-loop)."
#                 )
#                 # Check type is list
#                 assert isinstance(inputs, list), (
#                     f"get_dependencies() on '{obj.get_id(node)}' returned "
#                     f"{type(inputs)}, not a list."
#                 )

# # def test_get_outputs(converter_objects):
# #     """Tests that get_outputs returns correct values for data nodes."""
# #     for obj in converter_objects:
# #         obj.visit(obj.tree)
# #         for node in obj.dag_outputs:
# #             assert node not in obj.dag_outputs[node], \
# #                 f"{obj.get_id(node)} should not output itself."
# #             assert isinstance(obj.dag_outputs[node], list), \
# #                 f"Expected list, got {type(obj.dag_outputs[node])}."
# #             if isinstance(node, AstDagConverter.ast_assign_types + 
# #                           (ast.List, ast.Set, ast.Dict, ast.Tuple, ast.Constant, ast.Starred)):
# #                 assert obj.dag_outputs[node] == ["DATA NODE"], \
# #                     f"Expected ['DATA NODE'], got {obj.dag_outputs[node]}"
# def test_get_outputs(converter_objs):
#     """
#     Tests that get_outputs returns ['DATA NODE'] for data nodes, 
#     that it always returns a list, 
#     and that it doesn't return a list that contains itself.
#     """
#     for obj in converter_objs:
#         obj.visit(obj.tree)
#         for node in obj.dag_outputs:
#             outputs = obj.dag_outputs[node]

#             # 1) Should not contain itself
#             assert node not in outputs, (
#                 f"{obj.get_id(node)} of type '{node.__class__.__name__}' "
#                 f"should not output itself."
#             )

#             # 2) Must be a list
#             assert isinstance(outputs, list), (
#                 f"get_outputs() on '{node}' returned '{outputs}' of type "
#                 f"{type(outputs)}, should be a list."
#             )

#             # 3) If it's a data node, we expect ["DATA NODE"]
#             if isinstance(
#                 node,
#                 AstDagConverter.ast_assign_types +
#                 (ast.List, ast.Set, ast.Dict, ast.Tuple, ast.Constant, ast.Starred)
#             ):
#                 assert outputs == ["DATA NODE"], (
#                     f"get_outputs() on '{obj.get_id(node)}' returned {outputs}, "
#                     f"but '{node}' is a data node. Should return ['DATA NODE']."
#                 )


# @pytest.mark.skip(reason="Visualization only")
# def test_visualize_ast(converter_objects):
#     """Produces AST visuals for debugging."""
#     for i in range(0, 1):
#         converter_objects[i].visualize_ast()

# @pytest.mark.skip(reason="Visualization only")
# def test_visualize_dag(converter_objects):
#     """Produces DAG visuals for debugging."""
#     for i in range(0, len(converter_objects)):
#         converter_objects[i].visualize_dag(see_inputs=True, see_outputs=True)

