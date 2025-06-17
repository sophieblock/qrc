import ast,copy
import networkx as nx
import os
from pyvis.network import Network
from visast import visualise
import webbrowser
from torch.fx.experimental.rewriter import AST_Rewriter
from ..util.log import get_logger,logging

logger = get_logger(__name__) 
#-----------------How to use AstDagConverter class------------------#
#Instantiate AstDagConverter object,
#where script1 is the name of a python file or a string of code
#if script1 is a python file, set filename to True; otherwise,
#if a string of code, set filename to False
#  object1 = AstDagConverter(script1, filename=True)

#Visualize the AST
#  object1.visualize_ast()

#Build the dag_inputs and dag_outputs dictionaries
#  object1.visit(object1.tree)

#Visualize the DAG created
#  object1.visualize_dag()

#Build DAG using code refactor
# NOTE: create_dag() method not yet implemented
# object1.create_dag()


class AstDagConverter(ast.NodeVisitor):
    """Objects of AstDagConverter represent ASTs. Using AST, 
    Class methods find nodes and edges and display visual
    of the network model DAG.

    Authors:
        Ama Koranteng (ama.a.koranteng@boeing.com) """

    #AST node types considered processes: Call, IfExp, Raise, Assert, If, For, While,
    #                                     iterable comprehensions, certain Attribute nodes
    #AST node types consider data: Assigns (except for Lambda functions), iterables,
    #                              Constant nodes, some Name nodes, some Starred nodes

    #node types representing variable assignments
    ast_assign_types = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr)
    #node types representing iterable comprehensions
    ast_iterable_comp_types = (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                               ast.DictComp)
    #node types representing iterables
    ast_iterable_types = ast_iterable_comp_types + (ast.List, ast.Tuple, ast.Set,
                                                    ast.Dict)
    #node types that yield process nodes in the DAG
    #also node types represented in dag_outputs keys
    ast_process_types = ast_iterable_comp_types + (ast.Call, ast.IfExp, ast.If, ast.For,
                                                   ast.While, ast.Raise, ast.Assert)
    #node types that are candidates to yield inputs in the DAG
    ast_base_case_types = ast_process_types + ast_iterable_types + (ast.Name, ast.Constant,
                                                                    ast.NamedExpr)
    #node types represented in dag_inputs keys
    ast_nodes_for_inputs = ast_process_types + ast_iterable_types + ast_assign_types


    #Set filename=True if 'script' is a Python filename (as a string)
    # set filename=False if file is a string of the code itself
    def __init__(self, script, filename):
        #build AST
        if filename:
            with open(script, 'r', encoding="utf-8") as file:
                contents = file.read()
            self._tree = ast.parse(contents)
        else:
            self._tree = ast.parse(script)
        # if filename:
        #     with open(script, 'r', encoding="utf-8") as file:
        #         contents = file.read()
        #     # self._tree = ast.parse(contents)
        # else:
        #     contents = script
        # else:
        #     self._tree = ast.parse(script)
        # Use AST_Rewriter to transform code before AST processing
        # self._tree = AST_Rewriter().rewrite(lambda: exec(contents))

        #adds a 'parent' attribute for each AST node, storing the node's parent node
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        self._dag_inputs = {}
        self._dag_outputs = {}

    @property
    def tree(self):
        return self._tree

    @property
    def dag_inputs(self):
        return self._dag_inputs

    @property
    def dag_outputs(self):
        return self._dag_outputs

    def get_id(self, node):
        """Returns the ID (as a string) of an AST node.
        Returns a name if input node has one of the following types: 
        Call (function calls), Name (variables), FunctionDef (function definitions),
        arg (function arguments), Constant, Starred (starred variables), 
        Lambda (lambda functions), AnnAssign/AugAssign/NamedExpr (types of variable assignments). 
        Otherwise returns the reference (as a string) of the node object."""
        #grabs the id field of the AST node, depending on its type
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.FunctionDef):
            return node.name
        if isinstance(node, ast.arg):
            return str(node.arg)
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            return self.get_id(node.func) + "()"
        if isinstance(node, ast.Starred):
            return self.get_id(node.value)
        if isinstance(node, ast.Lambda):
            return self.get_id(node.parent)
        if isinstance(node, ast.Assign):
            #handles assign nodes with only a single variable
            if len(node.targets) == 1 and not isinstance(node.targets[0], ast.Tuple):
                return self.get_id(node.targets[0])
            #raise TypeError("Input is an Assign node with multiple targets/variables. To get its IDs, call get_id_assign() instead.")
        if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            #handles nodes that assign to a single variable
            if isinstance(node.target, ast.Name):
                return self.get_id(node.target)
            #handles nodes that assign to a more complex expression
            name_node_lst = self.get_descendants(node.target, (ast.Name,))
            self.get_id(name_node_lst[0])
        return str(node)

    def get_id_assign(self, node):
        """Returns the ID(s) (as a list of strings) for Assign nodes"""
        ids = []
        if not isinstance(node, ast.Assign):
            raise TypeError(f"Input '{node}' is not an Assign node")

        for target_node in node.targets:
            #handles when 'targets' field is a Tuple of Name nodes
            if isinstance(target_node, ast.Tuple):
                for subnode in target_node.elts:
                    ids.append(self.get_id(subnode))
            #handles when 'targets' is just a single Name node
            else:
                ids.append(self.get_id(target_node))

        return ids

    def get_descendants(self, node, node_types):
        """Takes in an AST node and a list/tuple of AST node types, returns a list
        of all descendant nodes with the requsted node types. Ignores nodes that 
        just serve as ID holders for Call nodes, and ignores FunctionDef arg nodes"""
        node_types = tuple(node_types)
        descendants = []
        children = []
        children.extend(ast.iter_child_nodes(node))
        
        #recursively finds child nodes of the requested types
        if not children:
            return descendants
        for child in children:
            #omit descendant nodes that are the ID-holder Name node of a Call node, as
            # these nodes don't correspond to real data for our purposes
            if isinstance(node, ast.Call) and isinstance(child, ast.Name) \
               and self.get_id(node) == self.get_id(child):
                pass
            elif isinstance(child, node_types):
                descendants.append(child)
            descendants.extend(self.get_descendants(child, node_types))

        return descendants

    def get_assign_node(self, name, line_num, funcDef_node=None):
        """Given a node id string (name) and a script line number (line_num),
        returns the Assign node of the same id that's active at line_num.
        If the assign node we're looking for is in a function definition,
        set funcDef to the corresponding FunctionDef node
        TODO: may miss assign nodes in nested function definitions -- see
        ast_dag_notes in doc folder for more details"""
        print(f"get_assign_node called with name='{name}', line_num={line_num},"
              f"funDef_node={(funcDef_node.name if funcDef_node else None)}")
        #assign nodes with the id "name"
        assign_nodes = []
        #list of assign nodes that exist within other function definitions
        assigns_to_ignore = []

        #build assigns_to_ignore list
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.Lambda)) and (node is not funcDef_node):
                assigns_to_ignore.extend(self.get_descendants(node, AstDagConverter.ast_assign_types))
        #build assign_nodes list
        for node in ast.walk(self.tree):
            if node not in assigns_to_ignore:
                if isinstance(node, ast.Assign) and name in self.get_id_assign(node):
                    assign_nodes.append(node)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) \
                     and self.get_id(node) == name:
                    assign_nodes.append(node)
        #from assign_nodes list, finds the node closest to (but not after) line_num
        if not assign_nodes:
            return None
        closest_assign_node = assign_nodes[0]
        for assign_node in assign_nodes:
            if (line_num >= assign_node.lineno) and (assign_node.lineno > closest_assign_node.lineno):
                closest_assign_node = assign_node

        return closest_assign_node

    def remove_arg_dependencies(self, def_node, dependencies):
        """Takes as input a FunctionDef/Lambda node and a list of dependencies,
        returns a pruned dependencies list with arg nodes (respresenting function
        arguments) nodes removed. For use in get_dependencies()"""
        #arg nodes to remove from dependencies
        args_to_remove = []
        #list of functionDef node's argument names
        funcDef_arg_names = []

        for dep_node in dependencies:
            #loops through the argument (arg) nodes of def_node to
            # populate funcDef_arg_names
            for arg_node in self.get_descendants(def_node.args, (ast.arg,)):
                funcDef_arg_names.append(self.get_id(arg_node))
            #populates args_to_remove
            if self.get_id(dep_node) in funcDef_arg_names:
                args_to_remove.append(dep_node)

        #removes all nodes in args_to_remove
        dependencies = [dep_node for dep_node in dependencies if dep_node not in args_to_remove]
        return dependencies

    def replace_name_nodes(self, dependencies, funcDef_node=None):
        """Takes in a list of dependencies, and a funcDef_node if the dependency
        list belongs to a node that's inside a function definition. Replaces any
        Name nodes (in dependencies) that represent data elements with their
        respective Assign nodes. Returns pruned dependencies list.
        For use in get_dependencies """
        to_remove = []
        to_add = []

        #finds the respective Assign nodes to add to dependencies list
        for dep_node in dependencies:
            if isinstance(dep_node, ast.Name):
                assign_node = self.get_assign_node(name=self.get_id(dep_node), \
                                                   line_num=dep_node.lineno, funcDef_node=funcDef_node)
                if assign_node is not None:
                    to_add.append(assign_node)
                    #keep Name nodes that represent Assign nodes that assign multiple variables
                    #-allows us to keep track of which variable the input node is dependent on
                    if isinstance(assign_node, ast.Assign) \
                       and ( len(assign_node.targets) > 1 or isinstance(assign_node.targets[0], ast.Tuple) ):
                       pass
                    else:
                        to_remove.append(dep_node)
        #removes Name nodes, then adds Assign nodes to dependencies list
        dependencies = [dep_node for dep_node in dependencies if dep_node not in to_remove]
        dependencies.extend(to_add)

        return dependencies

    def stop_dependencies_recursion(self, child, node):
        """For use in get_dependencies. Takes in a child node and its parent,
        returns True if neither child nor its descendants will yield any dependencies"""
        #builds a list of types of the granchildren of node (i.e. the children of child)
        grandchild_types = []
        for grandchild in ast.iter_child_nodes(child):
            grandchild_types.append(type(grandchild))

        #stops the recursion at Assign nodes that represent Lambda function definitions
        if isinstance(child, ast.Assign) and ast.Lambda in grandchild_types:
            return True
        #stops recursion at ID holders for a Call node
        if isinstance(node, ast.Call) and child == node.func and isinstance(child, ast.Name):
            return True
        #stops recursion at Lambda nodes
        if isinstance(node, ast.Assign) and isinstance(child, ast.Lambda):
            return True
        #stops recursion at arguments of a Lambda function node
        if isinstance(node, (ast.FunctionDef, ast.Lambda)) and child == node.args:
            return True
        #stops recursion at placeholder variables for ListComp, SetComp, etc nodes
        if hasattr(node, 'elt') and child == node.elt:
            return True
        #stops recursion at placeholder variables for DictComp nodes
        if isinstance(node, ast.DictComp) and child in [node.key, node.value]:
            return True
        #stops recursion at variable names for Delete and Assign nodes
        if hasattr(node, 'targets') and child in node.targets:
            return True
        #stops recursion at variable names or placeholders for
        # AugAssign, AnnAssign, NamedExpr, For, and Comprehension nodes
        if hasattr(node, 'target') and child == node.target:
            return True
        # stops recursion at Name nodes that represent annotations 
        # (e.g. in AnnAssign or arg nodes)
        if hasattr(node, 'annotation') and child == node.annotation:
            return True
        #stops recursion at Constant indices in Slice nodes 
        # (which are used for List slicing)
        if isinstance(child, ast.Slice):
            return True
        #stops recursion at child If nodes (since they represent elif statements)
        if isinstance(node, ast.If) and child in node.orelse and isinstance(child, ast.If):
            return True
        #otherwise, keep going!
        else:
            return False

    def get_dependencies(self, node):
        """Given a node, returns a list of AST nodes that represent all data
        and process dependencies of the input node. Meant to be used for the 
        following node types: 
        Call, Literals, all Assign nodes, Attribute, For, While, Raise, Assert, 
        List/Set/etc, If, IfExp, Lambda, and maybe others I've forgetten"""
        dependencies = []

        #for Call and Attribute nodes, gets dependencies from their corresponding
        # FunctionDef or Lambda node
        if isinstance(node, (ast.Call, ast.Attribute)):
            for some_node in ast.walk(self.tree):
                #checks if some_node is the corresponding FunctionDef/Lambda node
                if isinstance(some_node, (ast.FunctionDef, ast.Lambda)) \
                   and self.get_id(some_node) == self.get_id(node):
                    #continue the DepthFS recursion on the FunctionDef/Lambda node
                    dependencies.extend(self.get_dependencies(some_node))
                    #remove arg nodes from dependencies list
                    # (since they're just placeholders for actual data)
                    dependencies = self.remove_arg_dependencies(some_node, dependencies)

        #For If nodes that represent "elif" statements, and its parent If to dependencies list
        if isinstance(node.parent, ast.If) and isinstance(node, ast.If) \
           and node in (node.parent).orelse:
            dependencies.append(node.parent)
        #For Assign nodes contained in If/For/While statements, add the If/For/While
        # statement to dependencies list
        if isinstance(node.parent, (ast.If, ast.IfExp, ast.For, ast.While)) \
           and isinstance(node, AstDagConverter.ast_assign_types):
            dependencies.append(node.parent)

        #converts the ast.iter_child_nodes(node) generator to a list
        children = list(ast.iter_child_nodes(node))

        #one of many base cases in get_dependencies() recursion
        if not children:
            return dependencies
        for child in children:
            #checks whether child represents a variable that a For loop assigns to
            # e.g. in "for elem in listname" , elem would correspond to such a node
            loop_target = False
            if isinstance(child, ast.Name):
                for for_node in ast.walk(self.tree):
                    if isinstance(for_node, ast.For) and child in self.get_descendants(for_node, (ast.Name,)):
                        for name_node in self.get_descendants(for_node.target, (ast.Name,)) + [for_node.target]:
                            if self.get_id(child) == self.get_id(name_node):
                                loop_target = True
                                loop_iterable = for_node.iter
            
            #stop recursion if child or descendants won't yield dependencies
            if self.stop_dependencies_recursion(child, node):
                pass
            #if child node represents a For loop variable, adds corresponding
            # iterable as a dependency
            elif loop_target:
                dependencies.append(loop_iterable)
            #continues recursion for Assign node that assigns multiple variables to multiple values
            elif isinstance(node, ast.Assign) and child in node.targets and isinstance(child, ast.Tuple):
                dependencies.extend(self.get_dependencies(child))
            #add node to dependencies if it corresponds to data or process nodes
            elif isinstance(child, AstDagConverter.ast_base_case_types):
                dependencies.append(child)
            #otherwise, continue with this DepthFS-type recursion, proceeding with child node
            else:
                dependencies.extend(self.get_dependencies(child))

        #remove Lambda arg nodes from dependencies list
        # (since they're just placeholders for actual data)
        if isinstance(node, ast.Lambda):
            dependencies = self.remove_arg_dependencies(node, dependencies)
        #replace Name nodes with their respective Assign nodes
        # (since some Name nodes are just references to variable assignments)
        funcDef_node = None
        for some_node in ast.walk(self.tree):
            if isinstance(some_node, ast.FunctionDef) and node in self.get_descendants(some_node, (type(node),)):
                funcDef_node = some_node
        dependencies = self.replace_name_nodes(dependencies, funcDef_node)
        
        #remove self dependencies
        dependencies = [dep_node for dep_node in dependencies if dep_node != node]
        #remove duplicates
        dependencies = list(set(dependencies))

        return dependencies

    def get_outputs_builtin_call(self, call_node, node_type):
        """Finds iterable/string dependencies for AST nodes that represent Python
        built-in iterable/string methods (e.g. append, extend). node_type is 
        the type of AST node that call_node acts on (e.g. ast.List).
        For use in get_outputs_call()"""
        output_list = []
        for dep_node in self.get_dependencies(call_node):
            if isinstance(dep_node, node_type):
                output_list.append(dep_node)
            elif isinstance(dep_node, ast.Assign) and isinstance(dep_node.value, node_type):
                output_list.append(dep_node.value)
        #print(f"Node {self.get_id(call_node)} output list: {output_list}")
        return output_list

    def get_outputs_call(self, call_node):
        """Returns the data output list of a call node, for use in get_outputs
        TODO: Should return a *copy* of the Assign and Return nodes in the 
        corresponding FunctionDef node -- see ast_dag_notes in doc folder for
        more details"""

        output_list = []
        if not isinstance(call_node, (ast.Call, ast.Attribute)):
            raise TypeError("Input is not a Call or Attribute node.")

        #for Call nodes that correspond to List methods, add the list to output_list
        if self.get_id(call_node) in ['append', 'clear', 'extend', 'insert', 'pop', \
                                      'remove', 'reverse', 'sort', 'join']:
            output_list.extend(self.get_outputs_builtin_call(call_node, ast.List))
        #Sam but for Set methods
        if self.get_id(call_node) in ['add', 'clear', 'difference_update', 'discard', \
                                      'intersection_update', 'pop', 'remove', \
                                      'symmetric_difference_update', 'update']:
            output_list.extend(self.get_outputs_builtin_call(call_node, ast.Set))
        #Dict methods
        if self.get_id(call_node) in ['clear', 'pop', 'popitem', 'update']:
            output_list.extend(self.get_outputs_builtin_call(call_node, ast.Dict))
        #file methods
        if self.get_id(call_node) in ['write', 'writelines']:
            output_list.extend(self.get_outputs_builtin_call(call_node, ast.Constant))
        #string methods
        if self.get_id(call_node) in ['capitalize', 'casefold', 'format', 'format_map', \
                                      'lower', 'swapcase', 'title', 'upper', 'zfill']:
            output_list.extend(self.get_outputs_builtin_call(call_node, ast.Constant))

        #for Call nodes that correspond to cast methods, add the casted item
        if self.get_id(call_node) in ['int', 'float', 'str']:
            output_list.extend(self.get_dependencies(call_node))

        #find the function definition or lambda definition node (if it exists)
        #  and get it's return statements and Assign nodes
        for some_node in ast.walk(self.tree):
            if self.get_id(some_node) == self.get_id(call_node):
                #grabs all assign nodes within the function def
                # ensures that if a variable is reassigned, we only grab the latest
                # reassignment
                assign_nodes = self.get_descendants(some_node, AstDagConverter.ast_assign_types)
                assign_names = [self.get_id(a_node) for a_node in assign_nodes]
                for name in assign_names:
                    output_list.append(self.get_assign_node(name, some_node.end_lineno, some_node))

                # returns a list of some_node's Return/Body nodes
                if isinstance(some_node, ast.FunctionDef):
                    #add all return node dependencies to output_list
                    for return_node in self.get_descendants(some_node, (ast.Return,)):
                        output_list.append(return_node.value)
                        #output_list.extend(self.get_dependencies(return_node))
                if isinstance(some_node, ast.Lambda):
                    output_list.append(some_node.body)
                    #output_list.extend(self.get_dependencies(some_node.body))

        return output_list

    #TODO: SUBSTITUTION: Function call outputs maybe could be smarter than just "return the return statement".
    #                    Can they substitute function argument placeholders with the actual argument passed?
    #                    see ast_dag_notes in doc foler for more details
    #TODO: DATA OUTPUT:  If a process modifies an object but doesn't return it, can we ensure that that
    #                    object is considered data output of that process?
    def get_outputs(self, node):
        """Takes in an AST node, returns a list of its output AST nodes.
        If the input is a data node, returns the list ["DATA NODE"] """

        output_list = []

        #handles Call nodes
        if isinstance(node, (ast.Call, ast.Attribute)):
            output_list.extend(self.get_outputs_call(node))
            return list(set(output_list))

        #handles iterable composition nodes
        if isinstance(node, AstDagConverter.ast_iterable_comp_types):
            pass
        #TODO: can we ensure that, if an object (e.g. a list) gets modified by one
        #      of these node types, that that object gets returned as an output?
        #handles If, For, While, IfExp (ternary operator), Raise, Assert nodes
        elif isinstance(node, (ast.If, ast.For, ast.While)):
            for subnode in node.body + node.orelse:
                if isinstance(subnode, AstDagConverter.ast_assign_types):
                    output_list.append(subnode)
                elif not isinstance(subnode, AstDagConverter.ast_process_types):
                    output_list.extend(self.get_descendants(subnode, (ast.Assign, ast.AnnAssign)))
        elif isinstance(node, ast.IfExp):
            if isinstance(node.parent, AstDagConverter.ast_assign_types):
                output_list.extend([node.body, node.orelse])
            elif not isinstance(subnode, AstDagConverter.ast_process_types):
                output_list.extend(self.get_descendants(node.body, (ast.Assign, ast.AnnAssign)) + \
                                   self.get_descendants(node.orelse, (ast.Assign, ast.AnnAssign)) )
        elif isinstance(node, ast.Raise):
            output_list.append(node.exc)
        elif isinstance(node, ast.Assert):
            if node.msg is not None:
                output_list.append(node.msg)
        else:
            return ["DATA NODE"]

        #remove self dependencies
        output_list = [out_node for out_node in output_list if out_node != node]
        #remove duplicates
        output_list = list(set(output_list))

        return output_list

    def generic_visit(self, node):
        """Overrides ast.NodeVisitor generic_visit() function. When objectName.visit() 
        is called, this method gets called on each node in the AST. 
        Populates self.dag_inputs and self.dag_outputs for the AST. """
        #build a type list of the granchildren of node (i.e. children of child)
        child_types = []
        for child in ast.iter_child_nodes(node):
            child_types.append(type(child))
        #populates dag_inputs and dag_outputs dicts if the current AST node represents either
        # a process, or a data element with other data dependencies (e.g a list of elements)
        # also populates dag_inputs dict if the AST node accesses a class attribute
        if ( isinstance(node, AstDagConverter.ast_nodes_for_inputs) and ast.Lambda not in child_types \
             or isinstance(node, ast.Attribute) and not isinstance(node.parent, ast.Call) ) \
           and not isinstance(node, (ast.List, ast.Set, ast.Tuple, ast.Dict)):

            #populates dag_inputs dict
            #replaces AST nodes that correspond to processes with their respective outputs
            to_add = []
            to_remove = [node] #ensures that we're removing self-dependencies
            inputs = self.get_dependencies(node)
            #populates to_add and to_remove
            for in_node in inputs:
                if in_node in self.get_outputs(node):
                    to_remove.append(in_node)
                if isinstance(in_node, AstDagConverter.ast_process_types) and \
                   not isinstance(in_node, AstDagConverter.ast_iterable_comp_types) and \
                   self.get_outputs(in_node) and \
                   node not in self.get_outputs(in_node):
                    to_add.extend(self.get_outputs(in_node))
                    to_remove.append(in_node)

            inputs.extend(to_add)
            inputs = [in_node for in_node in inputs if in_node not in to_remove]
            inputs = list(set(inputs))
            self.dag_inputs[node] = inputs

        #populates dag_outputs dict
        if isinstance(node, AstDagConverter.ast_process_types):
            outputs = self.get_outputs(node)
            self.dag_outputs[node] = outputs

        #visits all nodes in DepthFS, left to right order
        super().generic_visit(node)

    def visualize_ast(self):
        """Outputs a visual of the AST"""
        visualise.graph(self.tree)

    def to_networkx(self):
        aST = self.tree
        GRAPH = nx.DiGraph()
        rootNodeID = "noRoot"
        edges = []
        labelDictionary = {}

        # Walk the tree, breadth-first, noting all edges.
        for node in ast.walk(aST):
            nodeID = str(node.__class__) + str(id(node))  # Unique name
            nodeLabel = str(node.__class__).split("ast.")[1].split("'>")[0]
            if nodeLabel == "Constant":
                nodeLabel += " " + str(node.value)
            elif nodeLabel == "FunctionDef":
                nodeLabel += " " + str(node.name)
            labelDictionary[nodeID] = nodeLabel

            if rootNodeID == "noRoot":
                rootNodeID = nodeID

            for child in ast.iter_child_nodes(node):
                childNodeID = str(child.__class__) + str(id(child))
                for edge in edges:
                    if edge[1] == childNodeID:
                        childNodeID += str(1)  # IDs aren't unique.  Fix.

                # If child is at the bottom of the tree, it won't get walked.
                # Label it manually.
                if labelDictionary.get(childNodeID) is None:
                    childLabel = str(child.__class__
                                    ).split("t.")[1].split("'>")[0]
                    if childLabel == "Constant":
                        childLabel += " " + str(child.value)

                    labelDictionary[childNodeID] = childLabel
                    if (childLabel == "Load"
                    or childLabel == "Store"
                    or childLabel == "Del"):
                        if hasattr(node, "id"):
                            nodeLabel = str(node.id)
                            labelDictionary[nodeID] = nodeLabel

                GRAPH.add_edge(nodeID, childNodeID)
                edges.append([nodeID, childNodeID])
        return GRAPH

    def set_graph_id(self, node):
        """Given a node, returns a label for the DAG visualization.
        For use in visualize_dag() """
        if isinstance(node, ast.Assign) and (len(node.targets) > 1 or isinstance(node.targets[0], ast.Tuple) ):
            return str(self.get_id_assign(node)) + str(node.lineno)
        if isinstance(node, AstDagConverter.ast_assign_types):
            return self.get_id(node)+str(node.lineno)
        if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Starred, ast.Constant)):
            return self.get_id(node)
        return node.__class__.__name__

    def set_unique_id(self, node):
        """Given a node, returns a unique ID for the graph visualization.
        For use in visualize_dag() """
        if isinstance(node, AstDagConverter.ast_assign_types):
            return self.get_id(node)+str(node.lineno)
        if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Starred, ast.Constant)):
            return self.get_id(node)+str(node.lineno)
        return self.get_id(node)

    def visualize_dag(
            self,
            see_inputs: bool = True,
            see_outputs: bool = True,
            filename: str | None = None,
            output_dir: str | None = None,
            notebook: bool = False,
            cdn_resources: str = "in_line",      # 'remote', 'local', or 'in_line'
            show: bool = False
        ):
        """
        Render the DAG.

        Parameters
        ----------
        see_inputs, see_outputs : bool
            Whether to include edges from dag_inputs / dag_outputs.
        filename : str | None
            Base name (no extension) for the HTML file. Ignored when notebook=True.
        output_dir : str | None
            Directory for the HTML file. Ignored when notebook=True.
        notebook : bool
            If True, return an inline PyVis widget (no file is written).
        """
    
        network = Network(directed=True, notebook=notebook)
        #build dag_inputs and dag_outputs dicts
        self.visit(self.tree)
        
        #inlcude nodes and edges from dag_inputs in visualization
        if see_inputs:
            for node, inputs in self.dag_inputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        network.add_node(node_uid, label=node_gid, color='#dd4b39')
                    else:
                        network.add_node(node_uid, label=node_gid)

                #add all input nodes and their edges
                for in_node in inputs:
                    in_node_uid = self.set_unique_id(in_node)
                    in_node_gid = self.set_graph_id(in_node)

                    if in_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(in_node):
                            network.add_node(in_node_uid, label=in_node_gid, color='#dd4b39')
                        else:
                            network.add_node(in_node_uid, label=in_node_gid)
                    #check for duplicate edge
                    if {'from': in_node_uid, 'to': node_uid, 'arrows': 'to'} not in network.get_edges():
                        network.add_edge(in_node_uid, node_uid)

        #inlcude nodes and edges from dag_outputs in visualization
        if see_outputs:
            for node, outputs in self.dag_outputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        network.add_node(node_uid, label=node_gid, color='#dd4b39')
                    else:
                        network.add_node(node_uid, label=node_gid)

                #add all output nodes and their edges
                for out_node in outputs:
                    out_node_uid = self.set_unique_id(out_node)
                    out_node_gid = self.set_graph_id(out_node)

                    if out_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(out_node):
                            network.add_node(out_node_uid, label=out_node_gid, color='#dd4b39')
                        else:
                            network.add_node(out_node_uid, label=out_node_gid)
                    #check for duplicate edge
                    if {'from': node_uid, 'to': out_node_uid, 'arrows': 'to'} not in network.get_edges():
                        network.add_edge(node_uid, out_node_uid)

        network.toggle_physics(True)
        if notebook:
            # Inline view only – PyVis handles the HTML string; no file on disk
            
     
            # network.show("example.html")
            return network.show("example.html",notebook=True)
        # html_file = f'{filename}.html'
        # network.show(html_file, notebook=False)
        # File output requested
        if filename is None:
            filename = "mygraph"
        if output_dir is None:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f"{filename}.html")

        # write file and open browser
        logger.debug(f"network: {network}")
        network.show(html_path, notebook=False)
  
        if show:
            webbrowser.open(html_path)

    #TODO: Create a DAG out of the dag_inputs and dag_outputs dictionaries,
    #      using the DAG refactor. Boil out data nodes if data_nodes = False
    def create_dag(self, 
                   data_nodes: bool = True, 
                   see_inputs: bool = True, 
                   see_outputs: bool  = True,
                   ) -> Network:
        """TODO: Docstring """

        network = Network(directed=True)
        #build dag_inputs and dag_outputs dicts
        self.visit(self.tree)

        #inlcude nodes and edges from dag_inputs in visualization
        if see_inputs:
            for node, inputs in self.dag_inputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        is_quantum = (('qubit' in node_gid) or ('ircuit' in node_gid))
                        network.add_node(node_uid, 
                                         label=node_gid, 
                                         shape='triangle',
                                         color='#cb30f2' if is_quantum else '#dd4b39',
                                         )
                    else:
                        is_quantum = (node_gid == 'QuantumCircuit')
                        network.add_node(node_uid, 
                                         label=node_gid,
                                         color='#ff33f8' if is_quantum else '#29cbf0',
                                         )

                #add all input nodes and their edges
                for in_node in inputs:
                    in_node_uid = self.set_unique_id(in_node)
                    in_node_gid = self.set_graph_id(in_node)

                    if in_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(in_node):
                            is_quantum = (('qubit' in in_node_gid) or ('ircuit' in in_node_gid))
                            network.add_node(in_node_uid, 
                                             label=in_node_gid, 
                                             shape='triangle',
                                             color='#cb30f2' if is_quantum else '#dd4b39',
                                             )
                        else:
                            is_quantum = (in_node_gid == 'QuantumCircuit')
                            network.add_node(in_node_uid, 
                                             label=in_node_gid,
                                             color='#ff33f8' if is_quantum else '#29cbf0',
                                             )
                    #check for duplicate edge
                    if {'from': in_node_uid, 'to': node_uid, 'arrows': 'to'} not in network.get_edges():
                        network.add_edge(in_node_uid, 
                                         node_uid,
                                         )

        #inlcude nodes and edges from dag_outputs in visualization
        if see_outputs:
            for node, outputs in self.dag_outputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        is_quantum = (('qubit' in node_gid) or ('ircuit' in node_gid))
                        network.add_node(node_uid, 
                                         label=node_gid, 
                                         shape='triangle',
                                         color='#cb30f2' if is_quantum else '#dd4b39',
                                         )
                    else:
                        is_quantum = (node_gid == 'QuantumCircuit')
                        network.add_node(node_uid, 
                                         label=node_gid,
                                         color='#ff33f8' if is_quantum else '#29cbf0',
                                         )

                #add all output nodes and their edges
                for out_node in outputs:
                    out_node_uid = self.set_unique_id(out_node)
                    out_node_gid = self.set_graph_id(out_node)

                    if out_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(out_node):
                            is_quantum = (('qubit' in out_node_gid) or ('ircuit' in out_node_gid))
                            network.add_node(out_node_uid, 
                                             label=out_node_gid, 
                                             shape='triangle',
                                             color='#cb30f2' if is_quantum else '#dd4b39',
                                             )
                        else:
                            is_quantum = (out_node_gid == 'QuantumCircuit')
                            network.add_node(out_node_uid, 
                                             label=out_node_gid,
                                             color='#ff33f8' if is_quantum else '#29cbf0',
                                             )
                    #check for duplicate edge
                    if {'from': node_uid, 'to': out_node_uid, 'arrows': 'to'} not in network.get_edges():
                        network.add_edge(node_uid, out_node_uid)

        network.toggle_physics(True)
        #network.show('mygraph.html', notebook=False)

        return network

    def create_dag_data(self, 
                        data_nodes: bool = True, 
                        see_inputs: bool = True, 
                        see_outputs: bool  = True,
                        ) -> Network:
        """TODO: Docstring """
        network = Network(directed=True)
        #build dag_inputs and dag_outputs dicts
        self.visit(self.tree)

        __data_nodes = set()
        __proc_nodes = set()
        __node_labels = {}
        __input_nodes = set()
        __output_nodes = set()
        __edges = []

        #inlcude nodes and edges from dag_inputs in visualization
        if see_inputs:
            for node, inputs in self.dag_inputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        __data_nodes.add(node_uid)

                    else:
                        __proc_nodes.add(node_uid)

                    __node_labels[node_uid] = node_gid

                #add all input nodes and their edges
                for in_node in inputs:
                    in_node_uid = self.set_unique_id(in_node)
                    in_node_gid = self.set_graph_id(in_node)

                    if in_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(in_node):
                            __data_nodes.add(in_node_uid)
                            __input_nodes.add(in_node_uid)
                        else:
                            __proc_nodes.add(in_node_uid)
                        __node_labels[in_node_uid] = in_node_gid

                    #check for duplicate edge
                    if {'from': in_node_uid, 'to': node_uid, 'arrows': 'to'} not in network.get_edges():
                        __edges.append((in_node_uid, node_uid))
                        #network.add_edge(in_node_uid, 
                        #                 node_uid,
                        #                 )

        #inlcude nodes and edges from dag_outputs in visualization
        if see_outputs:
            for node, outputs in self.dag_outputs.items():
                #set node's network ID
                node_uid = self.set_unique_id(node)
                node_gid = self.set_graph_id(node)

                #add node to network
                if node_uid not in network.get_nodes():
                    if "DATA NODE" in self.get_outputs(node):
                        is_quantum = (('qubit' in node_gid) or ('ircuit' in node_gid))
                        __data_nodes.add(node_uid)
                        __node_labels[node_uid] = node_gid
                    else:
                        is_quantum = (node_gid == 'QuantumCircuit')
                        __proc_nodes.add(node_uid)
                        __node_labels[node_uid] = node_gid

                #add all output nodes and their edges
                for out_node in outputs:
                    out_node_uid = self.set_unique_id(out_node)
                    out_node_gid = self.set_graph_id(out_node)

                    if out_node_uid not in network.get_nodes():
                        if "DATA NODE" in self.get_outputs(out_node):
                            __data_nodes.add(out_node_uid)
                        else:
                            __proc_nodes.add(out_node_uid)
                        __node_labels[out_node_uid] = out_node_gid
                    #check for duplicate edge
                    if {'from': node_uid, 'to': out_node_uid, 'arrows': 'to'} not in network.get_edges():
                        __edges.append((node_uid, out_node_uid))
                        #network.add_edge(node_uid, out_node_uid)

        return __data_nodes, __proc_nodes, __edges, __input_nodes, __output_nodes, __node_labels
