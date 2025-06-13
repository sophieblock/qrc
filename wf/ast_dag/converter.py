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
     # ...
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
