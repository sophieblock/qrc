# -*- coding: utf-8 -*-
"""Name.

This file contains 

**BOEING PROPRIETARY**

Authors:
    Joel Thompson (richard.j.thompson3@boeing.com)

"""

import argparse
import importlib
import os
import sys

from workflow.components import Resource, ResourceBroker
from workflow.examples.rootfinding import get_root_finding_project
from workflow.simulation.engine import WorkflowSimulator
from workflow.simulation.models.resources import ResourceModel, ResourceBrokerModel

from workflow import get_logger

logger = get_logger(__name__)

SUCCESS_CODE = 0


def build_parser():
    """Builds and returns a command-line argument parser.
    
    Returns:
        `ArgumentParser`: The command-line argument parser.
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    return parser


def simulate_root_finding_example():
    
    # project = root_finding_example()
    # domain = project.domains["Root-finding"]
    # workflow = Workflow("Newton Method Workflow", [domain.processes["Newton Method"]])
    # simulator = WorkflowSimulator(workflow)
    # simulator.run()    
    
    project = get_root_finding_project()
    domain = project.domains["Root-finding"]
    workflow = domain.workflows["Bisection method workflow"]
    
    r1 = Resource("Boeing Laptop", {"ram": 32 * 1.e9}, model=ResourceModel())
    
    resources = [r1]
    
    broker = ResourceBroker("All resources", resources, ResourceBrokerModel())
    
    workflow.set_broker(broker)
    
    simulator = WorkflowSimulator(workflow)
    print("Starting simulation...")
    simulator.run()
    print("Simulation completed")

    
def run_python_file(path):
    """
    Executes a Python script file containing Workflow Tool instructions.
    
    Args:
        path (str): The (existing) path to the Python file.
        
    Returns:
        int: Status code (zero upon success; an error code upon failure).
    
    """
    logger.info(f"Running Python script file: {path}")

    # Check that the file exists.

    if not os.path.exists(path):
        raise ValueError(f"No such existing Python script file: {path}")
    
    # Convert the path into a Python module format to import it.
    
    path = path.replace(".py", "").replace(".\\", "").replace("./", "").replace("/", ".").replace("\\", ".")
    
    # Attempt to import the Python script. This will execute all of its commands.
    
    importlib.import_module(path)
    
    # Upon finishing import, we are finished. Return an exit code.
    
    logger.info(f"Workflow tool successfully executed; nominal termination reached.")
    
    # Return a zero status code to indicate success of the operation.
    
    return SUCCESS_CODE

    
def run(args):
    if hasattr(args, "file") and args.file:
        result = run_python_file(args.file)
    sys.exit(result)


def main():
    # parser = build_parser()
    # args = parser.parse_args()
    # run(args)
    simulate_root_finding_example()
    
# --- Main program execution ---

    
if __name__ == "__main__":
    main()
