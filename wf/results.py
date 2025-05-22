import matplotlib.pyplot as plt
import pandas as pd

class WorkflowSimulationResults:
    """Lightweight exposure of a simulator's results.
    This class wraps a simulator that has completed execution 
    of a resource simulation, and exposes various elements and 
    metrics of the results. Note that this results object is 
    designed to be light-weight. Much more complex results can 
    be extracted and generated using the 'qrew.results.Results' 
    object.

    Note that this class is not designed to support any actual 
    time-dependent state result data. That information is 
    provided separately in the 'WorkflowState' object itself,
    which is returned by the simulator 'run()' method as well.
    """
    
    def __init__(self, success=False):
        self.success = success

    