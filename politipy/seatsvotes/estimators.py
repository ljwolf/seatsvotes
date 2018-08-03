import numpy as np
import pandas as pd
from . import utils as ut

class EstimatorMixin:
    def swing_ratio(simulator, n_sims = 1000, 
                    hinge_share = None, bin_width = .02, 
                    **simulation_configs):
        # needs
        # observed shares & weights
        # simulate_elections, n_sims=n_sims, hinge_share, **opts
        ...
    
    def bonus(simulator, hinge_share = None, **simulation_configs):
        # needs
        # simulate_elections, n_sims=n_sims, hinge_share, **opts, let hinge_share be None, focused on empirical
        ...
    
    def attainment_gap(simulator, **simulation_configs):
        # needs
        # simulate_elections, n_sims, n_batches

        ...
    
    def efficiency_gap(simulator, resample_turnout=True):
        # needs
        # simulate_elections, n_sims, empirical turnout, handle what to do if turnout is stochastic
        ...
    
    def sensitivity(simulator, statistic):
        # needs
        # canonical form for model data, so that we can delete & refit
        ...

    def _regularize_simulations(simulator, sims):
        # needs
        # schema for data, following Preprocessor
        ...

