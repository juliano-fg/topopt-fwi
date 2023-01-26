#!/usr/bin/env python
import numpy as np
class model_data:
    """
    Parameters to configure the forward problem and inversion.
    These parameters are the same for all examples.
    """
    def __init__(self):
        # Domain dimensions in km ------------------------------------------ #
        self.length = 10.0
        self.depth = 3.0
        self.layer = 1.0
        # FE Mesh parameters ----------------------------------------------- #
        self.mesh_type = 'right/left'
        self.nx = 240
        self.nz = 100
        # Frequencies ------------------------------------------------------ #
        freq_spacing = 0.125
        freq_lowest = 2.5
        freq_highest = 3.5
        freq_total = int((freq_highest-freq_lowest)/freq_spacing)+1
        self.freq = np.linspace(freq_lowest, freq_highest, freq_total)
        # Receivers -------------------------------------------------------- #
        recs_spacing = 0.1
        recs_first = 0.0
        recs_last = 10.0
        recs_total = int((recs_last-recs_first)/recs_spacing)+1
        self.recs = np.linspace(recs_first, recs_last, recs_total)
        self.recs_depth = 0.05
        # Sources ---------------------------------------------------------- #
        srcs_spacing = 0.5
        srcs_first = 0.0
        srcs_last = 10.0
        srcs_total = int((srcs_last-srcs_first)/srcs_spacing)+1
        self.srcs = np.linspace(srcs_first, srcs_last, srcs_total)
        self.srcs_depth = 0.0
        self.srcs_frequency = 15.0
        # Optimization  ---------------------------------------------------- #
        self.velocity_min = 1.5
        self.velocity_max = 4.5
        self.velocity_salt = 4.5
        self.ftol = 0
        self.gtol = 0
        self.maxiter_salt = 25
        self.maxiter_back = 5
        self.beta_update = [1,1,2,2,4,4,8,8]
        self.freq_update = [3,3,5,5,7,7,9,9]
        # ------------------------------------------------------------------ #
        self.damping = 5
        # ------------------------------------------------------------------ #