#!/usr/bin/env python
# --------------------------------------------------------------------------- #
# -                           Importing Modules                             - #
# --------------------------------------------------------------------------- #
import TopOptFWI 
from input_data import input_data
# --------------------------------------------------------------------------- #
# -       Preprocessing (create Mesh, Spaces, Absorbing layer, etc)         - #
# --------------------------------------------------------------------------- #
prep = TopOptFWI.PreProcessing(TopOptFWI.model_data(),input_data())
prep.initialize_model()
# --------------------------------------------------------------------------- #
# -                   Solving the Forward Problem                           - #
# --------------------------------------------------------------------------- #
obs_data = TopOptFWI.ForwardProblem(prep)
# --------------------------------------------------------------------------- #
# -              Initializing Optimization Variables                        - #
# --------------------------------------------------------------------------- #
ai,bi,mi,opts,bds = TopOptFWI.InitializeOpt(prep)
# --------------------------------------------------------------------------- #
# -                 Solving the Inverse Problem                             - #
# --------------------------------------------------------------------------- #
ai,bi,mi,outer_iter = TopOptFWI.Inversion(prep,obs_data,ai,bi,mi,opts,bds)
# --------------------------------------------------------------------------- #
# -            Saving Solutions and Evaluating Errors                       - #
# --------------------------------------------------------------------------- #
TopOptFWI.PlotVelocity([ai,bi,mi],prep,outer_iter)
TopOptFWI.EvalErrors([ai,bi,mi],prep,obs_data)
# --------------------------------------------------------------------------- #
# -                               End                                       - #
# --------------------------------------------------------------------------- #