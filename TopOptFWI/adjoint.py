#!/usr/bin/env python
from fenics import *
from fenics_adjoint import *
from .femsolver import FiniteElementSolver
import numpy as np
import multiprocessing as multiprocessing
queue = multiprocessing.Queue()

def fprob(prep,model,obs_data,outer_iter,back,item):
    """
    Calculate a single gradient component.
    """
    f=FiniteElementSolver(prep=prep,item=item,model=model)
    f.solve_gradient(obs_data=obs_data,outer_iter=outer_iter,back=back)
    if prep.input_data.traditional_fwi == True:
        queue.put(np.array([f.J, f.dJ_dm.vector().get_local()], dtype=object))
    elif prep.input_data.traditional_fwi == False:
        if back == True:
            queue.put(np.array([f.J, f.dJ_db.vector().get_local()], dtype=object))
        else:
            queue.put(np.array([f.J, f.dJ_da.vector().get_local()], dtype=object))
        
def adjointcalc(prep,model,obs_data,outer_iter,back):
    """
    Calculate the gradient components (for all sources and frequencies).
    """
    output = []
    p = []
    ii, iii, iiii = 0, 0, 0
    for i in range(np.math.floor(len(prep.freq_m)/prep.input_data.n_processes)):
        for j in range(prep.input_data.n_processes):
            p.append(multiprocessing.Process(target=fprob, args=(prep,model,obs_data,outer_iter,back,ii)))
            ii += 1
        for jj in range(prep.input_data.n_processes):
            p[iii].start() 
            iii += 1
        for jjj in range(prep.input_data.n_processes):
            output.append(queue.get())
            iiii += 1
    for j in range(len(prep.freq_m) - prep.input_data.n_processes*np.math.floor(len(prep.freq_m)/prep.input_data.n_processes)):
        p.append(multiprocessing.Process(target=fprob, args=(prep,model,obs_data,outer_iter,back,ii)))
        ii += 1
    for jj in range(len(prep.freq_m) - prep.input_data.n_processes*np.math.floor(len(prep.freq_m)/prep.input_data.n_processes)):
        p[iii].start() 
        iii += 1
    for jjj in range(len(prep.freq_m) - prep.input_data.n_processes*np.math.floor(len(prep.freq_m)/prep.input_data.n_processes)):
        output.append(queue.get())
        iiii += 1
    return output