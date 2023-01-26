#!/usr/bin/env python
from fenics import *
from fenics_adjoint import *
from .femsolver import FiniteElementSolver
import numpy as np
import multiprocessing as multiprocessing
queue = multiprocessing.Queue()
import sys

def fprob(prep,item):
    """
    Generate the observed data (for a single source and frequency).
    """
    f=FiniteElementSolver(prep=prep,item=item)
    f.solve_forward()
    queue.put(np.array([item, f.p.vector().get_local()], dtype=object))

def ForwardProblem(prep):
    """
    Generate the observed data (for all sources and frequencies).
    """
    output = []
    p = []
    ii, iii, iiii = 0, 0, 0
    print(' ') 
    print('\033[1m'+'\33[33m'+'    \U0001F4BB Solving Forward Problem...   '+'\033[0m')
    sys.stdout.write("\r" + '    '+ prep.animation[0])
    for i in range(np.math.floor(len(prep.freq)/prep.input_data.n_processes)):
        for j in range(prep.input_data.n_processes):
            p.append(multiprocessing.Process(target=fprob, args=(prep,ii)))
            ii += 1
        for jj in range(prep.input_data.n_processes):
            p[iii].start() 
            iii += 1
        for jjj in range(prep.input_data.n_processes):
            output.append(queue.get())
            iiii += 1
            print_bar(iiii,len(prep.freq),prep.animation)
    for j in range(len(prep.freq) - prep.input_data.n_processes*np.math.floor(len(prep.freq)/prep.input_data.n_processes)):
        p.append(multiprocessing.Process(target=fprob, args=(prep,ii)))
        ii += 1
    for jj in range(len(prep.freq) - prep.input_data.n_processes*np.math.floor(len(prep.freq)/prep.input_data.n_processes)):
        p[iii].start() 
        iii += 1
    for jjj in range(len(prep.freq) - prep.input_data.n_processes*np.math.floor(len(prep.freq)/prep.input_data.n_processes)):
        output.append(queue.get())
        iiii += 1
        print_bar(iiii,len(prep.freq),prep.animation)
    sys.stdout.flush()
    sys.stdout.write("\r" +'    '+ prep.animation[-1])
    print(' ')
    obs_data = [0]*len(output)
    for i in range(len(output)):
        obs_data[output[i][0]]=[output[i][0],output[i][1]]
    return obs_data

def print_bar(i,n,animation):
    """
    Print the progress bar for the forward problem. 
    """
    for j in range(9):
        if i == int((j+1)*n/(10)):
            sys.stdout.flush()
            sys.stdout.write("\r" + '    '+ animation[j+1])