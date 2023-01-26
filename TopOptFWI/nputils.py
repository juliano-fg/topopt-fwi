#!/usr/bin/env python3

from fenics import *
from fenics_adjoint import *
import numpy as np

def find_dofs(V,x1,x2,z1,z2):
    """
    Finds the DOF's within a rectangle area from a given FunctionSpace.

    Parameters
    ----------
    V : dolfin.function.functionspace.FunctionSpace
        Object that represents a finite element function space.
    x1 : float 
        Distance coordinate x1 (minimum).
    x2 : float 
        Distance coordinate x2 (maximum).
    z1 : float
        Depth coordinate z1 (minimum).
    z2 : float 
        Depth coordinate z2 (maximum).

    Returns
    ----------
    list
        List with the DOF's within the rectangle.
    """
    coordinates = V.tabulate_dof_coordinates()
    dofs = []
    for dof in range(len(coordinates)):
        if coordinates[dof][0] >= x1 and  coordinates[dof][0] <= x2:
            if coordinates[dof][1] >= z1 and  coordinates[dof][1] <= z2:
                dofs.append(dof)
    return dofs

def InitializeOpt(prep):
    """
    Initializes variables and parameters for the optimization problem.

    Parameters
    ----------
    prep : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.

    Returns
    ----------
    ai : ndarray 
        Initial variables for model a.
    bi : ndarray 
        Initial variables for model b.
    mi : ndarray 
        Initial variables for model m.
    opts : list 
        Dictionaries of solver options for the L-BFGS-B method.
    bds : list 
        Bounds on variables for the L-BFGS-B method.
    """
    if prep.input_data.traditional_fwi == False:
        ai,bi,mi = prep.a0.vector()[:],prep.b0.vector()[:],[0]
    else:
        ai,bi,mi = [0],[0],prep.m0.vector()[:]
    amin,amax = 0.0,1.0
    velocity_max = max([prep.model_data.velocity_max,prep.input_data.estimated_salt_vel])
    mmin,mmax = velocity_max**(-2),prep.model_data.velocity_min**(-2)
    opts = [{"disp": False,"maxcor": 10,"maxls": 10,
              "ftol": prep.model_data.ftol,"gtol": prep.model_data.gtol,
              "maxiter": prep.model_data.maxiter_salt },
            {"disp": False,"maxcor": 10,"maxls": 10,
            "ftol": prep.model_data.ftol,"gtol": prep.model_data.gtol,
            "maxiter": prep.model_data.maxiter_back }]
    bds =[[(amin, amax) for x in range(len(ai))],
          [(mmin, mmax) for x in range(len(bi))],
          [(mmin, mmax) for x in range(len(mi))]]
    return ai,bi,mi,opts,bds

def update_mscale(prep,it):
    """
    Updates the frequency band used for inversion.

    Parameters
    ----------
    prep : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    it : int
        Iteration of the outer optimization loop

    Returns
    ----------
    prep : core.preprocessing.prep1
        Updated object containing the preprocessing data for the FE problem.

    """
    final_mscale = int(prep.model_data.freq_update[it]*len(prep.model_data.srcs))
    prep.freq_m = prep.freq[0:final_mscale]
    if (it)%2==0:
        print(' ')
        print('\033[1m'+'\33[33m'+'    \U0001F527 Multiscale Configuration...    '+'\033[0m')
        print('\33[33m' + '    Frequency Band for Inversion: [{0}-{1}] Hz'.format(prep.freq_m[0],prep.freq_m[-1])  + '\033[0m')
        print(' ')
    return prep