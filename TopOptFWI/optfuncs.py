#!/usr/bin/env python3

from fenics import *
from fenics_adjoint import *
from .adjoint import adjointcalc
import numpy as np
from scipy import optimize
from .nputils import update_mscale
import sys

def Inversion(pp,od,ai,bi,mi,opts,bds):
    """
    Update the model (a, b or m) using the L-BFGS-B method from Scipy.

    Parameters
    ----------
    pp : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    od : list 
        List containing data for inversion (for all sources and frequencies).
    ai : ndarray 
        Initial guess for the variables a.
    bi : ndarray 
        Initial guess for the variables b.
    mi : ndarray 
        Initial guess for the variables m.
    opts : list 
        Dictionaries of solver options for the L-BFGS-B method.
    bds : list 
        Bounds on variables for the L-BFGS-B method.
        
    Returns
    ----------
    ai : ndarray 
        Updated model for the variables a.
    bi : ndarray 
        Updated model for the variables b.
    mi : ndarray 
        Updated model for the variables m.
    outer_iter : int 
        Outer iteration after the last update.
    """
    for outer_iter in range(len(pp.model_data.beta_update)):    
        pp = update_mscale(pp,outer_iter)
        if pp.input_data.traditional_fwi==True:
            res = optimize.minimize(fun_m,mi,
                                    args=(pp,od,[],outer_iter),
                                    method='L-BFGS-B',jac=True,
                                    bounds=bds[2],options=opts[0])
            mi = res.x
        else:
            res = optimize.minimize(fun_a,ai,
                                    args=(pp,bi,od,[],outer_iter),
                                    method='L-BFGS-B',jac=True,
                                    bounds=bds[0],options=opts[0])
            ai = res.x
            if pp.input_data.update_background==True and ((outer_iter+1)%2)==0:
                res = optimize.minimize(fun_b,bi,
                                        args=(pp,ai,od,[],outer_iter),
                                        method='L-BFGS-B',jac=True,
                                        bounds=bds[1],options=opts[1])
                bi = res.x
    return ai,bi,mi,outer_iter

def fun_m(mi,pp,od,fl,oi):
    """
    Callable Function sent to optimizer to update variables m.

    Parameters
    ----------
    mi : ndarray 
        Initial guess for the variables m.
    pp : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    od : list 
        List containing data for inversion (for all sources and frequencies).
    fl : list 
        List containing cost functional values.
    oi : int 
        Iteration number of the outer optimization loop.
        
    Returns
    ----------
    func : float 
        Cost functional value for the current iteration.
    dJ_dm : ndarray 
        Gradient vector for the current iteration.
    """
    m = Function(pp.Vreal)
    dJ_dm = np.zeros(int(pp.nnodes))
    m.vector()[:] = mi
    output = adjointcalc(pp,[m],od,oi,False)
    func = 0.0
    for i in range(len(output)):
        func = func + output[i][0]
        dJ_dm = dJ_dm + output[i][1][:]
    if len(fl)==0:
        fl.append(func)
        if (oi)%2==0:
            print('\033[1m'+'\33[33m'+'    \U0001F4C9 Running the L-BFGS-B code...   '+'\033[0m')
            print('\33[33m' + '    Updating model m'  + '\033[0m')
            sys.stdout.write("\r" + '    '+ pp.animation[0])
    else:
        if func<fl[-1]:
            fl.append(func)
            if len(fl)>=pp.model_data.maxiter_salt and len(fl)==pp.model_data.maxiter_salt:
                if (oi)%2==0:
                    sys.stdout.flush()
                    sys.stdout.write("\r" +'    '+ pp.animation[5])
                else:
                    sys.stdout.flush()
                    sys.stdout.write("\r" +'    '+ pp.animation[-1])
                    print(' ')
            else:
                print_bar2(len(fl),pp.model_data.maxiter_salt,pp.animation,oi)
    return (func, dJ_dm)

def fun_a(ai,pp,bi,od,fl,oi):
    """
    Callable Function sent to optimizer to update variables a.

    Parameters
    ----------
    ai : ndarray 
        Initial guess for the variables a.
    pp : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    bi : ndarray 
        Current variables b.
    od : list 
        List containing data for inversion (for all sources and frequencies).
    fl : list 
        List containing cost functional values.
    oi : int 
        Iteration number of the outer optimization loop.
        
    Returns
    ----------
    func : float 
        Cost functional value for the current iteration.
    dJ_da : ndarray 
        Gradient vector for the current iteration.
    """
    a = Function(pp.Vreal)
    b = Function(pp.Vreal)
    dJ_da = np.zeros(int(pp.nnodes))
    a.vector()[:] = ai
    b.vector()[:] = bi
    output = adjointcalc(pp,[a,b],od,oi,False)
    func = 0.0
    for i in range(len(output)):
        func = func + output[i][0]
        dJ_da = dJ_da + output[i][1][:]
    if len(fl)==0:
        fl.append(func)
        if (oi)%2==0:
            print('\033[1m'+'\33[33m'+'    \U0001F4C9 Running the L-BFGS-B code...   '+'\033[0m')
            print('\33[33m' + '    Updating model a'  + '\033[0m')
            sys.stdout.write("\r" + '    '+ pp.animation[0])
    else:
        if func<fl[-1]:
            fl.append(func)
            if len(fl)>=pp.model_data.maxiter_salt and len(fl)==pp.model_data.maxiter_salt:
                if (oi)%2==0:
                    sys.stdout.flush()
                    sys.stdout.write("\r" +'    '+ pp.animation[5])
                else:
                    sys.stdout.flush()
                    sys.stdout.write("\r" +'    '+ pp.animation[-1])
                    print(' ')
            else:
                print_bar2(len(fl),pp.model_data.maxiter_salt,pp.animation,oi)
    return (func, dJ_da)

def fun_b(bi,pp,ai,od,fl,oi):
    """
    Callable Function sent to optimizer to update variables b.

    Parameters
    ----------
    bi : ndarray 
        Initial guess for the variables b.
    pp : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    ai : ndarray 
        Current variables a.
    od : list 
        List containing data for inversion (for all sources and frequencies).
    fl : list 
        List containing cost functional values.
    oi : int 
        Iteration number of the outer optimization loop.
        
    Returns
    ----------
    func : float 
        Cost functional value for the current iteration.
    dJ_db : ndarray 
        Gradient vector for the current iteration.
    """
    a = Function(pp.Vreal)
    b = Function(pp.Vreal)
    dJ_db = np.zeros(int(pp.nnodes))
    a.vector()[:] = ai
    b.vector()[:] = bi
    output = adjointcalc(pp,[a,b],od,oi,True)
    func = 0.0
    for i in range(len(output)):
        func = func + output[i][0]
        dJ_db = dJ_db + output[i][1][:]
    if len(fl)==0:
        fl.append(func)
        print('\33[33m' + '    Updating model b'  + '\033[0m')
        sys.stdout.write("\r" + '    '+ pp.animation1[0])
    else:
        if func<fl[-1]:
            fl.append(func)
            if len(fl)>=pp.model_data.maxiter_back and len(fl)==pp.model_data.maxiter_back:
                sys.stdout.flush()
                sys.stdout.write("\r" +'    '+ pp.animation1[-1])
                print(' ')
            else:
                print_bar(len(fl),pp.model_data.maxiter_back,pp.animation1)
    return (func, dJ_db)

def fun_eval(var,pp,od):
    """
    Evaluates the Cost Functional.

    Parameters
    ----------
    var : list  
        List containing the current variables ([mi] or [ai,bi])
    pp : core.preprocessing.prep1
        Object containing the preprocessing data for the FE problem.
    od : list 
        List containing data for inversion (for all sources and frequencies).
        
    Returns
    ----------
    func : float 
        Cost functional value for the current variables.
    """
    if pp.input_data.traditional_fwi==True:
        mod = [var[0]]
    else:
        mod = [var[0],var[1]]  
    output = adjointcalc(pp,mod,od,len(pp.model_data.beta_update)-1,False)
    func = 0.0
    for i in range(len(output)):
        func = func + output[i][0]
    return func


def print_bar2(i, n, animation, oi):
    """
    Print the progress bar for the inverse problem.
    """
    for j in range(4):
        if i == int((j+1)*n/(5)):
            if (oi)%2==0:
                sys.stdout.flush()
                sys.stdout.write("\r" + '    ' + animation[j+1])
            else:
                sys.stdout.flush()
                sys.stdout.write("\r" + '    ' + animation[j+6])

def print_bar(i,n,animation):
    """
    Print the progress bar for the inverse problem.
    """
    for j in range(5):
        if i == int((j+1)*n/(6)):
            sys.stdout.flush()
            sys.stdout.write("\r" + '    ' + animation[j])