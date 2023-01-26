#!/usr/bin/env python

from fenics import *
from fenics_adjoint import *
from ufl import tanh
import matplotlib
import matplotlib.tri as tri 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import rc
import matplotlib.ticker as ticker
from .nputils import find_dofs
from .optfuncs import fun_eval

# Configure font type and size for all plots:
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams.update({'font.size': int(12)})

def PlotVelocity(var,prep,outer_iter):
    """
    Plot the velocity model in km/s.
    This figure is saved in the result folder.
    """
    if prep.input_data.traditional_fwi==True:
        m = Function(prep.Vreal)
        m.vector()[:] = var[2]
        c = project(1/sqrt(abs(m)),prep.Vreal)
    else:
        a = Function(prep.Vreal)
        a.vector()[:] = var[0]
        b = Function(prep.Vreal)
        b.vector()[:] = var[1]
        plot_salt(a,prep,outer_iter)
        a_tilde = helmholtz_filter(a, prep.weight_a, prep.Vreal)
        beta = prep.model_data.beta_update[outer_iter]
        if prep.input_data.projection == True:
            a_bar = (tanh(beta*0.5)+tanh(beta*(a_tilde-0.5)))/(tanh(beta*0.5)+tanh(beta*(1.0-0.5)))
        else:
            a_bar = a_tilde
        m = (prep.input_data.estimated_salt_vel**-2)*(a_bar**prep.input_data.q) + b*(1-a_bar**prep.input_data.q)
        c = project(1/sqrt(abs(m)),prep.Vreal)
    plot_velocity(c,prep,outer_iter,True)

def EvalErrors(var,prep,obs_data):
    """
    Evaluate both reconstruction and fitting errors.
    These values are printed in the terminal and saved in a file inside the result folder.
    """
    print(' ')
    print('\033[1m'+'\33[33m'+'    \U0001F50E Evaluating Errors  '+'\033[0m')
    if prep.input_data.traditional_fwi==True:
        m = Function(prep.Vreal)
        m.vector()[:] = var[2]
        c = project(1/sqrt(abs(m)),prep.Vreal)
        c_0 = project(1/sqrt(abs(prep.m0)),prep.Vreal)  
        J_0 = fun_eval([prep.m0],prep,obs_data)
        J = fun_eval([m],prep,obs_data)
    else:
        a = Function(prep.Vreal)
        a.vector()[:] = var[0]
        b = Function(prep.Vreal)
        b.vector()[:] = var[1]
        a_tilde = helmholtz_filter(a, prep.weight_a, prep.Vreal)
        beta = prep.model_data.beta_update[-1]
        if prep.input_data.projection == True:
            a_bar = (tanh(beta*0.5)+tanh(beta*(a_tilde-0.5)))/(tanh(beta*0.5)+tanh(beta*(1.0-0.5)))
        else:
            a_bar = a_tilde
        m = (prep.input_data.estimated_salt_vel**-2)*(a_bar**prep.input_data.q) + b*(1-a_bar**prep.input_data.q)
        c = project(1/sqrt(abs(m)),prep.Vreal)
        a_tilde_0 = helmholtz_filter(prep.a0, prep.weight_a, prep.Vreal)
        beta_0 = prep.model_data.beta_update[0]
        if prep.input_data.projection == True:
            a_bar_0 = (tanh(beta_0*0.5)+tanh(beta_0*(a_tilde_0-0.5)))/(tanh(beta_0*0.5)+tanh(beta_0*(1.0-0.5)))
        else:
            a_bar_0 = a_tilde_0
        m_0 = (prep.input_data.estimated_salt_vel**-2)*(a_bar_0**prep.input_data.q) + prep.b0*(1-a_bar_0**prep.input_data.q)
        c_0 = project(1/sqrt(abs(m_0)),prep.Vreal)
        J_0 = fun_eval([prep.a0,prep.b0],prep,obs_data)
        J = fun_eval([a,b],prep,obs_data)
    node = find_dofs(prep.Vreal,0.0,prep.model_data.length,-prep.model_data.depth,0.0)
    dif_a = c.vector()[node]-prep.truemodel.vector()[node]
    dif_b = c_0.vector()[node]-prep.truemodel.vector()[node]
    eps_c = np.sqrt(np.sum(np.power((dif_a),2)))/np.sqrt(np.sum(np.power((dif_b),2)))
    print('\33[33m' + '    Reconstruction Error: \u03B5_c = {0:.6E}'.format(eps_c)  + '\033[0m')
    eps_j = J/J_0
    print('\33[33m' + '    Normalized Fitting Error: \u03B5_J = {0:.6E}'.format(eps_j)  + '\033[0m')
    print("__________________________________________________________________")
    with open(prep.cb_folder+"/errors.txt", "w") as text_file:
        print("Reconstruction Error: {0:.6E}".format(eps_c), file=text_file)
        print("Normalized Fitting Error: {0:.6E}".format(eps_j), file=text_file)
    plot_overline(c,prep,4.0)
    plot_overline(c,prep,6.0)
    return eps_c,eps_j
        
def major_formatter(x, pos):
    """
    Invert axis.
    """
    label = str(-x) if x < 0 else str(x)
    return label

def mesh2triang(mesh):
    """
    Create triangular grid using a Delaunay triangulation.
    """
    xy = mesh.coordinates()
    ret = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
    return ret

def plot_fenics(obj):
    """
    Plot a FEniCS object.
    """
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().get_local()
            plt.tripcolor(mesh2triang(mesh), C,cmap=cm.get_cmap('jet', 64))
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud',
                          cmap=cm.get_cmap('jet', 64))
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')
        
def plot_salt(a,prep,outer_iter):
    """
    Plot the control variable 'a'.
    """
    fig = plot_fenics(a)
    ax = plt.gca()
    plt.xlabel('Distance [km]')
    plt.ylabel('Depth [km]')
    plt.xlim(0,prep.model_data.length)
    plt.ylim(-prep.model_data.depth,0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(int(5)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(int(3)))
    ax.yaxis.set_major_formatter(major_formatter)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cbar = plt.colorbar(fig, cax=cax)
    cbar.set_label('Opt. Variable')
    plt.clim(0.0,1.0)
    if outer_iter == 1001:
        plt.savefig(prep.cb_folder+'/a_initial'+'.png',format='png',dpi=600,transparent=False)
    else:
        plt.savefig(prep.cb_folder+'/a_recon'+'.png',format='png',dpi=600,transparent=False)
    plt.clf()  
    
def plot_velocity(c,prep,outer_iter,interface):
    """
    Plot the velocity 'c'.
    """
    fig = plot_fenics(c)
    if interface == True:
        path = 'models/example_'+prep.input_data.example.capitalize()+'.txt'
        f = open(path)
        with f:
            nrow,ncol = [int(float(x)) for x in next(f).split(',')]
            minx,maxx,minz,maxz = [float(x) for x in next(f).split(',')]
            x = np.linspace(-prep.model_data.layer, prep.model_data.length+prep.model_data.layer, ncol)
            z = np.linspace(prep.model_data.layer, -prep.model_data.depth-prep.model_data.layer, nrow)
            Z = np.zeros([nrow,ncol])
            k=0
            for line in f:
                Z[k,:]=([float(x) for x in line.split(',')])
                k+=1
        x_salt = []
        z_salt = []
        for j in range(len(z)):
            for i in range(len(x)):
                if Z[j,i]>2.0:
                    if i>0 and i<len(x)-1 and j>0 and j<len(z)-1:
                        if Z[j,i+1]<2.0 or Z[j,i-1]<2.0 or Z[j+1,i]<2.0 or Z[j-1,i]<2.0:
                            x_salt.append(x[i])
                            z_salt.append(z[j])
        plt.plot(x_salt[:],z_salt[:],'o',color='black',markersize=0.05,linewidth=1,markerfacecolor='black',markeredgecolor=None,markeredgewidth=0.05)
    if outer_iter == 1000:
        for r in range(len(prep.model_data.recs)):
            plt.plot(prep.model_data.recs[r],-prep.model_data.recs_depth,'v',color='black',markersize=3,linewidth=0.5,markerfacecolor='green',markeredgecolor='black',markeredgewidth=0.5,clip_on=False,zorder=100)
        for s in range(len(prep.model_data.srcs)):
            plt.plot(prep.model_data.srcs[s],-prep.model_data.srcs_depth,'o',color='black',markersize=3,linewidth=0.5,markerfacecolor='yellow',markeredgecolor='black',markeredgewidth=0.5,clip_on=False,zorder=101)
    ax = plt.gca()
    plt.xlabel('Distance [km]')
    plt.ylabel('Depth [km]')
    plt.xlim(0,prep.model_data.length)
    plt.ylim(-prep.model_data.depth,0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(int(5)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(int(3)))
    ax.yaxis.set_major_formatter(major_formatter)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cbar = plt.colorbar(fig, cax=cax)
    cbar.set_label('Velocity [km/s]')
    vel_max = max([prep.input_data.estimated_salt_vel,prep.input_data.true_salt_vel])
    if vel_max == prep.model_data.velocity_salt:
        plt.clim(prep.model_data.velocity_min,prep.model_data.velocity_max)
    else:
        plt.clim(prep.model_data.velocity_min,round(vel_max+0.5))
    if outer_iter == 1000:
        plt.savefig(prep.cb_folder+'/c_true'+'.png',format='png',dpi=600,transparent=False)
    elif outer_iter == 1001:
        plt.savefig(prep.cb_folder+'/c_initial'+'.png',format='png',dpi=600,transparent=False)
    else:
        plt.savefig(prep.cb_folder+'/c_recon'+'.png',format='png',dpi=600,transparent=False)
    plt.clf()   

def plot_overline(c,prep,dist):
    """
    Plot the velocity 'c' along a vertical line.
    """
    tol = 0.2*prep.spacing_x
    node = find_dofs(prep.Vreal,dist-tol,dist+tol,-prep.model_data.depth,0.0)
    true_line = prep.truemodel.vector()[node]
    recon_line = c.vector()[node]
    z = np.linspace(0.0,-prep.model_data.depth, len(true_line))
    z = np.linspace(-prep.model_data.depth,0.0, len(true_line))
    plt.plot( true_line, z, '--r',label='True')
    plt.plot( recon_line, z, '-b',label='Recon')
    plt.gca().yaxis.set_major_formatter(major_formatter)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Depth [km]')
    plt.legend(loc='lower left')
    plt.savefig(prep.cb_folder+'/line_'+str(int(dist))+'km.png',format='png',dpi=600,transparent=False)
    plt.clf() 
    
def helmholtz_filter(a, w, V):
    """
    Helmholtz-type spatial filter
    """
    if w == 0.0:
        a_f = a
    else:
        a_f = TrialFunction(V)
        vH = TestFunction(V)
        F = (inner(w*w*grad(a_f), grad(vH))*dx + inner(a_f, vH)*dx - inner(a, vH)*dx)
        aH = lhs(F)
        LH = rhs(F)
        a_f = Function(V)
        solve(aH == LH, a_f)
    return a_f
