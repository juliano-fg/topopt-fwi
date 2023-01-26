#!/usr/bin/env python
import numpy as np 
from fenics import *
from fenics_adjoint import *
from .nputils import find_dofs
import warnings
import os
from datetime import datetime
from .plotutils import plot_velocity,plot_salt
from ufl import tanh
import multiprocessing

class PreProcessing(object):
    """
    Pre-Processing step.
    """
    def __init__(self,model_data,input_data):
        self.model_data=model_data
        self.input_data=input_data
        self.isthereMESH=False
        self.isthereSPACE=False
        self.isthereTRUEMODEL=False
        self.isthereSPONGE=False
        self.isthereCASES=False
        self.isthereINITIAL=False
        print("__________________________________________________________________")
        print('  ')
        print('\033[1m'+'\33[33m'+'               TO-based FWI for Salt Reconstruction   '+'\033[0m')
        print('    ..........................................................    ')
        print('\33[33m'+'    Authors  : Juliano F. Goncalves, Emilio C.N. Silva '+'\033[0m')
        print('\33[33m'+'    Contact  : julianofg@usp.br, ecnsilva@usp.br'+'\033[0m')
        print('\33[33m'+'    Date     : Jan 2023'+'\033[0m')
        print("__________________________________________________________________")
        print('  ')
        print('\033[1m'+'\33[33m'+'    \U0001F4CB Problem Description...   '+'\033[0m')
        print('\33[33m' + '    Example: Model {0}'.format(self.input_data.example.capitalize())  + '\033[0m')
        if self.input_data.tofwi == True:
            self.input_data.traditional_fwi = False
        else:
            self.input_data.traditional_fwi = True
        if self.input_data.traditional_fwi == True:
            print('\33[33m' + '    Approach: Traditional FWI'  + '\033[0m')
        else:
            if self.input_data.update_background == False:
                print('\33[33m' + '    Approach: TO-based FWI (with fixed background)'  + '\033[0m')
            else:
                print('\33[33m' + '    Approach: TO-based FWI'  + '\033[0m')
            print('\33[33m' + '    True salt velocity: c_s = {0}'.format(self.input_data.true_salt_vel)  + '\033[0m')
            print('\33[33m' + '    Estimated salt velocity: c_s = {0}'.format(self.input_data.estimated_salt_vel)  + '\033[0m')
            print('\33[33m' + '    Initial slope for the background: b = {0}'.format(self.input_data.initial_slope)  + '\033[0m')
            print('\33[33m' + '    SIMP penalty exponent: q = {0}'.format(self.input_data.q)  + '\033[0m')
            print('\33[33m' + '    Heaviside Projection: {0}'.format(self.input_data.projection)  + '\033[0m')
            print('\33[33m' + '    Variable filter radius: r_a = {0}'.format(self.input_data.r_a)  + '\033[0m')
        print('\33[33m' + '    Gradient filter radius: r_s = {0}'.format(self.input_data.r_s)  + '\033[0m')
        self.weight_a = self.input_data.r_a/(2.0*sqrt(3.0))
        self.weight_s = self.input_data.r_s/(2.0*sqrt(3.0))
        print('\33[33m' + '    Inverse crime: {0}'.format(self.input_data.inverse_crime)  + '\033[0m')
        print('\33[33m' + '    Number of Processes: {0}/{1}'.format(self.input_data.n_processes,multiprocessing.cpu_count())  + '\033[0m')
        print('  ')
        print('\033[1m'+'\33[33m'+'    \U0001F4CF Data Acquisition...   '+'\033[0m')
        print('\33[33m' + '    {0} Sources ({2} Hz Ricker) at depth {1} km'.format(len(self.model_data.srcs),self.model_data.srcs_depth,self.model_data.srcs_frequency)  + '\033[0m')
        print('\33[33m' + '    {0} Receivers at depth {1} km'.format(len(self.model_data.recs),self.model_data.recs_depth)  + '\033[0m')
        print('\33[33m' + '    Frequency band [{0}-{1}] Hz with spacing {2} Hz'.format(self.model_data.freq[0],self.model_data.freq[-1],self.model_data.freq[1]-self.model_data.freq[0])  + '\033[0m')
        print('  ')
        input("  Press Enter to continue...")
        print('  ')
        print('\033[1m'+'\33[33m'+'    \U0001F6E0  PreProcessing Step - Creating...   '+'\033[0m')
        isExist = os.path.exists('results')
        if not isExist:
            os.makedirs('results')
        tdy = datetime.now()
        self.cb_folder = 'results/'+tdy.strftime("%d-%b-%y_")+tdy.strftime("%H")+'h'+tdy.strftime("%M")+'m'+tdy.strftime("%S")+'s'
        os.makedirs(self.cb_folder)
        with open(self.cb_folder+"/config.txt", "w") as text_file:
            print("Date: "+tdy.strftime("%d-%b-%y"), file=text_file)
            print("Time: "+tdy.strftime("%H:%M:%S"), file=text_file)
            print(".................................", file=text_file)
            print("Model: {0}".format(self.input_data.example.capitalize()), file=text_file)
            if self.input_data.traditional_fwi == True:
                print('Approach: Traditional FWI', file=text_file)
            else:
                print('Approach: TO-based FWI', file=text_file)
                print("Update Background: {0}".format(self.input_data.update_background), file=text_file)
                print("True salt velocity c_s: {0}".format(self.input_data.true_salt_vel), file=text_file)
                print("Estimated salt velocity c_s: {0}".format(self.input_data.estimated_salt_vel), file=text_file)
                print("SIMP penalty exponent q: {0}".format(self.input_data.q), file=text_file)
                print("Heaviside Projection: {0}".format(self.input_data.projection), file=text_file)
                print("Variable filter radius r_a: {0}".format(self.input_data.r_a), file=text_file)
            print("Gradient filter radius r_s: {0}".format(self.input_data.r_s), file=text_file)
            print("Initial slope for the background b: {0}".format(self.input_data.initial_slope), file=text_file)
            print("Inverse Crime: {0}".format(self.input_data.inverse_crime), file=text_file)
            print(".................................", file=text_file)
        self.animation = ["□ □ □ □ □ □ □ □ □ □   ","■ □ □ □ □ □ □ □ □ □   ","■ ■ □ □ □ □ □ □ □ □   ", "■ ■ ■ □ □ □ □ □ □ □   ", "■ ■ ■ ■ □ □ □ □ □ □   ", "■ ■ ■ ■ ■ □ □ □ □ □   ", "■ ■ ■ ■ ■ ■ □ □ □ □   ", "■ ■ ■ ■ ■ ■ ■ □ □ □   ", "■ ■ ■ ■ ■ ■ ■ ■ □ □   ", "■ ■ ■ ■ ■ ■ ■ ■ ■ □   ", "■ ■ ■ ■ ■ ■ ■ ■ ■ ■   "]
        self.animation1 = ["□ □ □ □ □ □ □ □ □ □   ","■ ■ □ □ □ □ □ □ □ □   ", "■ ■ ■ ■ □ □ □ □ □ □   ", "■ ■ ■ ■ ■ ■ □ □ □ □   ", "■ ■ ■ ■ ■ ■ ■ ■ □ □   ", "■ ■ ■ ■ ■ ■ ■ ■ ■ ■   "]
    def create_mesh(self):
        """
        Create the FE mesh.
        """
        self.mesh=RectangleMesh(Point(-self.model_data.layer,-self.model_data.layer-self.model_data.depth), 
                                Point(self.model_data.length+self.model_data.layer,self.model_data.layer), 
                                self.model_data.nx,self.model_data.nz,self.model_data.mesh_type)  
        if self.model_data.mesh_type=='crossed':
            self.nel=4*self.model_data.nx*self.model_data.nz
            self.nnodes=(self.model_data.nx+1)*(self.model_data.nz+1)+self.model_data.nx*self.model_data.nz
        elif self.model_data.mesh_type=='left/right' or self.model_data.mesh_type=='right/left':
            self.nel=2*self.model_data.nx*self.model_data.nz
            self.nnodes=(self.model_data.nx+1)*(self.model_data.nz+1)
        else:
            raise TypeError("Invalid mesh_type.")
        self.spacing_x=(self.model_data.length+2*self.model_data.layer)/self.model_data.nx
        self.spacing_z=(self.model_data.depth+2*self.model_data.layer)/self.model_data.nz
        if self.spacing_x != self.spacing_z:
            warnings.warn("Mesh is not regular.")
        self.isthereMESH=True
        print('\33[33m' + '    FE mesh with {0} elements and {1} nodes'.format(self.nel, self.nnodes) + '\033[0m')
    def create_spaces(self):
        """
        Create the FE spaces.
        """
        if self.isthereMESH==False:
            self.create_mesh()
        P1=FiniteElement('P','triangle',1)
        self.Vcomplex=FunctionSpace(self.mesh,P1*P1) 
        self.Vreal=FunctionSpace(self.mesh,P1)
        self.isthereSPACE=True
        print('\33[33m' + '    Trial and test spaces' + '\033[0m')
    def create_truemodel(self):
        """
        Create the True velocity model.
        """
        if self.isthereSPACE==False:
            self.create_spaces()
        path = 'models/example_'+ self.input_data.example.capitalize() +'.txt'
        f = open(path)
        with f:
            nrow,ncol = [int(float(x)) for x in next(f).split(',')]
            minx,maxx,minz,maxz = [float(x) for x in next(f).split(',')]
            x = np.linspace(minx, maxx, ncol)
            z = np.linspace(minz, maxz, nrow)
            Z = np.zeros([nrow,ncol])
            k=0
            for line in f:
                Z[k,:]=([float(x) for x in line.split(',')])
                k+=1
            Z_ = Z[int(0):,:]
            Z = np.fliplr(Z_[:, :]).flat
            res = Z[::-1]
        mesh_aux = RectangleMesh(Point(-self.model_data.layer,-self.model_data.layer-self.model_data.depth), 
                                 Point(self.model_data.length+self.model_data.layer,self.model_data.layer), 
                                 ncol-1, nrow-1-int(0), 'left/right')
        space_aux = FunctionSpace(mesh_aux, 'CG', 1)
        var_aux = Function(space_aux) 
        var_aux.vector()[vertex_to_dof_map(space_aux)] = res
        var_aux.set_allow_extrapolation(True)
        salt = project(var_aux, self.Vreal)  
        background=Expression('1.5-(x[1]+0.0)*b', degree=3,b=0.8333)
        back=interpolate(background,self.Vreal) 
        velocity=Function(self.Vreal)
        for node in range(len(salt.vector()[:])):
            if salt.vector()[node]>4.0:
                velocity.vector()[node] = self.input_data.true_salt_vel
            else:
                velocity.vector()[node] = back.vector()[node]
        self.truemodel=project(velocity, self.Vreal)
        self.isthereTRUEMODEL=True
        print('\33[33m' + '    True velocity model for example {0} with salt velocity {1} km/s'.format(self.input_data.example.capitalize(),self.input_data.true_salt_vel) + '\033[0m')
        plot_velocity(self.truemodel,self,1000,False)
    def create_sponge(self):
        """
        Create the linear sponge layer.
        """
        if self.isthereTRUEMODEL==False:
            self.create_truemodel()
        class damping_layer(UserExpression):
            def __init__(self, emax, L, D, l, *args, **kwargs):
                UserExpression.__init__(self, *args, **kwargs)
                self.emax = emax
                self.L = L
                self.D = D
                self.l = l
            def eval(self, value, X):
                distx = np.linalg.norm(np.array([X[0]-0.5*self.L, 0]))
                disty = np.linalg.norm(np.array([0, X[1]-(-0.5*self.D)]))
                testx = (distx >= 0.5*self.L and disty <  0.5*self.D)
                testy = (distx  < 0.5*self.L and disty >= 0.5*self.D)
                if testx or testy:
                    ref = np.max([float(testx)*(distx - 0.5*self.L), float(testy)*(disty - 0.5*self.D)])
                elif distx >= 0.5*self.L and disty >= 0.5*self.D:
                    ref = np.linalg.norm(np.array([distx - 0.5*self.L, disty - 0.5*self.D]))
                else:
                    ref = 0
                value[:] = self.emax*ref/self.l
            def value_shape(self):
                return ()
        self.sponge=interpolate(damping_layer(self.model_data.damping,self.model_data.length,self.model_data.depth,self.model_data.layer),self.Vreal)
        self.isthereSPONGE=True
        print('\33[33m' + '    Absorbing layer'+ '\033[0m')
    def create_cases(self):
        """
        Create lists to organize all subproblems based on their source and frequency.
        """
        if self.isthereSPONGE==False:
            self.create_sponge()
        tol=0.01*self.spacing_x
        recs_nodes = []
        for rec in range(len(self.model_data.recs)):
            node = find_dofs(self.Vreal,self.model_data.recs[rec]-tol,self.model_data.recs[rec]+tol,-self.model_data.recs_depth-tol,-self.model_data.recs_depth+tol)
            if len(node)>1:
                warnings.warn("Cannot find one single DOF for each receiver. Tolerance must be redefined.")
            elif len(node)<1:
                warnings.warn("Cannot find one single DOF for each receiver. Nodes are not coincidents.")
            recs_nodes.append(node[0])
        self.rec_obj = Function(self.Vreal)
        self.rec_obj.vector()[:] = 0.0
        self.rec_obj.vector()[recs_nodes] = 1.0
        self.srcs = self.model_data.srcs
        self.freq = np.ones(len(self.model_data.srcs))*self.model_data.freq[0]
        for f in range(len(self.model_data.freq)-1):
            self.srcs = np.concatenate((self.srcs, self.model_data.srcs), axis=None)
            self.freq = np.concatenate((self.freq, np.ones(len(self.model_data.srcs))*self.model_data.freq[f+1]), axis=None)
        self.isthereCASES=True
    def initialize_model(self):
        """
        Create initial model.
        """
        if self.isthereCASES==False:
            self.create_cases()
        linear_guess=Expression('pow((1.5-(x[1]+0.0)*b),-2)',degree=3,b=self.input_data.initial_slope)
        background=interpolate(linear_guess,self.Vreal)
        if self.input_data.traditional_fwi == False:
            a_unif = round(0.000125**(1/self.input_data.q),6) 
            self.a0=interpolate(Constant(a_unif),self.Vreal)
            self.b0=project(background,self.Vreal)
            print('\33[33m' + '    Initial models a_0 and b_0 for TO-based FWI'+ '\033[0m')
        else:
            self.m0=project(background,self.Vreal)
            print('\33[33m' + '    Initial model m_0 for traditional FWI'+ '\033[0m')
        self.isthereINITIAL=True
        subdomains = MeshFunction("size_t", self.mesh, 2)
        subdomains.set_all(0)
        self.dx = Measure('dx', domain=self.mesh, subdomain_data=subdomains)
        if self.input_data.traditional_fwi==True:
            c_0 = project(1/sqrt(abs(self.m0)),self.Vreal) 
            plot_velocity(c_0,self,1001,False)
        else:
            plot_salt(self.a0,self,1001)
            beta = self.model_data.beta_update[0]
            a_bar = (tanh(beta*0.5)+tanh(beta*(self.a0-0.5)))/(tanh(beta*0.5)+tanh(beta*(1.0-0.5)))
            m = (self.model_data.velocity_salt**-2)*(a_bar**self.input_data.q) + self.b0*(1-a_bar**self.input_data.q)
            c_0 = project(1/sqrt(abs(m)),self.Vreal)
            plot_velocity(c_0,self,1001,False)
        print('  ')
        input("  Press Enter to continue...")