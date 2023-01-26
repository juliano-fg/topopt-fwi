#!/usr/bin/env python

# import fenics_adjoint as fa
from fenics import *
from fenics_adjoint import *
from ufl import tanh
import numpy as np
import logging


parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
set_log_active(False)
logging.getLogger('FFC').setLevel(logging.FATAL)

class FiniteElementSolver(object):
    """
    Create the FE model and calculate the gradient. 
    """
    def __init__(self, prep=[], item=[], model=[]):
        self.prep=prep
        self.item=item
        if len(model)==2:
            self.a=model[0]
            self.b=model[1]
        elif len(model)==1:
            self.m=model[0]
    def solve_forward(self): 
        """
        Solve the FE model for the forward problem.
        """
        # Trial and test functions
        pR, pI = TrialFunctions(self.prep.Vcomplex)
        vR, vI = TestFunctions(self.prep.Vcomplex)
        # Boundary conditions
        bc = [DirichletBC(self.prep.Vcomplex.sub(0), 0.0, DomainBoundary()),
              DirichletBC(self.prep.Vcomplex.sub(1), 0.0, DomainBoundary())]
        # Source term and frequency
        source_area=Expression('x[0] > xS-d && x[0] < xS+d && x[1] > zS-d && x[1] < zS+d ? A : 0', 
                               degree=3, xS=self.prep.srcs[self.item], zS=-self.prep.model_data.srcs_depth, 
                               d=0.1*self.prep.spacing_x, A=1.0)
        fI=Constant(0.0)
        fR=interpolate(source_area,self.prep.Vreal)
        unit_source_mag=assemble(fR*dx)
        omega=2.0*np.pi*self.prep.freq[self.item]
        omega_s=2.0*np.pi*self.prep.model_data.srcs_frequency
        magnitude=1e6*((2.0*omega**2)/(np.sqrt(np.pi)*omega_s**3))*np.exp(-(omega**2)/(omega_s**2))
        if self.prep.input_data.inverse_crime==True:
            # Weak form considering constant density 
            a1=(omega**2/self.prep.truemodel**2)*(inner(pR,vR)+inner(pI,vI))+(omega**2/self.prep.truemodel**2)*(inner(pI,vR)-inner(pR,vI))
            a2=(omega*self.prep.sponge)*(inner(pR,vR)+inner(pI,vI))-(omega*self.prep.sponge)*(inner(pI,vR)-inner(pR,vI))
            a3=(inner(grad(pR), grad(vR)) + inner(grad(pI), grad(vI)))+(inner(grad(pI), grad(vR)) - inner(grad(pR), grad(vI)))
            a=a1*dx+a2*dx-a3*dx
            L=-((magnitude/unit_source_mag)*fR*vR+fI*vI)*dx-(fI*vR-(0.0/unit_source_mag)*fR*vI)*dx
        else:
            # Weak form considering variable density 
            facG = (1.0e5**3)/1000
            gardner=facG*0.31*(self.prep.truemodel*1000)**(0.25)
            density=project(gardner,self.prep.Vreal)
            bulk=density*self.prep.truemodel**2
            kappa=project(bulk,self.prep.Vreal)
            a1=(omega**2/kappa)*(inner(pR,vR)+inner(pI,vI)) + (omega**2/kappa)*(inner(pI,vR)-inner(pR,vI))
            a2=(inner(grad(1/density),grad(pR))*vR + inner(grad(1/density),grad(pI))*vI) + (inner(grad(1/density),grad(pI))*vR + inner(grad(1/density),grad(pR))*vI)
            a3=(1/density)*(inner(grad(pR), grad(vR)) + inner(grad(pI), grad(vI))) + (1/density)*(inner(grad(pI), grad(vR)) - inner(grad(pR), grad(vI)))
            d0=(1/density)*(omega*self.prep.sponge)*(inner(pR,vR)+inner(pI,vI)) - (1/density)*(omega*self.prep.sponge)*(inner(pI,vR)-inner(pR,vI)) 
            a=a1*dx+a2*dx-a3*dx+d0*dx
            L= - (1/density)*((magnitude/unit_source_mag)*fR*vR + fI*vI)*dx - (1/density)*(fI*vR - (0.0/unit_source_mag)*fR*vI)*dx
        # Solve the FE problem using umfpack
        p=Function(self.prep.Vcomplex)
        problem=LinearVariationalProblem(a,L,p,bcs=bc)
        solver=LinearVariationalSolver(problem)
        solver.parameters['linear_solver'] = 'umfpack'
        solver.solve() 
        pR_, pI_ = Function.split(p)
        # Store the real part of the pressure wavefield
        self.p = project(pR_,self.prep.Vreal)
    def solve_gradient(self,obs_data=0,outer_iter=0,back=False): 
        """
        Solve the FE model for the forward problem and calculate the gradient.
        """
        def helmholtz_filter(a, w):
            """
            Helmholtz-type spatial filter
            """
            if w == 0.0:
                a_f = a
            else:
                a_f = TrialFunction(self.prep.Vreal)
                vH = TestFunction(self.prep.Vreal)
                F = (inner(w*w*grad(a_f), grad(vH))*dx + inner(a_f, vH)*dx - inner(a, vH)*dx)
                aH = lhs(F)
                LH = rhs(F)
                a_f = Function(self.prep.Vreal)
                solve(aH == LH, a_f)
            return a_f
        def material_model(a,b):
            """
            Interpolation law combining the SIMP model with the Heaviside projection.
            """
            a_tilde = helmholtz_filter(a, self.prep.weight_a)
            beta = self.prep.model_data.beta_update[outer_iter]
            if self.prep.input_data.projection == True:
                a_bar = (tanh(beta*0.5)+tanh(beta*(a_tilde-0.5)))/(tanh(beta*0.5)+tanh(beta*(1.0-0.5)))
            else:
                a_bar = a_tilde
            m = (self.prep.input_data.estimated_salt_vel**-2)*(a_bar**self.prep.input_data.q) + b*(1-a_bar**self.prep.input_data.q)
            return m
        # Trial and test functions
        pR, pI = TrialFunctions(self.prep.Vcomplex)
        vR, vI = TestFunctions(self.prep.Vcomplex)
        # Boundary conditions
        bc = [DirichletBC(self.prep.Vcomplex.sub(0), 0.0, DomainBoundary()),
              DirichletBC(self.prep.Vcomplex.sub(1), 0.0, DomainBoundary())]
        # Source term and frequency
        source_area=Expression('x[0] > xS-d && x[0] < xS+d && x[1] > zS-d && x[1] < zS+d ? A : 0', 
                               degree=3, xS=self.prep.srcs[self.item], zS=-self.prep.model_data.srcs_depth, 
                               d=0.1*self.prep.spacing_x, A=1.0)
        fI=Constant(0.0)
        fR=interpolate(source_area,self.prep.Vreal)
        unit_source_mag=assemble(fR*dx)
        omega=2.0*np.pi*self.prep.freq_m[self.item]
        omega_s=2.0*np.pi*self.prep.model_data.srcs_frequency
        magnitude=1e6*((2.0*omega**2)/(np.sqrt(np.pi)*omega_s**3))*np.exp(-(omega**2)/(omega_s**2))
        # Weak form considering constant density
        if self.prep.input_data.traditional_fwi == True:
            a1=(self.m*omega**2)*(inner(pR,vR)+inner(pI,vI))+(self.m*omega**2)*(inner(pI,vR)-inner(pR,vI))
        else:
            a1=(material_model(self.a,self.b)*omega**2)*(inner(pR,vR)+inner(pI,vI))+(material_model(self.a,self.b)*omega**2)*(inner(pI,vR)-inner(pR,vI))
        a2=(omega*self.prep.sponge)*(inner(pR,vR)+inner(pI,vI))-(omega*self.prep.sponge)*(inner(pI,vR)-inner(pR,vI))
        a3=(inner(grad(pR), grad(vR)) + inner(grad(pI), grad(vI)))+(inner(grad(pI), grad(vR)) - inner(grad(pR), grad(vI)))
        a=a1*dx+a2*dx-a3*dx 
        L=-((magnitude/unit_source_mag)*fR*vR+fI*vI)*dx-(fI*vR-(0.0/unit_source_mag)*fR*vI)*dx
        # Solve the FE problem using umfpack
        p=Function(self.prep.Vcomplex)
        problem=LinearVariationalProblem(a,L,p,bcs=bc)
        solver=LinearVariationalSolver(problem)
        solver.parameters['linear_solver'] = 'umfpack'
        solver.solve() 
        pR_, pI_ = Function.split(p)
        # Collecting the observed data 
        ref = Function(self.prep.Vreal) 
        ref.vector().set_local(obs_data[self.item][1])
        ref.set_allow_extrapolation(True)
        # Calculating and storing the misfit function
        self.J = assemble(0.5*inner(pR_*self.prep.rec_obj-ref*self.prep.rec_obj,pR_*self.prep.rec_obj-ref*self.prep.rec_obj)*dx)
        # Applying the gradient filter
        if self.prep.input_data.traditional_fwi == True:
            dJ_dm = compute_gradient(self.J, Control(self.m))
            self.dJ_dm = helmholtz_filter( dJ_dm, self.prep.weight_s)
        else:
            if back == True:
                dJ_db = compute_gradient(self.J, Control(self.b))
                self.dJ_db = helmholtz_filter( dJ_db, self.prep.weight_s)
            else:
                dJ_da = compute_gradient(self.J, Control(self.a))
                self.dJ_da = helmholtz_filter( dJ_da, self.prep.weight_s)