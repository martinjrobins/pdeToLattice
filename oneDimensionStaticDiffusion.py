# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:13:52 2014

@author: mrobins
"""
    
    
from dolfin import *
import pyTyche as tyche
import matplotlib.pyplot as plt
import numpy as np

T = 1000.0       # total simulation time
pde_dt = 3      # time step
pde_nx = 50
compart_nx = 10
h = 1.0/compart_nx
D = 1.0/400.0
N = 1000
interface = 0.5
c_neg1_compartment_index = int(compart_nx/2)-1

compartmentsA = tyche.new_species(D)
compartments = tyche.new_compartments([0,0,0],[1.0,h,h],[h,h,h])
compartments.add_diffusion(compartmentsA);
tmp = tyche.new_xplane(interface-h,1)
compartments.scale_diffusion_across2(tmp,0.0)

# Create mesh and define function space
mesh = UnitInterval(pde_nx)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary and initial conditions
mu = 0.25; sigma = 0.1
u0 = Expression('N/(sigma*sqrt(2*3.14))*exp(-pow(x[0]-mu,2)/(2*pow(sigma,2)))',
                mu=mu, sigma=sigma,N=N)
                
# Define C_-1
class C_neg1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (interface-h, interface))
        
# Define compartment domain
class Compartment_domain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > interface
        
# Initialize cell and vertex mesh functions for C_-1
domains_cell = CellFunction("int", mesh)
domains_vertex = VertexFunction("int", mesh)
domains_cell.set_all(0)
domains_vertex.set_all(0)
c_neg1 = C_neg1()
c_neg1.mark(domains_cell, 1)
c_neg1.mark(domains_vertex, 1)
c_neg1_vertex_index = np.where(domains_vertex.array()==1)[0]
c_neg1_vertex_index = pde_nx-c_neg1_vertex_index
compartment_domain = Compartment_domain()
compartment_domain.mark(domains_cell, 2)

# Define new measures
dx = Measure("dx")[domains_cell]

# Boundary condition for interface
#def interface_bdry(x, on_boundary):
#    return near(x[0],interface)
#bc = NewmannBC(V, Constant(0), bdry)

# Initial condition
u_1 = interpolate(u0, V)

# Laplace term
u = TrialFunction(V)
v = TestFunction(V)
a_K = D*inner(nabla_grad(u), nabla_grad(v))*(dx(0)+dx(1))

# "Mass matrix" term
a_M = u*v*(dx(0)+dx(1))

M = assemble(a_M)
K = assemble(a_K)
A = M + pde_dt*K

# source term
#f = Expression('beta - 2 - 2*alpha', beta=beta, alpha=alpha)

# Compute solution
u = Function(V)
t = pde_dt

# integral over C_-1 pseudocompartment
c_neg1_integral = u_1*dx(1)
compartments_array = compartmentsA.get_compartments()

# setup plotting
plt.figure()
x_pde = np.arange(1,0-1.0/pde_nx,-1.0/pde_nx)
x_compart = np.arange(1-h,2,h)
print x_pde.shape,u_1.vector().array().shape
plot_pde, = plt.plot(x_pde,u_1.vector().array(),label='PDE')
plot_compart = plt.bar(x_compart,compartments_array[:,0,0]/h,width=h)
plt.xlim([0,2])
plt.ylim([0,1000])

# time loop
while t <= T:
    print 'time =', t
    
    # Update C_-1 compartment
    c_neg1_val = assemble(c_neg1_integral)
    compartments.set_compartment(compartmentsA,[interface-h/2,0,0],int(c_neg1_val))

    
    # Integrate lattice to t+pde_dt
    compartments.integrate_for_time(pde_dt,pde_dt)
    
    # update pde in C_-1
    compartments_array = compartmentsA.get_compartments()
    dC_neg1 = compartments_array[c_neg1_compartment_index,0,0]-int(c_neg1_val)
    print 'adding ',dC_neg1,' particles to pde C_-1'
    u_1.vector()[c_neg1_vertex_index] += dC_neg1/h
    
    plot_pde.set_ydata(u_1.vector().array())
    for rect, height in zip(plot_compart, compartments_array):
        rect.set_height(height/h)
    plt.pause(0.0001)

    
    # f.t = t
    #f_k = interpolate(f, V)
    #F_k = f_k.vector()
    #b = M*u_1.vector() + pde_dt*M*F_k
    b = M*u_1.vector()
    u0.t = t
    #bc.apply(A, b)
    solve(A, u.vector(), b)

    t += pde_dt
    u_1.assign(u)
 