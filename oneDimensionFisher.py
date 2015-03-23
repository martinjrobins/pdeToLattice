# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:13:52 2014

@author: mrobins
"""
    
    
from dolfin import *
import pyTyche as tyche
import matplotlib.pyplot as plt
import numpy as np

#####################
##    SETUP        ##
#####################

T = 1.0       # total simulation time
pde_dt = 1/100.0      # time step
pde_nx = 60
compart_nx = 20
h = 1.0/compart_nx
D = 1.0
k_1 = 1.0
k_2 = 1.0
N = 1000
interface = 0.5
c_neg1_compartment_index = int(compart_nx/2)-1


#####################
##       PDE       ##
#####################

# Create mesh and define function space
mesh = UnitIntervalMesh(pde_nx)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary and initial conditions
mu = 0.5; sigma = 0.2
u0 = Expression('N/(sigma*sqrt(2*3.14))*exp(-pow(x[0]-mu,2)/(2*pow(sigma,2)))',
                mu=mu, sigma=sigma,N=N)
                
# Define C_-1
class C_neg1(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (interface-h, interface))
        
# Define compartment domain
class Compartment_domain(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (interface, 1.0))
        
# Initialize cell and vertex mesh functions for C_-1
domains_cell = CellFunction("size_t", mesh)
domains_vertex = VertexFunction("size_t", mesh)
domains_cell.set_all(0)
domains_vertex.set_all(0)
compartment_domain = Compartment_domain()
compartment_domain.mark(domains_cell, 2)
compartment_domain.mark(domains_vertex, 2)
c_neg1 = C_neg1()
c_neg1.mark(domains_cell, 1)
c_neg1.mark(domains_vertex, 1)
c_neg1_vertex_index = np.where(domains_vertex.array()==1)[0]
c_neg1_vertex_index = pde_nx-c_neg1_vertex_index
compartment_vertex_index = np.where(domains_vertex.array()==2)[0]
compartment_vertex_index = pde_nx-compartment_vertex_index

# Define new measures
print domains_cell.array()

dx = Measure("dx")[domains_cell]

# Boundary condition for interface
#def interface_bdry(x, on_boundary):
#    return near(x[0],interface)
#bc = NewmannBC(V, Constant(0), bdry)

# Initial condition
u_1 = interpolate(u0, V)
u_2 = interpolate(u0, V)

u_1.vector()[compartment_vertex_index] = 0



# Laplace term
u = TrialFunction(V)
u2 = TrialFunction(V)
v = TestFunction(V)
a_K = D*inner(nabla_grad(u), nabla_grad(v))*(dx(0)+dx(1))
a_K2 = D*inner(nabla_grad(u2), nabla_grad(v))*(dx(0)+dx(1)+dx(2))

# "Mass matrix" term
a_M = u*v*(dx(0)+dx(1))
a_M2 = u*v*(dx(0)+dx(1)+dx(2))

M = assemble(a_M)
K = assemble(a_K)
A = M + pde_dt*K

M2 = assemble(a_M2)
K2 = assemble(a_K2)
A2 = M2 + pde_dt*K2

# source term
#f = Expression('beta - 2 - 2*alpha', beta=beta, alpha=alpha)

# reaction term
r = Vector(pde_nx)
r2 = Vector(pde_nx)

# Compute solution
u = Function(V)
u2 = Function(V)
t = 0

# integral over C_-1 pseudocompartment
c_neg1_integral = u_1*dx(1)


#####################
##    LATTICE      ##
#####################

compartmentsA = tyche.new_species(D)
compartments = tyche.new_compartments([0,0,0],[1.0,h,h],[h,h,h])
compartments.add_diffusion(compartmentsA);
tmp = tyche.new_xplane(interface-h,1)
compartments.scale_diffusion_across(compartmentsA,tmp,0.0)

for x in np.arange(interface+h/2,1,h):
    compartments.set_compartment(compartmentsA,[x,0,0],int(u0([x,0,0])*h))


#####################
##    PLOTTING     ##
#####################

compartments_array = compartmentsA.get_lattice()

# setup plotting
plt.figure()
x_pde = np.arange(0,1+1.0/pde_nx,1.0/pde_nx)
x_pde = np.arange(1,0-1.0/pde_nx,-1.0/pde_nx)
x_compart = np.arange(0,1,h)
print x_pde.shape,u_1.vector().array().shape
plot_pde, = plt.plot(x_pde,u_1.vector().array(),linewidth=2,label='PDE')
plot_pde2, = plt.plot(x_pde,u_2.vector().array(),linestyle='--',linewidth=2,label='PDE_c')
plot_compart = plt.bar(x_compart,compartments_array[:,0,0]/h,width=h)
plt.xlim([0,1])
plt.ylim([0,3000])
plt.legend()

#####################
##    TIME LOOP    ##
#####################
while t <= T:
    print 'time =', t
    
    # Update C_-1 compartment
    c_neg1_val = assemble(c_neg1_integral)
    compartments.set_compartment(compartmentsA,[interface-h/2,0,0],int(c_neg1_val))
    
    compartments_array[c_neg1_compartment_index,0,0] = int(c_neg1_val)
    plot_pde.set_ydata(u_1.vector().array())
    plot_pde2.set_ydata(u_2.vector().array())
    for rect, height in zip(plot_compart, compartments_array):
        rect.set_height(height/h)
    plt.savefig("test_%02.2f.pdf"%t)


    # Integrate lattice to t+pde_dt
    compartments.integrate_for_time(pde_dt,pde_dt)
    
    # update pde in C_-1
    compartments_array = compartmentsA.get_lattice()
    dC_neg1 = compartments_array[c_neg1_compartment_index,0,0]-int(c_neg1_val)
    u_1.vector()[c_neg1_vertex_index] += dC_neg1/h
    
    # f.t = t
    #f_k = interpolate(f, V)
    #F_k = f_k.vector()
    
    # Reaction Term u(u-1)
    r = (k_1*u_1 - k_2*u_1**2)
    r2 = (k_1*u_2 - k_2*u_2**2)
    print r
    
    b = M*u_1.vector() + pde_dt*r
    b2 = M2*u_2.vector()  + pde_dt*r2
    
    u0.t = t
    #bc.apply(A, b)
    solve(A, u.vector(), b)
    solve(A2, u2.vector(), b2)

    t += pde_dt
    u_1.assign(u)
    u_2.assign(u2)

    
 
