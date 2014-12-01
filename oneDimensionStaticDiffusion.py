# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:13:52 2014

@author: mrobins
"""
    
    
from dolfin import *
import numpy

# Create mesh and define function space
nx = 50
mesh = UnitInterval(nx)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
mu = 0.25; sigma = 0.1
u0 = Expression('1/(sigma*sqrt(2*3.14))*exp(-pow(x[0]-mu,2)/(2*pow(sigma,2)))',
                mu=mu, sigma=sigma)

#class Boundary(SubDomain):  # define the Dirichlet boundary
#    def inside(self, x, on_boundary):
#        return on_boundary
#
#boundary = Boundary()
#bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
#u_1 = project(u0, V)  # will not result in exact solution!

T = 1.9       # total simulation time
dt = 0.3      # time step

# Define variational problem

# Laplace term
u = TrialFunction(V)
v = TestFunction(V)
a_K = inner(nabla_grad(u), nabla_grad(v))*dx

# "Mass matrix" term
a_M = u*v*dx

M = assemble(a_M)
K = assemble(a_K)
A = M + dt*K

# f term
#f = Expression('beta - 2 - 2*alpha', beta=beta, alpha=alpha)

# Compute solution
u = Function(V)
t = dt
while t <= T:
    print 'time =', t
    # f.t = t
    #f_k = interpolate(f, V)
    #F_k = f_k.vector()
    #b = M*u_1.vector() + dt*M*F_k
    b = M*u_1.vector()
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)

    # Verify
#    u_e = interpolate(u0, V)
#    u_e_array = u_e.vector().array()
#    u_array = u.vector().array()
#    print 'Max error, t=%-10.3f:' % t, numpy.abs(u_e_array - u_array).max()

    t += dt
    u_1.assign(u)
    plot(u_1)
 