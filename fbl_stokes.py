#!/usr/bin/env python
# coding: utf-8

# In[4]:


from dolfin import *
#import matplotlib.pyplot as plt
#%matplotlib inline


# In[5]:


# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
	 "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


# In[6]:


# Load mesh

mesh = Mesh()
with XDMFFile("mesh_fbl_out8.xdmf") as infile:
    infile.read(mesh)

#plot(mesh, title="Mesh")
#plt.show()


# In[7]:


# Define function spaces
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
VP = MixedElement([V, P])
W = FunctionSpace(mesh , VP)


# In[8]:


# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def wall(x, on_boundary):
    return on_boundary and not bool(near(x[0], 0.0) or near(x[0], 1.0))


# In[9]:


# No-slip boundary condition for velocity
noslip = Constant((0.0, 0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, wall)
bc_right = DirichletBC(W.sub(1), Constant(0.), right)
bc_left = DirichletBC(W.sub(1), Constant(0.), left)

# Collect boundary conditions
bcs = [bc0,bc_right,bc_left]


# In[10]:


# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((1.0, 0.0, 0.0))
a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx


# In[11]:


# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)


# In[12]:


# Create Krylov solver and AMG preconditioner
solver = KrylovSolver(krylov_method, "amg")
solver.parameters["monitor_convergence"] = True


# In[13]:


# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)


# In[14]:


# Solve
U = Function(W)
solver.solve(U.vector(), bb)


# In[15]:


# Get sub-functions
u, p = U.split()


# In[16]:


ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p


# In[ ]:




