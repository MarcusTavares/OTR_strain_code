# -------------------------------------------------------------------------------------------------------------------------------------
# Supplemental codes for
# Prediction of loss of barrier properties in cracked thin coatings on polymer substrates subjected to tensile strain
# by Marcus Vinícius Tavares da Costa and E. Kristofer Gamstedt
# License:
# This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html
# description: 
# The code resolves the boundary-value problem of steady-state diffusion for a 2D unit cell of a cracked coating on a polymer films 
# described in our paper. As an example, the the calculation below has imputs from MOX coatings of 20 nm shown in the paper.
# ---------------------------------------------------------------------------------------------------------------------------------------

from dolfin import *
import numpy as np

#experimental data
COD_exp = [22.8531E-3, 34.9888E-3, 46.8041E-3, 47.8886E-3, 51.4634E-3, 57.6158E-3, 64.3079E-3, 68.1088E-3, 72.9887E-3, 75.2334E-3, 79.4197E-3, 82.3293E-3, 86.6720E-3, 88.5974E-3, 93.5763E-3]
L_exp = [94.9200, 10.1700, 5.8512, 4.4035, 2.8926, 2.3952, 2.2442, 2.1464, 1.9867, 1.8956, 1.7578, 1.6995, 1.6345, 1.5859, 1.5093]
h_s= 120
h_c = 20E-3
OTR_s = 35
OTR_c = 0.0001
D_s = 1 
j = 1

strain=np.linspace(0.0, 0.16, num=17)
strain_range = 15

count = 0
max_c = []
while count < strain_range:
    L= L_exp[0+count]
    COD= COD_exp[0+count]
    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], -L/2)
        
    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], L/2)

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], -h_s)

    class Coating(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[1], 0.0) and between(x[0], (-L/2, -COD/2) and (COD/2,L/2)))    
     
    class Crack(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[1], 0.0) and between(x[0], (-COD/2,COD/2)) )

    # Initialize sub-domain instances
    left = Left()
    coating = Coating()
    right = Right()
    bottom = Bottom()
    crack = Crack()
    
    # Define mesh
    mesh = RectangleMesh(Point(-L/2, 0.0), Point(L/2, -h_s), 20000, 60)

    # Initialize mesh function for interior domains
    domains = MeshFunction('size_t',mesh,mesh.topology().dim())
    domains.set_all(0)

    # Initialize mesh function for boundary domains
    boundaries=MeshFunction('size_t',mesh,mesh.topology().dim()-1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    coating.mark(boundaries, 2)
    right.mark(boundaries, 3)
    bottom.mark(boundaries, 4)
    crack.mark(boundaries, 5)
    
    # Define input data
    g_L = Constant(0.0)
    g_R = Constant(0.0)
    g_coating = Constant(0.0)
    g_bottom = Constant(0.0)
    g_crack = j
    f = Constant(0.0)

    # Define function space and basis functions
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define Dirichlet boundary conditions at top and bottom boundaries
    bcs = [DirichletBC(V, 0.0, boundaries, 4)]

    # Define new measures associated with the interior domains and
    # exterior boundaries
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # Define variational form
    F = (inner(D_s*grad(u), grad(v))*dx(0) + inner(D_s*grad(u), grad(v))*dx(1)
        - g_L*v*ds(1) - g_R*v*ds(3) - g_coating*v*ds(2) - g_crack*v*ds(5)
        - f*v*dx(0) - f*v*dx(1))

    # Separate left and right hand sides of equation
    a, L = lhs(F), rhs(F)

    # Solve the problem
    u = Function(V)
    solve(a == L, u, bcs)

    #tabulate the nodal coordinates
    u_1 = project(u,V)
    dim = V.dim()
    N = mesh.geometry().dim()
    dof_x = V.tabulate_dof_coordinates().reshape(dim,N)

    # Extract dof indices where some condition on the first coordinate of dof
    # is met
    x = dof_x[:, 0]
    indices = np.where(np.logical_and(x > -COD/2, x < COD/2))[0]

    # Get coordinates of dof
    xs = dof_x[indices]
    y=xs[:, 1]
    indices2 = np.where(np.logical_and(y > -0.000001, y < 0.000001))[0]
    c_along_crack = xs[indices2]

    # Get value of dof 
    vals = u_1.vector()[indices]
    crack_boundary = vals[indices2]
    max_crack_c= np.max(crack_boundary, axis=0)
    max_c.append(max_crack_c)

    # Save solution in VTK format
    #vtkfile = File("test_simu_3.pvd")
    #vtkfile << (u, count)
    count = count + 1

# geometric factor
C_FEM = np.divide((np.multiply((np.divide(COD_exp,L_exp)),(h_s*j))),max_c)

# P_ILT
h_cp = h_s + h_c
OTR_ILT =np.power(((h_s/(h_cp*OTR_s)) + (h_c/(h_cp*OTR_c))),-1)

# OTR-strain
OTR_cs = C_FEM*OTR_s + (1 - 1.75*C_FEM)*OTR_ILT

# Plot OTR-strain
import matplotlib.pyplot as plt
OTR_plot = np.hstack([OTR_ILT,OTR_ILT, OTR_cs])    
fig = plt.figure()
plt.plot(strain, OTR_plot, label='OTR',color='blue', linewidth=3)
plt.legend()
plt.xlabel("Tensile strain",fontsize=16)
plt.ylabel("OTR",fontsize=16)
plt.scatter(strain, OTR_plot, color='blue', marker='^')