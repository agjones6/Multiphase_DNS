#
# ANDY JONES
# MAE 577
# Fluid Simulation Code
#

# =========================== CURRENT CAPABILITY  ==============================
#
#           Dimensions: 2D output in (x,y,t) format
#     Boundary Options: Wall, Periodic, Source, Outflow
#       Pressure Solve: iterative gradient solver, constant gradient
#   Domain Description: The domain is described by a class (domain_class)
#    Compiling Options: Only CPU and float64 are supported
#               Output: Saved as an '.hdf5' file
#                         u    = x-direction velocity 3D array (x,y,t)
#                         v    = y-direction velocity 3D array (x,y,t)
#                         dP_x = x-direction Pressure Gradient 3D array (x,y,t)
#                         dP_y = y-direction Pressure Gradient 3D array (x,y,t)
#                         t    = time 1D array (t)
#                         x    = x-grid locations 1D array (x)
#                         y    = y-grid locations 1D array (y)
#
# ==============================================================================

# Importing pachages
import time
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import h5py
from numba import jit, cuda, vectorize, float64, float32, guvectorize
from functions_2d import *

# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# =============================================================================
#                           Defining Simulation
# =============================================================================
# Class to contain all of the options for input
oc = option_class()
oc.pressure_solve = "value"
oc.output_file = "./Output/testing2/run_1.h5"
oc.show_progress = False

# Initializing the domain class
dc = domain_class(N_x=0,
                  N_y=30,
                  L_x=0.02,
                  L_y=0.01,
                  dt = 5e-6,
                  data_type=data_type
                  # dP_x=dP_analytic THis is not used
                  )

dc.check_dtype()

# Setting initial pressure gradient
dc.dP_x = 0.0
dc.dP_y = 0.0
dc.P = 101325 # atmospheric pressure in pascals

# Initial Velocities
dc.u_init = -0.01 #0.03 #u_analytic_mean
dc.v_init = 0.0 #u_analytic_mean

# Setting the time
dc.T = 60
dc.N_t = dc.T/dc.dt

dc.top   = "wall"
dc.bottom = "wall"
dc.left  = "periodic"
dc.right = "periodic"
dc.set_bounds()

# Putting a blockage in the flow
width = 0.008
st_x    = int(dc.N_x//4 - (width//dc.h)*0.5)
en_x = int(st_x+width//dc.h)
height = 0.012
st_y    = 0 #int(dc.N_y//2 - (height//dc.h)*0.5)
en_y = int(st_y+height//dc.h)

width2 = 0.0015
st_x2 = int(dc.N_x//4 - ((width+width2)//dc.h)*0.5)
en_x2 = int(st_x2+(width2)//dc.h)
height2 = 0.0005
st_y2 = int(en_y)
en_y2 = int(st_y+(height+height2)//dc.h)

# dc.domain_map[st_x:en_x,st_y:en_y] = "w"
# dc.domain_map[st_x2:en_x2,st_y2:en_y2] = "w"

# Changing the wall numbers
dc.domain_map[dc.domain_map == "w"] = "w_0"
# dc.domain_map[:,-1] = "w_1"

# Changing Soure Numbers
# dc.domain_map[1,:] = "s_0" # Left
dc.domain_map[dc.domain_map == "s"] = "s_0"
# dc.domain_map[-1,:] = "s_1" # Right
# dc.domain_map[:,-1] = "s_1" # Top

# wall velocities
dc.u_B = [0, 0] # 4.69 is the target for
dc.v_B = [0, 0]

# Source Terms
dc.u_S    = [0.0, 0.0]
dc.v_S    = [0, 0]
dc.dP_x_S = [0. , 0]
dc.dP_y_S = [0., 0.]
dc.P_S = [dc.P, dc.P-(2.4*dc.L_y)] # -2.4 Pa/m

# Getting a mesh of x and y values
# dc.x_grid = np.arange(0-dc.h/2,dc.L_x+dc.h/2,dc.h)
# dc.y_grid = np.arange(0-dc.h/2,dc.L_y+dc.h/2,dc.h)

# Showing a picture of the domain if desired
show_my_domain = False
if show_my_domain:
    print(dc.domain_map.T)
    # print(np.flip(dc.domain_map).T)
    # print(dc.domain_map.T)
    # plt.figure()
    # show_domain(dc.domain_map.T)
    # plt.show()
    exit()

# =============================================================================
#                          Running the Simulation
# =============================================================================

# Initializing the flow class
fc = flow_class(dc)

# Testing to see if I can make all of the values equal to a value an array equal to something

# Initializing pressure field
for i in range(len(fc.P[:,2])):
    fc.P[i,1:-1] =  dc.P - (dc.x_grid[0] - dc.x_grid[i])*-2.4
    # u[i,1:-1] =  dc.u_init - (dc.x_grid[0] - dc.x_grid[i])*0.0001

# %% Applying the boundary conditions
if dc.data_type == "float64":
    numpy_dtype = np.float64
elif dc.data_type == "float32":
    numpy_dtype = np.float32

# Temporarily making a velocity profile for the inlet and exit
u_max = 0.03
y_min = 0.005
u_fun = lambda y: (-u_max/y_min**2)*y**2 + u_max
u_prof = u_fun(dc.y_grid - y_min)
# for i in range(len(dc.y_grid[1:-1])):
#     u_ishift[i+1,1:-1] = u_prof[1:-1]
# u_ishift[-1,:] = u_prof

# Defining a class that is used for saving values
sc = save_class(fc)

while fc.t < dc.T: # and not user_done:
    # --> Checking time step if desired
    if dc.check_dt:

        # Getting the magnitude of velocity at each point
        vel_mag = vel_mag_fun(fc.u,fc.v)

        # Max Velocity
        max_vel = np.max(vel_mag)

        # Ensuring the maximum velocity is not 0
        if max_vel != 0:
            dc.dt = (dc.h/max_vel) * oc.dt_multiplier

        if dc.dt > oc.dt_max:
            dc.dt = oc.dt_max

        if dc.dt < oc.dt_min:
            dc.dt = oc.dt_min

    # Getting values for the predictor step
    fc.v_jshift[:,-1] = 0
    fc.v_jshift[:,-2] = 0
    fc.v_jshift[:,0] = 0

    # Calculating Advection and Diffusion terms
    fc.A_x[1:-1,1:-1] = A_x_fun(dc.h,fc.u_ishift,fc.v_jshift)
    fc.A_y[1:-1,1:-1] = A_y_fun(dc.h,fc.u_ishift,fc.v_jshift)
    fc.D_x[1:-1,1:-1] = D_fun(dc.h, fc.u_ishift)
    fc.D_y[1:-1,1:-1] = D_fun(dc.h, fc.v_jshift)

    # Predictor Step
    fc.u_ishift_star = vel_star(fc.u_ishift, dc.dt, fc.A_x, dc.nu, fc.D_x) # u + dc.dt * ( -A_x + nu * D_x )
    fc.v_jshift_star = vel_star(fc.v_jshift, dc.dt, fc.A_y, dc.nu, fc.D_y) #v + dc.dt * ( -A_y + nu * D_y )

    # Setting the ghost cells of the preiction velocities
    fc.u_ishift_star = set_ghost(dc.domain_map, fc.u_ishift_star, dc.u_B, source=dc.u_S)
    fc.v_jshift_star = set_ghost(dc.domain_map, fc.v_jshift_star, dc.v_B, source=dc.v_S)

    # HARD CODED BOUNDARIES
    fc.v_jshift_star[:,-1] = 0
    fc.v_jshift_star[:,-2] = 0
    fc.v_jshift_star[:,0] = 0

    # Calculating the new pressures if it is desired
    if oc.pressure_solve == "gradient":
        fc.dP_x, fc.dP_y = calc_pressure_grad(dc, fc.dP_x, fc.dP_y, u_ishift_star, v_ishift_star)
        fc.dP_x = set_ghost(dc.domain_map,fc.dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        fc.dP_y = set_ghost(dc.domain_map,fc.dP_y, dc.u_B, type="pressure",source=dc.dP_y_S)

    elif oc.pressure_solve.lower() == "value":
        # Setting the ghost cells for pressure
        fc.P = set_ghost(dc.domain_map, fc.P, dc.u_B, h=dc.h, type="pressure",source=dc.P_S)

        # Calculating the cell valued pressures
        fc.P = calc_pressure(dc, fc.P, fc.u_ishift_star, fc.v_jshift_star)

        # Calculating the Differential Pressure in x and y
        fc.dP_x[1:-1,1:-1] = (fc.P[2:,1:-1] - fc.P[1:-1,1:-1])/dc.h
        fc.dP_y[1:-1,1:-1] = (fc.P[1:-1,2:] - fc.P[1:-1,1:-1])/dc.h

        # Rounding the Differential Pressures
        fc.dP_x = np.round(fc.dP_x,6)
        fc.dP_y = np.round(fc.dP_y,6)

        # plt.figure()
        # plt.contourf(dc.x_grid[1:-1],dc.y_grid[1:-1],P[1:-1,1:-1].T/np.max(P[1:-1,1:-1]))
        # plt.contourf(dc.x_grid[1:-1],dc.y_grid[1:-1],dP_y[1:-1,1:-1].T)
        # plt.show()
        # plt.pause(0.001)
        # exit()
    elif oc.pressure_solve == "constant_gradient":
        dP_x = fc.dP_x
        dP_y = fc.dP_y

    # Calculating the new time shifted velocities
    fc.u_ishift = vel_ishift_fun(fc.u_ishift_star, dc.dt, dc.rho, dc.h, fc.dP_x, dc.F_x)
    fc.v_jshift = vel_ishift_fun(fc.v_jshift_star, dc.dt, dc.rho, dc.h, fc.dP_y, dc.F_y)

    # Applying boundaries to the new shifted velocities
    fc.u_ishift = set_ghost(dc.domain_map,fc.u_ishift,dc.u_B,source=dc.u_S)
    fc.v_jshift = set_ghost(dc.domain_map,fc.v_jshift,dc.v_B,source=dc.v_S)

    # HARD CODED BOUNDARIES
    fc.v_jshift[:,-1] = 0
    fc.v_jshift[:,-2] = 0
    fc.v_jshift[:,0] = 0

    # Getting the unshifted values back out
    fc.u[1:-1,1:-1] = u_new_fun(fc.u_ishift)
    fc.v[1:-1,1:-1] = v_new_fun(fc.v_jshift)
    # Setting boundary Conditions and updating time
    fc.t += dc.dt

    # Getting rid of very small numbers
    fc.u = np.round(fc.u,6)
    fc.v = np.round(fc.v,6)

    sc.save_values(dc, fc, oc)

print("Simulation Completed")
exit()
