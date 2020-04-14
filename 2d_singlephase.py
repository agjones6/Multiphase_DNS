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
import copy
import pickle

# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# =============================================================================
#                           Defining Simulation
# =============================================================================
# Class to contain all of the options for input
oc = option_class()
oc.pressure_solve = "value"
oc.output_file = "./Output/testing4/run_1.h5"
oc.show_progress = False
# oc.dt_max = 1e-3
oc.dt_multiplier = 0.5
oc.min_Ploops = int(20)
oc.max_Ploops = int(10000)
oc.Ptol = 1e-4
oc.dt = 1e-6
# oc.max_size = 1e6
oc.M = 3
elapsed_time = lambda st_t: time.time() - st_t

# It wasnt the max number of loops on Pressure

# --> Code for saving classes
# testfile = open("new_file.obj","wb")
# pickle.dump(oc,testfile)
# testfile.close()
# new_oc = pickle.load(open("new_file.obj","rb"))
# print(new_oc)
# exit()

# Initializing the domain class
dc = domain_class(N_x=0,
                  N_y=20,
                  L_x=0.02, #0.06, # 1 ft ~= 0.3m
                  L_y=0.01, #0.06,
                  dt = 5e-6,
                  data_type=data_type
                  # dP_x=dP_analytic THis is not used
                  )

dc.check_dtype()

# Values for the actual Boat
w_boat = 6.553
h_boat = 2.4892
u_boat = 4.69

# Setting initial pressure gradient
dc.dP_x = 0.0
dc.dP_y = 0.0
dc.P = 101325 # atmospheric pressure in pascals

# Initial Velocities
dc.u_init = 0.02 #0.03 #u_analytic_mean
dc.v_init = 0.0 #u_analytic_mean

# Setting the time
dc.T = 6
dc.N_t = dc.T/dc.dt

dc.top   = "outflow"
dc.bottom = "outflow"
dc.left  = "outflow"
dc.right = "outflow"
dc.set_bounds()
# print(dc.domain_map2)

# Changing the wall numbers
dc.domain_map[str_index(dc.domain_map,"w")] = "w_0"
# dc.domain_map[:,-1] = "w_1"
# print(str_index(dc.domain_map,"w"))

# ---> Drawing a box
    # Putting a blockage in the flow
    #   (width, height, angle, [x0,y0])
# dc.draw_box(0.002, 0.005, 60, [0.01,0],letter="w")

# ---> Boat shape
# w = 6.553m
# h = 2.4892m
h_model =  dc.L_y/3
w_model = (6.553/2.4892) * h_model
ori_x = dc.L_x/3-(w_model/2)
ori_y = dc.L_y/2-(h_model/2)
ang = 30
dc.draw_box(w_model, h_model, 90, [ori_x, ori_y],letter="w_1")
# dc.draw_box(h, w/2, ang, [ori_x - h*np.sin(ang*(np.pi*2/360)), ori_y+h/2 + h*np.cos(ang*(np.pi*2/360))],letter="f")
dc.draw_box(h_model, w_model/2, ang, [ori_x, ori_y+h_model/2],letter="f", origin_point="br")
dc.draw_box(h_model, w_model/2, -ang, [ori_x, ori_y+h_model/2],letter="f")

# ---> Wedge on the boat
w2,h2 = dc.h*5, h_model*0.06954*3 # 0.06954 is the actual fraction of the boat height as defined here
dc.draw_box(w2, h2, 60, [ori_x+w_model*(4/5), ori_y+h_model],letter="w_1")

# Changing Soure Numbers
# dc.domain_map[1,:] = "s_0" # Left
# dc.domain_map[dc.domain_map == "s"] = "s_0"
# dc.domain_map[-1,:] = "s_1" # Right
# dc.domain_map[:,-1] = "s_1" # Top

# wall velocities
# dc.u_B = [0.0, 0.03] # 4.69 is the target for boat
dc.u_B = [0.0, u_boat * (dc.L_y/h_boat)]
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

# Updating the shifted domain maps
dc.update_bound_vals()

# Showing a picture of the domain if desired
show_my_domain = True
if show_my_domain:
    print(dc.domain_map.T)
    plt.imshow((np.fliplr(dc.C).T))
    # plt.contourf(dc.C.T)
    plt.show()
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

# --------------------- Bubble Stuff ----------------------------------------
# Defining more than one bubble
x_b_list = [dc.L_x/2] #,dc.L_x/4,dc.L_x/4]
y_b_list = [dc.L_y/2] #,dc.L_y*(1/3),dc.L_y*(2/3)]
r_b_list = [dc.L_y/4] #,dc.L_y/6,dc.L_y/6]
x_b_list = []
y_b_list = []
r_b_list = []

# Defining a level set function and initializing a bubble class
bc = bubble_class(fc.u_ishift[:,:])
for x_b,y_b,r_b in zip(x_b_list,y_b_list,r_b_list):
    bc.add_droplet(dc,x_b,y_b,r_b)
bc.calc_all_psi_d()

# Testing the reinitialization step
dist_field = distance_field(oc.M,bc,dc)
dist_field.psi[:,:] = set_ghost(dc.domain_map, dist_field.psi[:,:], dc.u_B, h=dc.h, type="psi", source=[0])

# Looping 2*M times to satisfy constraints
for i in range(dist_field.num_loops*10):
    dist_field.predict_psi(dc,fc)
    dist_field.correct_psi(dc,fc)
    dist_field.psi[:,:] = set_ghost(dc.domain_map, dist_field.psi[:,:], dc.u_B, h=dc.h, type="psi", source=[0])

# Applyinh boundary conditions
dist_field.psi[:,0] = dist_field.psi[:,1]
dist_field.psi[:,-1] = dist_field.psi[:,-2]
dist_field.psi_ishift[:,0]  = dist_field.psi_ishift[:,1]
dist_field.psi_ishift[:,-1] = dist_field.psi_ishift[:,-2]
dist_field.psi_jshift[:,0] = dist_field.psi_jshift[:,1]
dist_field.psi_jshift[:,-1] = dist_field.psi_jshift[:,-3]
dist_field.psi_jshift[:,-2] = dist_field.psi_jshift[:,-3]

# bc.psi[:,:] = np.copy(dist_field.psi[:,:])
bc.psi = set_ghost(dc.domain_map, bc.psi, dc.u_B, h=dc.h, type="psi", source=[0])
bc.psi[:,0] = bc.psi[:,1]
bc.psi[:,-1] = bc.psi[:,-2]

# updating the fluid properties
# Calculating the neccessary values for Lpsi using the class funcitons
bc.D_psi_fun()
bc.M_switch()
bc.psi_shift_fun(fc)
bc.calc_f_arr(oc.M,dc.h)
bc.calc_F_gamma(dc.gamma,dist_field.M,dc.h)

dist_field.f = set_ghost(dc.domain_map, dist_field.f, dc.u_B, h=dc.h, type="psi", source=[0])
dist_field.f_ishift = set_ghost(dc.domain_map, dist_field.f_ishift, dc.u_B, h=dc.h, type="psi", source=[0])
dist_field.f_jshift = set_ghost(dc.domain_map, dist_field.f_jshift, dc.u_B, h=dc.h, type="psi", source=[0])
fc.rho[:,:] = update_properties(bc.f[:,:],dc.rho_l,dc.rho_g)
fc.mu[:,:] = update_properties(bc.f[:,:],dc.mu_l,dc.mu_g)
fc.rho_ishift = update_properties(bc.f_ishift[:,:],dc.rho_l,dc.rho_g)
fc.rho_jshift = update_properties(bc.f_jshift[:,:],dc.rho_l,dc.rho_g)
fc.mu_ishift = update_properties(bc.f_ishift[:,:],dc.mu_l,dc.mu_g)
fc.mu_jshift = update_properties(bc.f_jshift[:,:],dc.mu_l,dc.mu_g)
# ----------------------------------------------------------------------------

# Initializing pressure field
# for i in range(len(fc.P[:,2])):
#     fc.P[i,1:-1] =  dc.P - (dc.x_grid[0] - dc.x_grid[i])*-2.4
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
# for i in range(len(dc.x_grid[:])):
#     fc.u_ishift[i,1:-1] = u_prof[1:-1]

fc.u_ishift = set_ghost(dc.dm_ishift, fc.u_ishift, dc.u_B, type="u", source=dc.u_S,h=dc.h)
fc.u[1:-1,1:-1] = u_new_fun(fc.u_ishift)


# Defining a class that is used for saving values
sc = save_class(fc,bc)
# plt.figure()
visc_t_lim = (1/4)*((dc.h**2)/dc.nu)
while fc.t < dc.T: # and not user_done:
    # --> Checking time step if desired
    if dc.check_dt:

        # Getting the magnitude of velocity at each point
        vel_mag = vel_mag_fun(fc.u,fc.v)

        # Max Velocity
        max_vel = np.max(vel_mag)

        # Ensuring the maximum velocity is not 0
        if max_vel != 0:
            dc.dt = np.amin([(dc.h/max_vel) * oc.dt_multiplier,visc_t_lim])
        else:
            dc.dt = visc_t_lim

        if dc.dt > oc.dt_max:
            dc.dt = oc.dt_max

        if dc.dt < oc.dt_min:
            dc.dt = oc.dt_min

    # Calculating Advection and Diffusion terms
    fc.A_x[1:-1,1:-1] = A_x_fun(dc.h,fc.u_ishift,fc.v_jshift)
    fc.A_y[1:-1,1:-1] = A_y_fun(dc.h,fc.u_ishift,fc.v_jshift)
    fc.D_x[1:-1,1:-1] = D_fun(dc.h, fc.u_ishift)
    fc.D_y[1:-1,1:-1] = D_fun(dc.h, fc.v_jshift)

    # --> Bubble Stuff
    # bc.calc_F_gamma(dc.gamma,dist_field.M,dc.h)

    fc.u_ishift_star = vel_star(fc.u_ishift, dc.dt, fc.A_x, fc.mu_ishift/fc.rho_ishift, fc.D_x,bc.F_gamma_x/fc.rho_ishift)
    fc.v_jshift_star = vel_star(fc.v_jshift, dc.dt, fc.A_y, fc.mu_jshift/fc.rho_jshift, fc.D_y,bc.F_gamma_y/fc.rho_jshift)

    # print(fc.u_ishift_star)

    # Setting the ghost cells of the preiction velocities
    fc.u_ishift_star = set_ghost(dc.dm_ishift, fc.u_ishift_star, dc.u_B, type="u", source=dc.u_S,h=dc.h)
    fc.v_jshift_star = set_ghost(dc.dm_jshift, fc.v_jshift_star, dc.v_B, type="v", source=dc.v_S,h=dc.h)

    # Calculating the new pressures if it is desired
    if oc.pressure_solve == "gradient":
        fc.dP_x, fc.dP_y = calc_pressure_grad(dc, fc.dP_x, fc.dP_y, u_ishift_star, v_ishift_star)
        fc.dP_x = set_ghost(dc.domain_map,fc.dP_x, dc.u_B, type="p",source=dc.dP_x_S)
        fc.dP_y = set_ghost(dc.domain_map,fc.dP_y, dc.u_B, type="p",source=dc.dP_y_S)

    elif oc.pressure_solve.lower() == "value":
        # Setting the ghost cells for pressure
        fc.P = set_ghost(dc.domain_map, fc.P, dc.u_B, h=dc.h, type="p",source=dc.P_S)

        # Calculating the cell valued pressures
        fc.P = calc_pressure(dc,oc,np.copy(fc.rho[:,:]), np.copy(fc.P), fc.u_ishift_star, fc.v_jshift_star)

        # Calculating the Differential Pressure in x and y
        fc.dP_x[1:-1,1:-1] = (fc.P[2:,1:-1] - fc.P[1:-1,1:-1])/dc.h
        fc.dP_y[1:-1,1:-1] = (fc.P[1:-1,2:] - fc.P[1:-1,1:-1])/dc.h

    elif oc.pressure_solve == "constant_gradient":
        dP_x = fc.dP_x
        dP_y = fc.dP_y

    # Calculating the new time shifted velocities
    fc.u_ishift = vel_ishift_fun(fc.u_ishift_star, dc.dt, fc.rho, dc.h, fc.dP_x, dc.F_x)
    fc.v_jshift = vel_ishift_fun(fc.v_jshift_star, dc.dt, fc.rho, dc.h, fc.dP_y, dc.F_y)

    # Applying boundaries to the new shifted velocities
    fc.u_ishift = set_ghost(dc.dm_ishift, fc.u_ishift, dc.u_B, type="u", source=dc.u_S,h=dc.h)
    fc.v_jshift = set_ghost(dc.dm_jshift, fc.v_jshift, dc.v_B, type="v", source=dc.v_S,h=dc.h)


    # --------------------- Bubble Stuff ----------------------------------------
    # # Handling the level set funciton for the bubble
    # bc.predict_psi(dc,fc)
    # bc.correct_psi(dc,fc)
    # # plt.clf()
    # # test = np.copy(fc.v_jshift)
    # # # test[test == 0 ] = np.nan
    # # plt.streamplot(dc.x_grid,dc.y_grid,fc.u_ishift.T,fc.v_jshift.T)
    # # plt.show()
    # bc.psi = set_ghost(dc.domain_map, bc.psi, dc.u_B, h=dc.h, type="psi", source=[0])
    #
    #
    # # Correcting Psi to be a distance function by looping 2*M times
    # dist_field.psi[:,:] = np.copy(bc.psi[:,:])
    # for i in range(dist_field.num_loops*2):
    #     dist_field.predict_psi(dc,fc)
    #     dist_field.correct_psi(dc,fc)
    #     dist_field.psi[:,:] = set_ghost(dc.domain_map, dist_field.psi[:,:], dc.u_B, h=dc.h, type="psi", source=[0])
    #     # plt.clf()
    #     # test = np.copy(dist_field.psi)
    #     # # test[test == 0 ] = np.nan
    #     # plt.contourf(test.T)
    #     # plt.title(fc.t)
    #     # plt.show()
    #
    # # Applying boundaries
    # dist_field.psi[:,0] = dist_field.psi[:,1]
    # dist_field.psi[:,-1] = dist_field.psi[:,-2]
    # # dist_field.psi_ishift[:,0]  = dist_field.psi_ishift[:,1]
    # # dist_field.psi_ishift[:,-1] = dist_field.psi_ishift[:,-2]
    # # dist_field.psi_jshift[:,0] = dist_field.psi_jshift[:,1]
    # # dist_field.psi_jshift[:,-1] = dist_field.psi_jshift[:,-3]
    # # dist_field.psi_jshift[:,-2] = dist_field.psi_jshift[:,-3]
    #
    # # Changing the Psi in the bubble class to the normalized one
    # # bc.psi[:,:] = np.copy(dist_field.psi[:,:])
    # bc.D_psi_fun()
    # bc.M_switch()
    # bc.psi_shift_fun(fc)
    # bc.calc_f_arr(oc.M,dc.h)
    # dist_field.calc_f_arr(dist_field.M,dc.h)
    # dist_field.f = set_ghost(dc.domain_map, dist_field.f, dc.u_B, h=dc.h, type="psi", source=[0])
    # dist_field.f_ishift = set_ghost(dc.dm_ishift, dist_field.f_ishift, dc.u_B, h=dc.h, type="psi", source=[0])
    # dist_field.f_jshift = set_ghost(dc.dm_jshift, dist_field.f_jshift, dc.u_B, h=dc.h, type="psi", source=[0])
    # fc.rho[:,:] = update_properties(bc.f[:,:],dc.rho_l,dc.rho_g)
    # fc.mu[:,:] = update_properties(bc.f[:,:],dc.mu_l,dc.mu_g)
    # fc.rho_ishift = update_properties(bc.f_ishift[:,:],dc.rho_l,dc.rho_g)
    # fc.rho_jshift = update_properties(bc.f_jshift[:,:],dc.rho_l,dc.rho_g)
    # fc.mu_ishift = update_properties(bc.f_ishift[:,:],dc.mu_l,dc.mu_g)
    # fc.mu_jshift = update_properties(bc.f_jshift[:,:],dc.mu_l,dc.mu_g)
    # ----------------------------------------------------------------------------

    # Getting the unshifted values back out
    fc.u[1:-1,1:-1] = u_new_fun(fc.u_ishift)
    fc.v[1:-1,1:-1] = v_new_fun(fc.v_jshift)

    # Setting boundary Conditions and updating time
    fc.t += dc.dt

    # Getting rid of very small numbers
    # fc.u = np.round(fc.u,6)
    # fc.v = np.round(fc.v,6)

    sc.save_values(elapsed_time,dc, fc, bc, oc)

print("Simulation Completed")
exit()
