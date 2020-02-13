#
# ANDY JONES
# MAE 577 | HOMEWORK 3
#

# Importing pachages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#                                 Functions
# =============================================================================
def set_ghost(u, u_B):
    bound_type = 1
    try:
        if len(u_B) == 2:
            bound_type = 2
        elif len(u_B) == 1:
            u_B = u_B[0]
            bound_type = 1
    except:
        bound_type = 1
        pass

    # Checking to make sure the velocity array passed in is 2D
    if len(u.shape) == 2:
        num_i, num_j = u.shape
        num_i = num_i - 1
        num_j = num_j - 1

    # Doing the wall ghost cells (all i except i == 0 and i == num_i)
    for i in range(1,num_i):
        if bound_type == 1:
            u[i,0] = 2*u_B - u[i,1]
            u[i,num_j] = 2*u_B - u[i,num_j-1]
        elif bound_type == 2:
            u[i,0] = 2*u_B[0] - u[i,1]
            u[i,num_j] = 2*u_B[1] - u[i,num_j-1]


    # Handling the periodic boundary conditions
    for j in range(1,num_j):
        u[0,j] = u[num_i - 1,j]
        u[num_i,j] = u[1,j]

    return u

# =============================================================================
#                             Setting Up Problem
# =============================================================================

# Setting constants
N_x = 0    # Number of nodes in the x direction
N_y = 20    # Number of nodes in the y direction
N_t = 2e2
dt = 0.1
t = 0
T  = N_t * dt
u_B = [0.02, -0.01]     # [m/s] Wall Velocity.
L_x = 0.005  # [m]
L_y = 0.01  # [m]
check_dt = True

# FLuid Properties
u_max = 0.03    # m/s
y_min = 0.005   # m
rho   = 1e3   # kg/m^3
nu    = 1e-6  # m^2/s
dP    = -u_max * 2 * rho * nu / y_min**2 # [Pa/m]


# Analytic Solution
u_fun = lambda y: (-u_max/y_min**2)*y**2 + u_max

# Defining a y distribution and calculating the velocity at each y point
y_dist = np.linspace(-y_min,y_min,1000)
u_vals = u_fun(y_dist)
u_analytic_mean = np.mean(u_vals)
u_init = u_analytic_mean
u_init = 0

# Checking the step size to ensure it is compatible with both directions
if N_y == 0:
    h = L_x/N_x # [m] spacial step size
    if L_y/h != L_y//h:
        N_x0 = N_x
        while L_y/h != L_y//h:
            N_x += 1
            h = L_x/N_x
        print("N_x was changed ", N_x0, " -> ", N_x)
    N_y = L_y//h
if N_x == 0:
    h = L_y/N_y # [m] spacial step size
    if L_x/h != L_x//h:
        N_y0 = N_y
        while L_y/h != L_y//h:
            N_y += 1
            h = L_y/N_y
        print("N_y was changed ", N_y0, " -> ", N_y)
    N_x = L_x//h
x_vals = np.arange(0+h/2,L_x,h)
y_vals = np.arange(0+h/2,L_y,h)
# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# %% ==========================================================================
#                         Setting Initial Conditions
# =============================================================================
N_x = int(N_x)
N_y = int(N_y)

# Setting all of the initial Values to 0
u = np.zeros((N_x + 2, N_y + 2))
u[1:-1,1:-1] = u_init
# plt.plot(y_vals, u[:,1:-1].T)

# %% Applying the boundary conditions
u = set_ghost(u, u_B)

# Creating lists to store all of the variables
u_list = [u]
t_list = [t]

# plt.plot(y_vals, u_list[-1][1:-1,1:-1].T)
# plt.show()
# exit()
plt.figure()
while t < T: # and not user_done:
    # Plotting the velocity profile
    plt.cla()
    plt.plot(u_list[-1][1,1:-1].T, y_vals,'-*',linewidth=2)
    plt.plot([0,u_max],[0,0],'k',linewidth=3)
    plt.plot([0,u_max],[L_y,L_y],'k',linewidth=3)
    # plt.plot(x_vals, u_list[-1][1:-1,1:-1])
    plt.title(str(round(t,4)))
    plt.xlabel("velocity")
    plt.ylabel("y")
    # plt.xlim([0,u_max + 0.1 * u_max])
    plt.ylim([0,L_y])
    plt.pause(0.001)
    # if t == 0:
    #     plt.show()


    # Checking time step if desired
    if check_dt:
        # checking dt
        eps_y = (np.mean(u[1:-1,1:-1])**3)/L_y
        if eps_y != 0:
            Tou_y = (nu/eps_y)**0.5
            if dt/Tou_y > 1:
                print(dt)
                dum = dt
                dt = Tou_y*0.9
                print("\n","decreasing dt  " + str(round(dum,5)) + " --> " + str(round(dt,5)))

    A_x = np.zeros(u.shape)
    D_x = np.zeros(u.shape)
    u_star = np.zeros(u.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            # Advection Term
            A_x[i,j] = (1/(4*h)) * ( (u[i+1,j] + u[i,j])**2 - (u[i-1,j] + u[i,j])**2 )

            # Diffusion Term
            D_x[i,j] = (1/h**2) * ( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])

            # Predictor Step
            u_star[i,j] = u[i,j] + dt * ( -A_x[i,j] + nu * D_x[i,j] )

    # Doing the boundaries on the star velocities
    u_star = set_ghost(u_star,u_B)

    u_ishift = np.zeros(u.shape)
    u_ishift_star = np.zeros(u.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            # Getting the shifted star velocities
            u_ishift_star[i,j] = (1/2) * ( u_star[i+1,j] + u_star[i,j] )

            # Calculating the new time velocities
            u_ishift[i,j] = u_ishift_star[i,j] - (dt/(rho*h)) * dP * h

    u_ishift = set_ghost(u_ishift,u_B)

    # Getting the unshifted values back out
    u_new = np.zeros(u.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            u_new[i,j] = (1/2) * (u_ishift[i,j] + u_ishift[i - 1,j])

    # Setting boundary Conditions and updating time
    u_new = set_ghost(u_new, u_B)
    u = u_new
    t = t + dt

    # Appending values to the lists for storage
    u_list.append(u_new)
    t_list.append(t)

# plt.show()


# exit()
plt.figure()
num_plts = 4
ind_step = len(u_list)//num_plts
for i in range(1,num_plts+1):
    plt.subplot(2,2,i)
    if i == num_plts:
        c_index = -1
    else:
        c_index = int(ind_step*(i-1))

    plt.plot(u_list[c_index][1,1:-1].T, y_vals,'-*',linewidth=2)
    plt.plot([0,u_max],[0,0],'k',linewidth=3)
    plt.plot([0,u_max],[L_y,L_y],'k',linewidth=3)
    # plt.plot(x_vals, u_list[-1][1:-1,1:-1])
    plt.xlabel("velocity")
    plt.ylabel("y")
    # plt.xlim([0,u_max + 0.1 * u_max])
    plt.ylim([0,L_y])

    plt.title("t = " + str(round(t_list[c_index],4)))
plt.show()
