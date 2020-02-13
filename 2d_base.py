#
# ANDY JONES
# MAE 577 | HOMEWORK 3
#

# Importing pachages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import keyboard
# from ipywidgets import Label, HTML, HBox, Image, VBox, Box
# from ipyevents import Event
# from IPython.display import display
# from getkey import getkey, keys

# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# =============================================================================
#                                 Functions
# =============================================================================
def set_ghost(map, u, u_B):
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

    def find_fluid(map,loc):
        x = loc[0]
        y = loc[1]

        x_max,y_max = map.shape
        x_min,y_min = 0,0

        x_max = x_max - 1
        y_max = y_max - 1

        top_i = (x,y+1)
        bot_i = (x,y-1)
        right_i = (x+1,y)
        left_i = (x-1,y)
        if y < y_max:
            top = map[top_i]
        else:
            top = ""

        if y > y_min:
            bottom = map[bot_i]
        else:
            bottom = ""

        if x < x_max:
            right = map[right_i]
        else:
            right = ""

        if x > x_min:
            left = map[left_i]
        else:
            left = ""

        bound_array = np.array([top,bottom,left,right])
        index_array = np.array([top_i, bot_i, right_i, left_i])
        try:
            final_index = index_array[bound_array=="f"][0]
            final_index = (final_index[0],final_index[1])
        except:
            final_index = (x,y)

        return final_index
        print(final_index)

    for i in range(len(map[:,0])):
        for j in range(len(map[0,:])):
            if map[i,j] == "w":
                f_ind = find_fluid(map,[i,j])
                print(f_ind)
                # if bound_type == 1:
                #     u[i,j] = 2*u_B - u[i,j+1]
                # elif bound_type == 2:
                #     u[i,0] = 2*u_B[0] - u[i,1]
                #     u[i,num_j] = 2*u_B[1] - u[i,num_j-1]


    exit()
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

def make_plot(mp, x, y, u, v, **kwargs):
    # Handling optional arguments
    LB = kwargs.get("LB",[])
    UB = kwargs.get("UB",[])
    plot_type = kwargs.get("plot_type","profile")
    sub_type = kwargs.get("sub_type",["u","y"])

    # Plotting the velocity profile
    # --> Plotting
    mp.cla()

    vars = [[],[]]
    if plot_type.lower() == "profile":
        for i in range(len(sub_type)):
            if "x" in sub_type[i]:
                vars[i] = x
            if "y" in sub_type[i]:
                vars[i] = y
            if "u" in sub_type[i]:
                vars[i] = u
            if "v" in sub_type[i]:
                vars[i] = v

        # Making the main plot
        mp.plot(vars[0], vars[1],'-*',linewidth=2)

        # Adding the walls if it is desired
        if LB != []:
            mp.plot(mp.get_xlim(),[LB,LB],'k',linewidth=3)
        if UB != []:
            mp.plot(mp.get_xlim(),[UB,UB],'k',linewidth=3)

        # Making a 0 line
        if mp.get_xlim()[0] <= 0 and mp.get_xlim()[1] >=0 and LB != [] and UB != []:
            mp.plot([0,LB],[0,UB],'--k')

        # Setting the boundaries
        if LB != [] and UB != []:
            mp.set_ylim([LB,UB])


    elif plot_type.lower() == "field":
        M = np.hypot(u, v)
        mp.quiver(x,y,u,v,M,linewidth=0.1,edgecolor=(0,0,0),cmap="jet")


    # elif plot_type.lower() == "surface":
    #     Axes3D.
def show_domain(map):
    # Changing the values to integers
    map[map=="f"] = 0
    map[map=="w"] = 1
    map[map=="p"] = 2

    try:
        map = np.double(map)
    except:
        print("couldn't convert to integers")
        return
    # Ploting the image
    plt.imshow(map,origin="bottom")

def set_boundary(N_space,**kwargs):

    left = kwargs.get("left","periodic")
    right = kwargs.get("right","periodic")
    top = kwargs.get("top","wall")
    bottom = kwargs.get("bottom","wall")

    def my_str(arg):
        if "periodic" in arg:
            return "p"
        elif "wall" in arg:
            return "w"
        else:
            return "f"

    # Creating empty array
    N_space = np.array(N_space) + 2
    # blk_array = np.chararray(N_space)
    blk_array = np.empty(N_space,dtype="object")
    blk_array[:] = "f"

    # Going through the edges of the array to assign boundary conditions
    map = blk_array

    map[0,:] = my_str(left)
    map[-1,:] = my_str(right)
    map[:,0] = my_str(top)
    map[:,-1] = my_str(bottom)

    return map


# =============================================================================
#                             Analytic Solution
# =============================================================================
# Fluid Properties
u_max = 0.03    # m/s
y_min = 0.005   # m
rho   = 1e3   # kg/m^3
nu    = 1e-6  # m^2/s
dP_analytic  = -u_max * 2 * rho * nu / y_min**2 # [Pa/m]


# Analytic Solution
u_fun = lambda y: (-u_max/y_min**2)*y**2 + u_max

# Defining a y distribution and calculating the velocity at each y point
y_dist = np.linspace(-y_min,y_min, 1000)
u_vals = u_fun(y_dist)
u_analytic_mean = np.mean(u_vals)

# =============================================================================
#                             Setting Up Problem
# =============================================================================
# Spacial Domain
N_x = 0    # Number of nodes in the x direction
N_y = 30    # Number of nodes in the y direction
L_x = 0.002  # [m]
L_y = 0.01  # [m]

# Time Domain Variables
dt = 1e-1
N_t = 2e2
t = 0
T  = N_t * dt

# Wall Velocities
# u_B = [-0.02, 0.03]     # [m/s] Wall Velocity.
u_B = 0
v_B = 0     # [m/s] Wall Velocity.
# u_B = [0.0, 0]     # [m/s] Wall Velocity.

# Pressure Gradient
dP_x = dP_analytic*100
dP_y = 0 #-1

# Initial Velocities
u_init = u_analytic_mean
# u_init = 0
# v_init = 0.02
v_init = -0.01

# Options for Stability
check_dt = True
lower_tolerance = 1e-20

# %% ==========================================================================
#                         Setting Initial Conditions
# =============================================================================

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

# Getting a mesh of x and y values
x_vals = np.arange(0+h/2,L_x,h)
y_vals = np.arange(0+h/2,L_y,h)

# Converting the Number of Nodes to integers
N_x = int(N_x)
N_y = int(N_y)

# Intializing the velocity Array
u = np.zeros((N_x + 2, N_y + 2))
u[1:-1,1:-1] = u_init
v = np.zeros((N_x + 2, N_y + 2))
# v[1:-1,1:N_y//2] = v_init
# v[1:-1,N_y//2:-1] = -v_init
v[1:-1,1:-1] = v_init

# Defining the wall boundary
domain_map = set_boundary([N_x,N_y])
# domain_map[4,0:5] = "w"
# show_domain(domain_map.T)
# plt.show()

# %% Applying the boundary conditions
u = set_ghost(domain_map, u, u_B)
v = set_ghost(v, v_B)
exit()

# Creating lists to store all of the variables
u_list = [u]
v_list = [v]
t_list = [t]

plt.figure()
my_plot1 = plt.subplot(2,1,1)
my_plot2 = plt.subplot(2,2,3)
my_plot3 = plt.subplot(2,2,4)
while t < T: # and not user_done:
    # --> VECTOR FIELD PLOT
    make_plot(my_plot1, x_vals, y_vals,
              u_list[-1][1:-1,1:-1].T,
              v_list[-1][1:-1,1:-1].T,
              LB=0,
              UB=L_y,
              plot_type="field",
              sub_type=["u","y"]
              )
    # Axis labels and title
    my_plot1.set_title(str(round(t,4)))
    my_plot1.set_xlabel("x")
    my_plot1.set_ylabel("y")

    # --> U DISTRIBITION PROFILE
    make_plot(my_plot2, x_vals, y_vals,
              u_list[-1][1:-1,1:-1].T,
              v_list[-1][1:-1,1:-1].T,
              LB=0,
              UB=L_y,
              plot_type="profile",
              sub_type=["u","y"]
              )

    # Axis labels and title
    my_plot2.set_xlabel("x-velocity (->)")
    my_plot2.set_ylabel("y")

    # --> V DISTRIBUTION PROFILE
    make_plot(my_plot3, x_vals, y_vals,
              u_list[-1][1:-1,1:-1].T,
              v_list[-1][1:-1,1:-1].T,
              LB=0,
              UB=L_y,
              plot_type="profile",
              sub_type=["v","y"]
              )

    # Axis labels and title
    my_plot3.set_xlabel("y-velocity (^)")
    my_plot3.set_ylabel("y")

    # Showing the plot
    plt.pause(0.001)

    # Showing the initial Conditions
    if t == 0:
        plt.pause(2)
    # exit()

    # --> Checking time step if desired
    if check_dt:
        # checking dt
        eps_x = (np.mean(u[1:-1,1:-1])**3)/L_x
        eps_y = (np.mean(v[1:-1,1:-1])**3)/L_y
        if eps_x != 0:
            Tou_x = abs((nu/eps_x))**0.5
            if dt/Tou_x > 1 :
                dum = dt
                dt = Tou_x*0.7
                print("\n","decreasing dt  " + str(round(dum,5)) + " --> " + str(round(dt,5)))
        if eps_y != 0:
            Tou_y = abs((nu/eps_y))**0.5
            if dt/Tou_y > 1:
                dum = dt
                dt = Tou_y*0.7
                print("\n","decreasing dt  " + str(round(dum,5)) + " --> " + str(round(dt,5)))

    A_x = np.zeros(u.shape)
    A_y = np.zeros(v.shape)
    D_x = np.zeros(u.shape)
    D_y = np.zeros(v.shape)
    u_star = np.zeros(u.shape)
    v_star = np.zeros(u.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            # Advection Term
            A_x[i,j] = (1/(4*h)) * ( (u[i+1,j] + u[i,j])**2 - (u[i-1,j] + u[i,j])**2
                                   + (u[i,j+1] + u[i,j]) * (1/4)*((v[i,j]+v[i,j+1]+v[i+1,j]+v[i+1,j+1]) + (v[i,j]+v[i,j+1]+v[i-1,j]+v[i-1,j+1]))
                                   - (u[i,j] + u[i,j-1]) * (1/4)*((v[i,j]+v[i,j-1]+v[i+1,j]+v[i+1,j-1]) + (v[i,j]+v[i,j-1]+v[i-1,j]+v[i-1,j-1])) )

            A_y[i,j] = (1/(4*h)) * ( (v[i,j+1] + v[i,j])**2 - (v[i,j-1] + v[i,j])**2
                                  + (v[i+1,j] + v[i,j]) * (1/4)*((u[i,j]+u[i,j+1]+u[i+1,j]+u[i+1,j+1]) + (u[i,j]+u[i,j+1]+u[i-1,j]+u[i-1,j+1]))
                                  - (v[i,j] + v[i-1,j]) * (1/4)*((u[i,j]+u[i,j-1]+u[i+1,j]+u[i+1,j-1]) + (u[i,j]+u[i,j-1]+u[i-1,j]+u[i-1,j-1])) )

            # Diffusion Term
            D_x[i,j] = (1/h**2) * ( u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
            D_y[i,j] = (1/h**2) * ( v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j])

    # Predictor Step
    u_star = u + dt * ( -A_x + nu * D_x )
    v_star = v + dt * ( -A_y + nu * D_y )

    # Doing the boundaries on the star velocities
    u_star = set_ghost(u_star,u_B)
    v_star = set_ghost(v_star,v_B)

    u_ishift = np.zeros(u.shape)
    v_ishift = np.zeros(v.shape)
    u_ishift_star = np.zeros(u.shape)
    v_ishift_star = np.zeros(v.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            # Getting the shifted star velocities
            # u_ishift_star = (1/2) * (u_star[2:-1] + u_star[1:-2])
            u_ishift_star[i,j] = (1/2) * ( u_star[i+1,j] + u_star[i,j] )
            v_ishift_star[i,j] = (1/2) * ( v_star[i,j+1] + v_star[i,j] )

    # Calculating the new time velocities
    u_ishift = u_ishift_star - (dt/(rho*h)) * dP_x * h
    v_ishift = v_ishift_star - (dt/(rho*h)) * dP_y * h

    u_ishift = set_ghost(u_ishift,u_B)
    v_ishift = set_ghost(v_ishift,v_B)

    # Getting the unshifted values back out
    u_new = np.zeros(u.shape)
    v_new = np.zeros(v.shape)
    for i in range(1, 1 + N_x):
        for j in range(1, 1 + N_y):
            u_new[i,j] = (1/2) * (u_ishift[i,j] + u_ishift[i - 1,j])
            v_new[i,j] = (1/2) * (v_ishift[i,j] + v_ishift[i - 1,j])

    # Setting boundary Conditions and updating time
    u_new = set_ghost(u_new, u_B)
    v_new = set_ghost(v_new, v_B)
    u = u_new
    v = v_new
    t += dt

    # Getting rid of very small numbers
    u[abs(u)<lower_tolerance] = 0
    v[abs(v)<lower_tolerance] = 0

    # Appending values to the lists for storage
    u_list.append(u_new)
    v_list.append(v_new)
    t_list.append(t)

    # key = getkey(blocking=True)
    # # print(key)
    # if "a" in key:
    #     # t = T
    #     print("break Loop")
    #     # break




# plt.show()


# exit()
# plt.figure()
# num_plts = 4
# ind_step = len(u_list)//num_plts
# for i in range(1,num_plts+1):
#     plt.subplot(2,2,i)
#     if i == num_plts:
#         c_index = -1
#     else:
#         c_index = int(ind_step*(i-1))
#
#     plt.plot(v_list[c_index][1,1:-1].T, y_vals,'-*',linewidth=2)
#     # plt.plot([0,u_max],[0,0],'k',linewidth=3)
#     # plt.plot([0,u_max],[L_y,L_y],'k',linewidth=3)
#     # plt.plot(x_vals, u_list[-1][1:-1,1:-1])
#     # plt.xlabel("velocity")
#     # plt.ylabel("y")
#     # plt.xlim([0,u_max + 0.1 * u_max])
#     # plt.ylim([0,L_y])
#
#     plt.title("t = " + str(round(t_list[c_index],4)))
plt.show()
