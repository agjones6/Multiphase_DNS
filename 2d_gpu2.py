#   !     /usr/bin/python -u This has the potential to change the python environment

#
# ANDY JONES
# MAE 577
#

# The goal of this is to change the 2d_base.py to be object base

# Importing pachages
import time
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import h5py
from numba import jit, cuda, vectorize, float64, float32, guvectorize

# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# =============================================================================
#                                 Functions
# =============================================================================
def dP_x_new_fun(C, dP_x, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (dP_x[2:,1:-1] + dP_x[:-2,1:-1] + dP_x[1:-1,2:] + dP_x[1:-1,:-2]
                  - (rho * h/dt)*(
                    (u_ishift_star[2:,1:-1] + u_ishift_star[1:-1,1:-1])
                  - (u_ishift_star[1:-1,1:-1] + u_ishift_star[:-2,1:-1])
                  + (v_ishift_star[2:,1:-1] + v_ishift_star[1:-1,1:-1])
                  - (v_ishift_star[2:,:-2] + v_ishift_star[1:-1,:-2])))

def dP_y_new_fun(C, dP_y, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (dP_y[1:-1,2:] + dP_y[1:-1,:-2] + dP_y[2:,1:-1] + dP_y[:-2,1:-1]
                  - (rho * h/dt)*(
                    (v_ishift_star[1:-1,2:]   + v_ishift_star[1:-1,1:-1])
                  - (v_ishift_star[1:-1,1:-1]     + v_ishift_star[1:-1,:-2])
                  + (u_ishift_star[1:-1,2:]   + u_ishift_star[1:-1,1:-1])
                  - (u_ishift_star[:-2,2:] + u_ishift_star[:-2,1:-1])))

def make_plot(mp, x, y, u, v, **kwargs):
    # Handling optional arguments
    LB = kwargs.get("LB",[])
    UB = kwargs.get("UB",[])
    plot_type = kwargs.get("plot_type","profile")
    sub_type = kwargs.get("sub_type",["u","y"])
    show_0 = kwargs.get("show_0","")

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
        if show_0 == "x":
            lim_x = mp.get_xlim()
            lim_y = mp.get_ylim()
            if lim_x[0] <= 0 and lim_x[1] >=0:
                mp.plot([0,0],[lim_y[0],lim_y[1]],'--k')
        elif show_0 == "y":
            lim_x = mp.get_xlim()
            lim_y = mp.get_ylim()
            if lim_y[0] <= 0 and lim_y[1] >=0:
                mp.plot([lim_x[0],lim_x[1]],[0,0],'--k')

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
    map[map=="p"] = 1
    map[map=="o"] = 2

    # Getting the wall number
    map[map == "w"] = 3
    map[map == "w_0"] = 4
    map[map == "w_1"] = 5
    map[map == "w_2"] = 6
    map[map == "w_3"] = 7

    # SOurce
    map[map == "s"] = 8


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
        elif "outflow" in arg:
            return "o"
        elif "source" in arg:
            return "s"
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
    map[:,-1] = my_str(top)
    map[:,0] = my_str(bottom)

    return map

def set_ghost(map, u, u_B=0, **kwargs):

    type = kwargs.get("type","velocity")
    source = kwargs.get("source",0)

    bound_type = 1
    try:
        if len(u_B) == 2:
            bound_type = 2
    except:
        bound_type = 1
        pass

    # Checking to make sure the velocity array passed in is 2D
    if len(u.shape) == 2:
        num_i, num_j = u.shape
        num_i = num_i - 1
        num_j = num_j - 1

    def find_fluid(map,loc,**kwargs):
        adj = kwargs.get("adj",0)

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

        if x > x_min:
            left = map[left_i]
        else:
            left = ""

        if x < x_max:
            right = map[right_i]
        else:
            right = ""

        bound_array = np.array([top,bottom,left,right])
        index_array = np.array([top_i, bot_i, left_i, right_i])
        try:
            final_index = index_array[bound_array=="f"][0]
            final_index = (final_index[0]+(final_index[0]-x)*adj,final_index[1]+(final_index[1]-y)*adj)
        except:
            final_index = (x,y)

        return final_index

    for i in range(len(map[:,0])):
        for j in range(len(map[0,:])):
            # Getting the current cell type
            cm = map[i,j]

            # Setting the wall ghost cells
            if "w" in cm:
                if type.lower() == "pressure":
                    # Pressure gradient in the wall will always be 0
                        # NOTE THIS MAY NEED TO CHANGE
                    u[i,j] = 0
                else:
                    f_ind = find_fluid(map,[i,j])

                    # Using the fluid index to set the ghost cell value
                    if (i,j) == f_ind:
                        # This is for the unused corners of the domain
                        u[i,j] = 0
                    else:
                        # Checking to see if there are different walls
                        if len(cm.split("_")) > 1:
                            w_type = cm.split("_")[1]
                            w_type = int(w_type)
                            if len(u_B) != 0:
                                u[i,j] = 2*u_B[w_type] - u[f_ind]
                            else:
                                u[i,j] = 2*u_B - u[f_ind]
                        else:
                            if len(u_B) != 0:
                                u[i,j] = 2*u_B[0] - u[f_ind]
                            else:
                                u[i,j] = 2*u_B - u[f_ind]

            # Setting periodic Boundaries
            elif "p" in cm:
                f_ind = find_fluid(map,[i,j],adj=-2)
                x_i,y_i = f_ind
                if x_i > num_i:
                    x_i = x_i-num_i
                if y_i > num_j:
                    y_i = y_i-num_j
                    # print(f_ind[0]-num_i)
                if x_i < 0:
                    x_i = x_i + num_i
                if y_i < 0:
                    y_i = y_i + num_j

                u[i,j] = u[x_i,y_i]

            # Setting Outflow Boundaries
            elif "o" in cm:
                # The Property on the boundary is equal to the fluid near the boundary
                f_ind = find_fluid(map,[i,j])
                x_i,y_i = f_ind
                # print(cm, (i,j), "-->", f_ind)
                u[i,j] = u[x_i,y_i]

            # Setting the source term ghost cells
            elif "s" in cm:
                # Checking to see if there are more than one source terms
                if len(cm.split("_")) > 1:
                    s_type = cm.split("_")[1]
                    s_type = int(s_type)
                    if len(source) != 0:
                        u[i,j] = source[s_type]
                    else:
                        u[i,j] = source
                else:
                    if len(u_B) != 0:
                        u[i,j] = source[0]
                    else:
                        u[i,j] = source
    return u

class domain_class:
    def __init__(self, **kwargs):
        # Defining the Spacial Domain
        self.N_x = kwargs.get("N_x",60) # Number of nodes in the x direction
        self.N_y = kwargs.get("N_y",0) # Number of nodes in the y direction
        self.L_x = kwargs.get("L_x",0.02)  # [m]
        self.L_y = kwargs.get("L_y",0.01)  # [m]

        # Time Domain Variables
        self.dt = kwargs.get("dt", 1e-3)
        self.N_t = kwargs.get("N_t", 6e2)
        self.t = kwargs.get("t", 0)
        self.T  = kwargs.get("T", self.N_t * self.dt)

        # Wall Velocities [m/s]
        self.u_B = [0]
        self.v_B = [0]

        # Pressure Gradient
        self.dP_x = 0
        self.dP_y = 0

        # Initial Velocities
        self.u_init = 0
        self.v_init = 0

        # External Forces
        self.F_x = 0
        self.F_y = 0

        # Options for Stability
        self.check_dt = True
        self.lower_tolerance = 1e-10

        # %% ==========================================================================
        #                         Setting Initial Conditions
        # =============================================================================
        self.h = 0
        # Checking the step size to ensure it is compatible with both directions
        if self.N_y == 0:
            self.h = self.L_x/self.N_x # [m] spacial step size
            if self.L_y/self.h != self.L_y//self.h:
                self.N_x0 = self.N_x
                while self.L_y/self.h != self.L_y//self.h:
                    self.N_x += 1
                    self.h = self.L_x/self.N_x
                print("N_x was changed ", self.N_x0, " -> ", self.N_x)
            self.N_y = self.L_y//self.h

        # Checking the number of nodes in the x direction
        if self.N_x == 0:
            self.h = self.L_y/self.N_y # [m] spacial step size
            if self.L_x/self.h != self.L_x//self.h:
                self.N_y0 = self.N_y
                while self.L_y/self.h != self.L_y//self.h:
                    self.N_y += 1
                    self.h = self.L_y/self.N_y
                print("N_y was changed ", self.N_y0, " -> ", self.N_y)
            self.N_x = self.L_x//self.h

        # Making sure the length of the domain and step size are good
        if self.N_x != 0 and self.N_y != 0:
            if self.h == 0:
                self.h = self.L_x/self.N_x
            else:
                self.L_x = self.h * self.N_x
                self.L_y = self.h * self.N_y

        # Getting a mesh of x and y values
        # self.x_grid = np.arange(0-self.h/2,self.L_x+self.h/2,self.h)
        # self.y_grid = np.arange(0-self.h/2,self.L_y+self.h/2,self.h)
        self.x_grid = np.linspace(0-self.h/2,self.L_x+self.h/2,int(self.N_x + 2))
        self.y_grid = np.linspace(0-self.h/2,self.L_y+self.h/2,int(self.N_y + 2))

        print(len(self.x_grid),len(self.y_grid))
        # Converting the Number of Nodes to integers
        self.N_x = int(self.N_x)
        self.N_y = int(self.N_y)

        # Defining the wall boundary
        self.top   ="wall"
        self.bottom="wall"
        self.left  ="periodic"
        self.right ="periodic"
        self.set_bounds()

        # self.domain_map = set_boundary([self.N_x,self.N_y],
        #                                top   =self.top,
        #                                bottom=self.bottom,
        #                                left  =self.left,
        #                                right =self.right)

    def set_bounds(self):
        self.domain_map = set_boundary([self.N_x,self.N_y],
                                       top   =self.top,
                                       bottom=self.bottom,
                                       left  =self.left,
                                       right =self.right)

        self.C = calc_C(self.domain_map)

class flow_class:
    def __init__(self,domain_class):
        dc = domain_class
        # Intializing the velocity Array
        self.u = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.u_init
        self.v = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.v_init

        # Setting the boundaries on the initial velocity array
        self.u = set_ghost(dc.domain_map, self.u, dc.u_B,source=dc.u_S)
        self.v = set_ghost(dc.domain_map, self.v, dc.v_B,source=dc.v_S)

        # Setting the time variable
        self.t = dc.t

        # Setting the Pressure Gradients
        self.dP_x = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.dP_x
        self.dP_y = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.dP_y

        # Setting the boundaries on the initial Pressure Array
        self.dP_x = set_ghost(dc.domain_map, self.dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        self.dP_y = set_ghost(dc.domain_map, self.dP_y, dc.v_B, type="pressure",source=dc.dP_y_S)

def bound_list(map,loc,**kwargs):
    # Returns a list of the fluids on surrounding a given index in a given domain
    #   [top, bottom, left, right]
    adj = kwargs.get("adj",0)

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

    if x > x_min:
        left = map[left_i]
    else:
        left = ""

    if x < x_max:
        right = map[right_i]
    else:
        right = ""

    bound_array = np.array([top,bottom,left,right])

    return bound_array

def calc_C(map):
    x_len, y_len = map.shape
    x_len -= 1
    y_len -= 1

    C = np.ones(map.shape)
    for i in range(x_len):
        for j in range(y_len):
            c_bounds = bound_list(map,[i,j])
            my_bool_f = np.array("f" == c_bounds)
            my_bool_p = np.array("p" == c_bounds)
            C[i,j] = np.sum(my_bool_f) + np.sum(my_bool_p)

    return C
def check_conv(past, current, tol):

    diff = abs(past - current)
    if np.all(diff < tol):
        return True
    else:
        return False
def calc_pressure(dc, dP_x, dP_y, u_ishift_star, v_ishift_star):

    x_len, y_len = dc.domain_map.shape
    x_len -= 1
    y_len -= 1

    tol = 1e-4

    dP_x_new = np.zeros(dP_x.shape)
    dP_y_new = np.zeros(dP_y.shape)

    x_conv = False
    y_conv = False
    count = 0

    while not x_conv and not y_conv and count < 100:
        if not x_conv: # (i+1) = 2:, (i-1) = :-2, i = 1:-1
            dP_x_new[1:-1,1:-1] = dP_x_new_fun(dc.C, dP_x, dc.rho, dc.h, dc.dt, u_ishift_star, v_ishift_star)

        if not y_conv:
            dP_y_new[1:-1,1:-1] = dP_y_new_fun(dc.C, dP_y, dc.rho, dc.h, dc.dt, u_ishift_star, v_ishift_star)

        # Checking if the current time step converges with the new step
        x_conv = check_conv(dP_x, dP_x_new, tol)
        y_conv = check_conv(dP_y, dP_y_new, tol)

        if not x_conv:
            dP_x = dP_x_new
        if not y_conv:
            dP_y = dP_y_new

        count += 1
    if count > 5:
        print("pressure took ",count," to converge")
    return dP_x_new, dP_y_new

def A_x_fun(h,u,v): # (i+1) = 2:, (i-1) = :-2, i = 1:-1
    return (1/(4*h)) * ( (u[2:,1:-1] + u[1:-1,1:-1])**2 - (u[:-2,1:-1] + u[1:-1,1:-1])**2
                           + (u[1:-1,2:] + u[1:-1,1:-1]) * (1/4)*((v[1:-1,1:-1]+v[1:-1,2:]+v[2:,1:-1]+v[2:,2:]) + (v[1:-1,1:-1]+v[1:-1,2:]+v[:-2,1:-1]+v[:-2,2:]))
                           - (u[1:-1,1:-1] + u[1:-1,:-2]) * (1/4)*((v[1:-1,1:-1]+v[1:-1,:-2]+v[2:,1:-1]+v[2:,:-2]) + (v[1:-1,1:-1]+v[1:-1,:-2]+v[:-2,1:-1]+v[:-2,:-2])) )

def A_y_fun(h,u,v):
    return (1/(4*h)) * ( (v[1:-1,2:] + v[1:-1,1:-1])**2 - (v[1:-1,:-2] + v[1:-1,1:-1])**2
                          + (v[2:,1:-1] + v[1:-1,1:-1]) * (1/4)*((u[1:-1,1:-1]+u[1:-1,2: ]+u[2:,1:-1]+u[2:,2:]) + (u[1:-1,1:-1]+u[1:-1,2:]+u[:-2,1:-1]+u[:-2,2:]))
                          - (v[1:-1,1:-1] + v[:-2,1:-1]) * (1/4)*((u[1:-1,1:-1]+u[1:-1,:-2]+u[2:,1:-1]+u[2:,:-2]) + (u[1:-1,1:-1]+u[1:-1,:-2]+u[:-2,1:-1]+u[:-2,:-2])) )

def D_fun(h,vel):
    return (1/h**2) * ( vel[2:,1:-1] + vel[:-2,1:-1] + vel[1:-1,2:] + vel[1:-1,:-2] - 4*vel[1:-1,1:-1])

# @vectorize(['float32(float32, float32, float32)',
#             'float64(float64, float64, float64)'],
#            target='cuda')

def vel_star(vel, dt, A, nu, D):
    return vel + dt * ( -A + nu * D )

# @vectorize(['float64(float64)'],nopython=True) # ,target='cuda')
# @jit(["float64[:,:](float64[:,:])"], nopython=True, target="cpu")
def shift_values(vel_star):
    return (1/2) * ( vel_star[2:,1:-1] + vel_star[1:-1,1:-1] )

def vel_ishift_fun(vel_ishift_star, dt, rho, h, dP, F):
    return vel_ishift_star - (dt/(rho*h)) * dP * h + F
def u_new_fun(vel_shift):
    return (1/2) * (vel_shift[1:-1,1:-1] + vel_shift[:-2,1:-1])
def v_new_fun(vel_shift):
    return (1/2) * (vel_shift[1:-1,1:-1] + vel_shift[1:-1,:-2])
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
# N_y = 100 => Real Elapsed Time 74.65, Sim Time 75.841
pressure_solve = "gradient" # "constant_gradient"
output_file = "./Output/MB_26.h5"
show_progress = False
write_interval = 0.005
dt_multiplier = 0.5

dt_max = 0.1
dt_min = 1e-10

real_start_time = time.time()
elapsed_time = lambda st_t: time.time() - st_t

# Initializing the domain class
dc = domain_class(N_x=0,
                  N_y=100,
                  L_x=0.02,
                  L_y=0.04,
                  dt = 5e-6
                  # dP_x=dP_analytic THis is not used
                  )
dc.rho   = 1e3   # kg/m^3
dc.nu    = 1e-6  # m^2/s
dc.check_dt = True

# Setting initial pressure gradient
dc.dP_x = 0 #dP_analytic
dc.dP_y = 0

# Initial Velocities
dc.u_init = 0 #u_analytic_mean
dc.v_init = 0 #u_analytic_mean

# Setting the time
dc.T = 2
dc.N_t = dc.T/dc.dt

dc.top   = "wall"
dc.bottom = "wall"
dc.left  = "source"
dc.right = "source"
dc.set_bounds()

# Putting a blockage in the flow
width = 0.001
st_x    = int(dc.N_x//1.2 - (width//dc.h)*0.5)
en_x = int(st_x+width//dc.h)
height = 0.004
st_y    = 0 #int(dc.N_y//2 - (height//dc.h)*0.5)
en_y = int(st_y+height//dc.h)

width2 = 0.0015
st_x2 = int(dc.N_x//1.2 - ((width+width2)//dc.h)*0.5)
en_x2 = int(st_x2+(width2)//dc.h)
height2 = 0.0005
st_y2 = int(en_y)
en_y2 = int(st_y+(height+height2)//dc.h)

dc.domain_map[st_x:en_x,st_y:en_y] = "w"
dc.domain_map[st_x2:en_x2,st_y2:en_y2] = "w"

# Changing the wall numbers
dc.domain_map[dc.domain_map == "w"] = "w_0"
# dc.domain_map[:,-1] = "w_1"

# Changing Soure Numbers
dc.domain_map[dc.domain_map == "s"] = "s_0"
dc.domain_map[-1,:] = "s_1" # Right
# dc.domain_map[:,-1] = "s_1" # Top

# wall velocities
dc.u_B = [0, 0] # 4.69 is the target for
dc.v_B = [0, 0]

# Source Terms
dc.u_S    = [0, -0.05]
dc.v_S    = [0, 0]
dc.dP_x_S = [0, 0]
dc.dP_y_S = [0, 0]

# Getting a mesh of x and y values
# dc.x_grid = np.arange(0-dc.h/2,dc.L_x+dc.h/2,dc.h)
# dc.y_grid = np.arange(0-dc.h/2,dc.L_y+dc.h/2,dc.h)


# Initializing the flow class
fc = flow_class(dc)
u = fc.u
v = fc.v
t = fc.t
dP_x = fc.dP_x
dP_y = fc.dP_y

# Showing a picture of the domain
show_my_domain = False
if show_my_domain:
    print(dc.domain_map)
    # print(np.flip(dc.domain_map).T)
    # print(dc.domain_map.T)
    # plt.figure()
    # show_domain(dc.domain_map.T)
    # plt.show()
    exit()

# %% Applying the boundary conditions

# Creating lists to store all of the variables
u_list = [u]
v_list = [v]
t_list = [t]
dP_x_list = [dP_x]
dP_y_list = [dP_y]

# Pulling some values out of the domain class
x_vals = dc.x_grid
y_vals = dc.y_grid
N_x    = dc.N_x
N_y    = dc.N_y

A_x = np.zeros(u.shape, dtype=np.float64)
A_y = np.zeros(v.shape, dtype=np.float64)
D_x = np.zeros(u.shape, dtype=np.float64)
D_y = np.zeros(v.shape, dtype=np.float64)
u_star = np.zeros(u.shape, dtype=np.float64)
v_star = np.zeros(u.shape, dtype=np.float64)
u_ishift = np.zeros(u.shape, dtype=np.float64)
v_ishift = np.zeros(v.shape, dtype=np.float64)
u_ishift_star = np.zeros(u.shape, dtype=np.float64)
v_ishift_star = np.zeros(v.shape, dtype=np.float64)
u_new = np.zeros(u.shape, dtype=np.float64)
v_new = np.zeros(v.shape, dtype=np.float64)

# This is a counting variable for saving hdf5 file
save_count = 0

if show_progress:
    plt.figure()
    my_plot1 = plt.subplot(2,1,1)
    my_plot2 = plt.subplot(2,2,3)
    my_plot3 = plt.subplot(2,2,4)
    my_plot4 = plt.subplot(2,1,2)
while t < dc.T: # and not user_done:
    if show_progress:
        # --> VECTOR FIELD PLOT
        my_plot1.clear()
        make_plot(my_plot1, x_vals, y_vals,
                  u_list[-1][:,:].T,
                  v_list[-1][:,:].T,
                  LB=0,
                  UB=dc.L_y,
                  plot_type="field",
                  sub_type=[]
                  )
        # Axis labels and title
        my_plot1.set_title(str(round(t,6)))
        my_plot1.set_xlabel("x")
        my_plot1.set_ylabel("y")

        # --> PRESSURE GRADIENT FIELD
        my_plot4.clear()
        make_plot(my_plot4, x_vals, y_vals,
                  dP_x_list[-1][:,:].T,
                  dP_y_list[-1][:,:].T,
                  LB=0,
                  UB=dc.L_y,
                  plot_type="field",
                  sub_type=[]
                  )
        # Axis labels and title
        # my_plot1.set_title(str(round(t,6)))
        my_plot1.set_xlabel("x")
        my_plot1.set_ylabel("y")

        # # --> U DISTRIBITION PROFILE
        # my_plot2.clear()
        # make_plot(my_plot2, x_vals, y_vals,
        #           u_list[-1][1:-1,1:-1].T,
        #           v_list[-1][1:-1,1:-1].T,
        #           LB=0,
        #           UB=dc.L_y,
        #           plot_type="profile",
        #           sub_type=["u","y"],
        #           show_0="x"
        #           )
        #
        # # Axis labels and title
        # my_plot2.set_xlabel("x-velocity (->)")
        # my_plot2.set_ylabel("y")
        #
        # # --> V DISTRIBUTION PROFILE
        # my_plot3.clear()
        # make_plot(my_plot3, x_vals, y_vals,
        #           u_list[-1][1:-1,1:-1],
        #           v_list[-1][1:-1,1:-1],
        #           # LB=0,
        #           # UB=dc.L_x,
        #           plot_type="profile",
        #           sub_type=["x","v"],
        #           show_0="y"
        #           )
        #
        # # Axis labels and title
        # my_plot3.set_xlabel("y-velocity (^)")
        # my_plot3.set_ylabel("y")

        # Showing the plot
        plt.pause(0.001)

        # Showing the initial Conditions
        if t == 0:
            plt.pause(1e-4)
        # exit()

    # --> Checking time step if desired
    if dc.check_dt:
        def vel_mag_fun(u,v):
            vel_mag = ((u**2 + v**2)**(0.5))/2
            return vel_mag

        # Getting the magnitude of velocity at each point
        vel_mag = vel_mag_fun(u,v)

        # Max Velocity
        max_vel = np.max(vel_mag)

        # Ensuring the maximum velocity is not 0
        if max_vel != 0:
            dc.dt = (dc.h/max_vel) * dt_multiplier

        if dc.dt > dt_max:
            dc.dt = dt_max

        if dc.dt < dt_min:
            dc.dt = dt_min

    # Reseting values
    # A_x[:] = 0
    # A_y[:] = 0
    # D_x[:] = 0
    # D_y[:] = 0
    # u_star[:] = 0
    # v_star[:] = 0
    # u_ishift[:] = 0
    # v_ishift[:] = 0
    # u_ishift_star[:] = 0
    # v_ishift_star[:] = 0

    # Getting values for the predictor step
    A_x[1:-1,1:-1] = A_x_fun(dc.h,u,v)
    A_y[1:-1,1:-1] = A_y_fun(dc.h,u,v)
    D_x[1:-1,1:-1] = D_fun(dc.h, u)
    D_y[1:-1,1:-1] = D_fun(dc.h, v)

    # Predictor Step
    u_star = vel_star(u, dc.dt, A_x, nu, D_x) # u + dc.dt * ( -A_x + nu * D_x )
    v_star = vel_star(v, dc.dt, A_y, nu, D_y) #v + dc.dt * ( -A_y + nu * D_y )

    # Doing the boundaries on the star velocities
    u_star = set_ghost(dc.domain_map, u_star, dc.u_B, source=dc.u_S)
    v_star = set_ghost(dc.domain_map, v_star, dc.v_B, source=dc.v_S)

    # Shifting the velocitiy values
    u_ishift_star[1:-1,1:-1] = shift_values(u_star)
    # exit()
    v_ishift_star[1:-1,1:-1] = shift_values(v_star)

    # Calculating the new pressures if it is desired
    if pressure_solve == "gradient":
        dP_x, dP_y = calc_pressure(dc, dP_x, dP_y, u_ishift_star, v_ishift_star)
        dP_x = set_ghost(dc.domain_map,dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        dP_y = set_ghost(dc.domain_map,dP_y, dc.u_B, type="pressure",source=dc.dP_y_S)
    elif pressure_solve == "constant_gradient":
        dP_x = fc.dP_x
        dP_y = fc.dP_y

    # Calculating the new time velocities
    u_ishift = vel_ishift_fun(u_ishift_star, dc.dt, rho, dc.h, dP_x, dc.F_x)
    v_ishift = vel_ishift_fun(v_ishift_star, dc.dt, rho, dc.h, dP_y, dc.F_y)

    # Applying boundaries to the shifted velocities
    u_ishift = set_ghost(dc.domain_map,u_ishift,dc.u_B,source=dc.u_S)
    v_ishift = set_ghost(dc.domain_map,v_ishift,dc.v_B,source=dc.v_S)

    # Getting the unshifted values back out
    u_new[1:-1,1:-1] = u_new_fun(u_ishift)
    v_new[1:-1,1:-1] = v_new_fun(v_ishift)

    # Setting boundary Conditions and updating time
    u_new = set_ghost(dc.domain_map,u_new, dc.u_B,source=dc.u_S)
    v_new = set_ghost(dc.domain_map,v_new, dc.v_B,source=dc.v_S)
    u = u_new
    v = v_new
    t += dc.dt

    # Getting rid of very small numbers
    u[abs(u)<dc.lower_tolerance] = 0
    v[abs(v)<dc.lower_tolerance] = 0

    # Appending values to the lists for storage
    u_list.append(u_new)
    v_list.append(v_new)
    t_list.append(t)
    dP_x_list.append(dP_x)
    dP_y_list.append(dP_y)

    # Saving an HDF5 File
    if save_count * write_interval <= t or t == dc.T:
        dm = dc.domain_map
        hf = h5py.File(output_file, "w")
        hf.create_dataset("u", data=u_list)
        hf.create_dataset("v", data=v_list)
        hf.create_dataset("t", data=t_list)
        hf.create_dataset("dP_x", data=dP_x_list)
        hf.create_dataset("dP_y", data=dP_y_list)
        hf.create_dataset("x", data=x_vals)
        hf.create_dataset("y", data=y_vals)
        # hf.create_dataset("domain_map", data=dm)
        # hf.create_dataset("domain", data=dc)
        hf.close()
        save_count += 1
        print("")
        print("\tWrote Output at t = ", round(t,5))
        print("\tReal Elapsed Time =", elapsed_time(real_start_time))
        print("\tdt = ", dc.dt)
        print("")
    print(t)
print("Simulation Completed")
plt.show()

exit()
