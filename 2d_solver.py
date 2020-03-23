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

# NOTE: This code is going ot be set up to basically always carry 'ghost' cells from boundaries.
    #   This implies the matrices will have an extra dimension compared to the
    #   dimensions of the domain. ie (0,0) corresponds to a point outside the domain

# =============================================================================
#                                 Functions
# =============================================================================
my_target = "cpu"
data_type = "float64"
# (i+1) = 2:, (i-1) = :-2, i = 1:-1
@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def dP_x_new_fun(C, dP_x, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (dP_x[2:,1:-1] + dP_x[:-2,1:-1] + dP_x[1:-1,2:] + dP_x[1:-1,:-2]
                  - (rho * h/dt)*(
                    (u_ishift_star[2:,1:-1]   - u_ishift_star[1:-1,1:-1])
                  - (u_ishift_star[1:-1,1:-1] - u_ishift_star[:-2,1:-1])
                  + (v_ishift_star[2:,1:-1]   - v_ishift_star[1:-1,1:-1])
                  - (v_ishift_star[2:,:-2]    - v_ishift_star[1:-1,:-2])))

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def dP_y_new_fun(C, dP_y, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (dP_y[1:-1,2:] + dP_y[1:-1,:-2] + dP_y[2:,1:-1] + dP_y[:-2,1:-1]
                  - (rho * h/dt)*(
                    (v_ishift_star[1:-1,2:]   - v_ishift_star[1:-1,1:-1])
                  - (v_ishift_star[1:-1,1:-1] - v_ishift_star[1:-1,:-2])
                  + (u_ishift_star[1:-1,2:]   - u_ishift_star[1:-1,1:-1])
                  - (u_ishift_star[:-2,2:]    - u_ishift_star[:-2,1:-1]) ))

# Absolute pressure calculation
@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def P_new_fun(C, P, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (P[2:,1:-1] + P[:-2,1:-1] + P[1:-1,2:] + P[1:-1,:-2]
                  - (rho * h/dt)*(
                    (u_ishift_star[1:-1,1:-1] - u_ishift_star[:-2,1:-1])
                  + (v_ishift_star[1:-1,1:-1] - v_ishift_star[1:-1,:-2]) ))

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def A_x_fun(h,u,v): # (i+1) = 2:, (i-1) = :-2, i = 1:-1
    return ((1.0/(4.0*h)) * ( (u[2:,1:-1] + u[1:-1,1:-1])**2.0 - (u[:-2,1:-1] + u[1:-1,1:-1])**2.0
                           + (u[1:-1,2:] + u[1:-1,1:-1]) * (1.0/4.0)*((v[1:-1,1:-1]+v[1:-1,2:]+v[2:,1:-1]+v[2:,2:])) #  + (v[1:-1,1:-1]+v[1:-1,2:]+v[:-2,1:-1]+v[:-2,2:])
                           - (u[1:-1,1:-1] + u[1:-1,:-2]) * (1.0/4.0)*((v[1:-1,1:-1]+v[1:-1,:-2]+v[2:,1:-1]+v[2:,:-2]) ) )) #+ (v[1:-1,1:-1]+v[1:-1,:-2]+v[:-2,1:-1]+v[:-2,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def A_y_fun(h,u,v):
    return (1.0/(4.0*h)) * ( (v[1:-1,2:] + v[1:-1,1:-1])**2.0 - (v[1:-1,:-2] + v[1:-1,1:-1])**2.0
                          + (v[2:,1:-1] + v[1:-1,1:-1]) * (1.0/4.0)*((u[1:-1,1:-1]+u[1:-1,2: ]+u[2:,1:-1]+u[2:,2:]) ) #+ (u[1:-1,1:-1]+u[1:-1,2:]+u[:-2,1:-1]+u[:-2,2:])
                          - (v[1:-1,1:-1] + v[:-2,1:-1]) * (1.0/4.0)*((u[1:-1,1:-1]+u[1:-1,:-2]+u[2:,1:-1]+u[2:,:-2]) ) ) #+ (u[1:-1,1:-1]+u[1:-1,:-2]+u[:-2,1:-1]+u[:-2,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def D_fun(h,vel):
    return (1/h**2) * ( vel[2:,1:-1] + vel[:-2,1:-1] + vel[1:-1,2:] + vel[1:-1,:-2] - 4*vel[1:-1,1:-1])


@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," + data_type + "[:,:]," + data_type + "," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def vel_star(vel, dt, A, nu, D):
    return vel + dt * ( -A + nu * D )

# @guvectorize(["void(" + data_type + "[:,:]," + data_type + "[:,:])"],
#                 "(n,p)->(n,p)", nopython=True, target=my_target)
@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def u_shift_values(vel_star):
    # result = vel_star
    result = np.zeros(vel_star.shape)
    result[1:-1,1:-1] = (1/2) * ( vel_star[2:,1:-1] + vel_star[1:-1,1:-1] )
    return result

# @guvectorize(["void(" + data_type + "[:,:]," + data_type + "[:,:])"],
#                 "(n,p)->(n,p)", nopython=True, target=my_target)
@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def v_shift_values(vel_star):
    # result = vel_star
    result = np.zeros(vel_star.shape)
    result[1:-1,1:-1] = (1/2) * ( vel_star[1:-1,2:] + vel_star[1:-1,1:-1] )
    return result

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + ")"],
      nopython=True, target=my_target)
def vel_ishift_fun(vel_ishift_star, dt, rho, h, dP, F):
    return vel_ishift_star - (dt/(rho*h)) * dP * h + F # I took the h out of dP

@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def u_new_fun(vel_shift):
    return (1/2) * (vel_shift[1:-1,1:-1] + vel_shift[:-2,1:-1])

@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def v_new_fun(vel_shift):
    return (1/2) * (vel_shift[1:-1,1:-1] + vel_shift[1:-1,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def vel_mag_fun(u,v):
    vel_mag = ((u**2 + v**2)**(0.5))/2
    return vel_mag

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
    h = kwargs.get("h",1)

    bound_type = 1
    try:
        if len(u_B) == 2:
            bound_type = 2
    except:
        bound_type = 1
        pass

    # Setting an original value matrix
    u0 = u

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

        x_max -= 1
        y_max -= 1

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
            try:
                # This is for the corners
                # print(x,y,bound_array)
                final_index = [("w" not in b and "" != b) for b in bound_array]
                final_index = index_array[final_index][0]
                final_index = (final_index[0]+(final_index[0]-x)*adj,final_index[1]+(final_index[1]-y)*adj)
                # print(final_index)
            except:
                # print("didnt Work")
                final_index = (x,y)

        return final_index

    for i in range(len(map[:,0])):
        for j in range(len(map[0,:])):
            # Getting the current cell type
            cm = map[i,j]
            # print(i,j)

            if type.lower() == "pressure":
                # Pressure gradient in the wall will always be 0
                # NOTE THIS MAY NEED TO CHANGE

                # Setting periodic Boundaries
                if "p" in cm:
                    # f_ind1 = find_fluid(map,[i,j],adj=1)
                    # f_ind2 = find_fluid(map,[i,j])
                    # u[i,j] = (u0[f_ind1] - u0[f_ind2])*h + u0[f_ind2]

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

                    u[i,j] = u0[x_i,y_i]

                elif not "f" in cm:
                    u[i,j] = 0
                # if not "f" in cm:
                #     u[i,j] = 0
            else:
                # Setting the wall ghost cells
                if "w" in cm:
                    f_ind = find_fluid(map,[i,j])

                    # Using the fluid index to set the ghost cell value
                    # if (i,j) == f_ind:
                    #     # This is for the unused corners of the domain
                    #     u[i,j] = 0
                    # else:
                    # Checking to see if there are different walls
                    if len(cm.split("_")) > 1:
                        w_type = cm.split("_")[1]
                        w_type = int(w_type)
                        if len(u_B) != 0:
                            u[i,j] = 2*u_B[w_type] - u0[f_ind]
                        else:
                            u[i,j] = 2*u_B - u0[f_ind]
                    else:
                        if len(u_B) != 0:
                            u[i,j] = 2*u_B[0] - u0[f_ind]
                        else:
                            u[i,j] = 2*u_B - u0[f_ind]

                # Setting Outflow Boundaries
                elif "o" in cm:
                    # --> Simple approach where the ghost cell equals the nearest fluid value
                    # The Property on the boundary is equal to the fluid near the boundary
                    # f_ind = find_fluid(map,[i,j])
                    # x_i,y_i = f_ind
                    # u[i,j] = u0[x_i,y_i]

                    # --> Linear Extrapolation approach
                    f_ind1 = find_fluid(map,[i,j],adj=1)
                    f_ind2 = find_fluid(map,[i,j])
                    u[i,j] = (u0[f_ind1] - u0[f_ind2])*h + u0[f_ind2]

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

                    u[i,j] = u0[x_i,y_i]

            # Setting the source term ghost cells
            if "s" in cm:
                # Checking to see if there are more than one source terms
                if len(cm.split("_")) > 1:
                    s_type = cm.split("_")[1]
                    s_type = int(s_type)
                    if len(source) != 0:
                        u[i,j] = source[s_type]
                        # if type.lower() == "pressure":
                        #     print(i,j, source[s_type])
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

        # Data type
        self.data_type = kwargs.get("data_type","float64")

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

    def check_dtype(self):
        if self.data_type == "float64":
            self.h = np.float64(self.h)
            self.dt = np.float64(self.dt)
            self.t = np.float64(self.t)
            self.T = np.float64(self.T)
            self.rho = np.float32(self.rho)
            self.nu = np.float32(self.nu)
        elif self.data_type == "float32":
            self.h = np.float32(self.h)
            self.dt = np.float32(self.dt)
            self.t = np.float32(self.t)
            self.T = np.float32(self.T)
            self.rho = np.float32(self.rho)
            self.nu = np.float32(self.nu)

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

        # Setting the intial pressure values
        self.P = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.P

        # Setting the boundaries on the initial Pressure Arrays
        self.dP_x = set_ghost(dc.domain_map, self.dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        self.dP_y = set_ghost(dc.domain_map, self.dP_y, dc.v_B, type="pressure",source=dc.dP_y_S)
        self.P = set_ghost(dc.domain_map, self.P, dc.v_B, type="pressure",source=dc.P_S)

        if dc.data_type == "float64":
            self.u.astype(np.float64)
            self.v.astype(np.float64)
            self.t = np.float64(self.t)
            self.dP_x.astype(np.float64)
            self.dP_y.astype(np.float64)
            self.P.astype(np.float64)
        elif dc.data_type == "float32":
            self.u.astype(np.float32)
            self.v.astype(np.float32)
            self.t = np.float32(self.t)
            self.dP_x.astype(np.float32)
            self.dP_y.astype(np.float32)
            self.P.astype(np.float32)

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
    # x_len -= 1
    # y_len -= 1
    C = np.ones(map.shape)
    for i in range(x_len):
        for j in range(y_len):
            c_bounds = bound_list(map,[i,j])
            my_bool_f = np.array("f" == c_bounds)
            my_bool_p = np.array("p" == c_bounds)
            my_bool_o = np.array("o" == c_bounds)
            my_bool_s = np.array(["s" in b for b in c_bounds])
            C[i,j] = np.sum(my_bool_f)  + np.sum(my_bool_s) + np.sum(my_bool_p) # + np.sum(my_bool_o)
    return C

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:])"],nopython=True,target=my_target)
def calc_diff(past,current):
    return np.absolute(past - current)

def check_conv(past, current, tol):
    diff = calc_diff(past, current)
    if np.all(diff < tol):
        return True
    else:
        return False

def calc_pressure_grad(dc, dP_x, dP_y, u_ishift_star, v_ishift_star):

    x_len, y_len = dc.domain_map.shape
    x_len -= 1
    y_len -= 1

    tol = 1e-6

    dP_x_new = dP_x
    dP_y_new = dP_y

    x_conv = False
    y_conv = False
    count = 0

    while not x_conv and not y_conv and count < 100:
        if not x_conv: # (i+1) = 2:, (i-1) = :-2, i = 1:-1
            dP_x_new[1:-1,1:-1] = dP_x_new_fun(dc.C, dP_x, dc.rho, dc.h, dc.dt, u_ishift_star, v_ishift_star)

        if not y_conv:
            dP_y_new[1:-1,1:-1] = dP_y_new_fun(dc.C, dP_y, dc.rho, dc.h, dc.dt, u_ishift_star, v_ishift_star)

        # Checking if the current time step converges with the new step
        x_conv = check_conv(dP_x[1:-1,1:-1], dP_x_new[1:-1,1:-1], tol)
        y_conv = check_conv(dP_y[1:-1,1:-1], dP_y_new[1:-1,1:-1], tol)

        if not x_conv:
            dP_x = dP_x_new
        if not y_conv:
            dP_y = dP_y_new

        count += 1
    if count > 5:
        print("pressure took ",count," to converge")
    return dP_x_new, dP_y_new


def calc_pressure(dc, P, u_ishift_star, v_ishift_star):

    # Tolerance for convergence
    tol = 1e-6

    # Setting a dummy matrix to be changed
    P_new = P

    conv = False
    count = 0

    while not conv and count < 100:
        # Calculating the new Pressure scalar field
        P_new[1:-1,1:-1] = P_new_fun(dc.C, P, dc.rho, dc.h, dc.dt, u_ishift_star, v_ishift_star)

        # Checking if the current time step converges with the new step
        conv = check_conv(P[1:-1,1:-1], P_new[1:-1,1:-1], tol)

        P = P_new

        count += 1
    if count > 1:
        print("pressure took ",count," to converge")

    return P_new
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
#                           Defining Simulation
# =============================================================================
pressure_solve = "value"
output_file = "./Output/hwk4/run_27.h5"
show_progress = False
write_interval = 120
dt_multiplier = 0.5

# Minimum and Maximum Time Step
dt_max = 0.05
dt_min = 1e-30

# Number of bytes that is allowed to be stored locally
max_size = 10e6

real_start_time = time.time()
elapsed_time = lambda st_t: time.time() - st_t

# Initializing the domain class
dc = domain_class(N_x=0,
                  N_y=50,
                  L_x=0.10,
                  L_y=0.05,
                  dt = 5e-6,
                  data_type=data_type
                  # dP_x=dP_analytic THis is not used
                  )
dc.rho   = 1e3   # kg/m^3
dc.nu    = 1e-6  # m^2/s
dc.check_dt = True

dc.check_dtype()
# print(dc.h)
# exit()
# Setting initial pressure gradient
dc.dP_x = 0.0
dc.dP_y = 0.0
dc.P = 101325 # atmospheric pressure in pascals

# Initial Velocities
dc.u_init = 0.01 #0.03 #u_analytic_mean
dc.v_init = 0. #u_analytic_mean

# Setting the time
dc.T = 5
dc.N_t = dc.T/dc.dt

dc.top   = "wall"
dc.bottom = "wall"
dc.left  = "periodic"
dc.right = "periodic"
dc.set_bounds()
# print(dc.C.T)
# print(dc.domain_map.T)
# exit()
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
# dc.domain_map[dc.domain_map == "s"] = "s_0"
# dc.domain_map[-1,:] = "s_1" # Right
# dc.domain_map[:,-1] = "s_1" # Top

# wall velocities
dc.u_B = [0, 0] # 4.69 is the target for
dc.v_B = [0, 0]

# Source Terms
dc.u_S    = [0., 0.]
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
u = fc.u
v = fc.v
t = fc.t
dP_x = fc.dP_x
dP_y = fc.dP_y
P = fc.P

# %% Applying the boundary conditions
if dc.data_type == "float64":
    numpy_dtype = np.float64
elif dc.data_type == "float32":
    numpy_dtype = np.float32

numpy_dtype = np.float32
# Creating lists to store all of the variables
u_list = np.array(u, dtype=numpy_dtype)
v_list = np.array(v, dtype=numpy_dtype)
t_list = np.array(t, dtype=numpy_dtype)
dP_x_list = np.array(dP_x, dtype=numpy_dtype)
dP_y_list = np.array(dP_y, dtype=numpy_dtype)
P_list = np.array(P, dtype=numpy_dtype)

# Pulling some values out of the domain class
x_vals = dc.x_grid
y_vals = dc.y_grid
N_x    = dc.N_x
N_y    = dc.N_y

numpy_dtype = np.float64
A_x = np.zeros(u.shape, dtype=numpy_dtype)
A_y = np.zeros(v.shape, dtype=numpy_dtype)
D_x = np.zeros(u.shape, dtype=numpy_dtype)
D_y = np.zeros(v.shape, dtype=numpy_dtype)
u_star = np.zeros(u.shape, dtype=numpy_dtype)
v_star = np.zeros(v.shape, dtype=numpy_dtype)
u_ishift = np.zeros(u.shape, dtype=numpy_dtype)
v_ishift = np.zeros(v.shape, dtype=numpy_dtype)
u_ishift_star = np.zeros(u.shape, dtype=numpy_dtype)
v_ishift_star = np.zeros(v.shape, dtype=numpy_dtype)
u_new = np.zeros(u.shape, dtype=numpy_dtype)
v_new = np.zeros(v.shape, dtype=numpy_dtype)

# This is a counting variable for saving hdf5 file
save_count = 0

if show_progress:
    plt.figure()
    my_plot1 = plt.subplot(2,1,1)
    my_plot4 = plt.subplot(2,1,2)

while t < dc.T: # and not user_done:
    # Showing the current velocity and pressure fields
    if show_progress:
        # --> VECTOR FIELD PLOT
        my_plot1.clear()
        make_plot(my_plot1, x_vals, y_vals,
                  u[:,:].T,
                  v[:,:].T,
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
                  dP_x[:,:].T,
                  dP_y[:,:].T,
                  LB=0,
                  UB=dc.L_y,
                  plot_type="field",
                  sub_type=[]
                  )
        # Axis labels and title
        # my_plot1.set_title(str(round(t,6)))
        my_plot1.set_xlabel("x")
        my_plot1.set_ylabel("y")

        # Showing the plot
        plt.pause(0.001)

        # Showing the initial Conditions
        if t == 0:
            plt.pause(1e-4)
        # exit()

    # --> Checking time step if desired
    if dc.check_dt:

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
    u_ishift_star = u_shift_values(u_star)
    v_ishift_star = v_shift_values(v_star)

    # u_ishift_star = set_ghost(dc.domain_map, u_ishift_star, dc.u_B, source=dc.u_S)
    # v_ishift_star = set_ghost(dc.domain_map, v_ishift_star, dc.v_B, source=dc.v_S)

    # Calculating the new pressures if it is desired
    if pressure_solve == "gradient":
        # dP_x = set_ghost(dc.domain_map,dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        # dP_y = set_ghost(dc.domain_map,dP_y, dc.u_B, type="pressure",source=dc.dP_y_S)

        dP_x, dP_y = calc_pressure_grad(dc, dP_x, dP_y, u_ishift_star, v_ishift_star)
        dP_x = set_ghost(dc.domain_map,dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        dP_y = set_ghost(dc.domain_map,dP_y, dc.u_B, type="pressure",source=dc.dP_y_S)
        # dP_x[:,:] = 0
    elif pressure_solve.lower() == "value":
        P = set_ghost(dc.domain_map, P, dc.u_B, h=dc.h, type="pressure",source=dc.P_S)
        # plt.figure()
        # plt.contourf(x_vals,y_vals,P.T/np.max(P.T))
        P = calc_pressure(dc, P, u_ishift_star, v_ishift_star)

        # plt.figure()
        # plt.contourf(x_vals,y_vals,P.T)#/np.max(P.T))
        dP_x[1:-1,1:-1] = (P[2:,1:-1] - P[1:-1,1:-1])/dc.h
        dP_y[1:-1,1:-1] = (P[1:-1,2:] - P[1:-1,1:-1])/dc.h
        dP_x = set_ghost(dc.domain_map,dP_x, dc.u_B, type="pressure",source=dc.dP_x_S)
        dP_y = set_ghost(dc.domain_map,dP_y, dc.u_B, type="pressure",source=dc.dP_y_S)

        dP_x[-2,:] = 0
        dP_x[1,:] = 0
        dP_x[:,-2] = 0
        dP_x[:,1] = 0
        dP_y[-2,:] = 0
        dP_y[1,:] = 0
        dP_y[:,-2] = 0
        dP_y[:,1] = 0

        # for ln in dP_x.T:
        #     print(ln)

        # plt.show()
        # plt.pause(0.001)
        # exit()
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
    u[1:-1,1:-1] = u_new_fun(u_ishift)
    v[1:-1,1:-1] = v_new_fun(v_ishift)

    # Setting boundary Conditions and updating time
    # u = set_ghost(dc.domain_map, u, dc.u_B, source=dc.u_S)
    # v = set_ghost(dc.domain_map, v, dc.v_B, source=dc.v_S)
    t += dc.dt

    # Getting rid of very small numbers
    u[abs(u)<dc.lower_tolerance] = 0
    v[abs(v)<dc.lower_tolerance] = 0

    # Appending values to the lists for storage
    u_list = np.dstack((u_list, u))
    v_list = np.dstack((v_list, v))
    t_list = np.append(t_list, t)
    dP_x_list = np.dstack((dP_x_list, dP_x))
    dP_y_list = np.dstack((dP_y_list, dP_y))
    P_list = np.dstack((P_list, P))

    # Calculating the size of all of the arrays
    current_size = u_list.nbytes + v_list.nbytes + dP_x_list.nbytes + dP_y_list.nbytes + P_list.nbytes

    # Saving an HDF5 File if the size of the arrays reaches a certain number of bytes
    #   or the write interval is reached
    if save_count * write_interval <= t or t >= dc.T or current_size >= max_size:
        if save_count == 0:
            # --> This occurs on the first save and creates the file
            hf = h5py.File(output_file, "w")
            # These change every time loop
            hf.create_dataset("u", data=u_list, maxshape=(u.shape[0],u.shape[1],None), compression="gzip", chunks=True)
            hf.create_dataset("v", data=v_list, maxshape=(v.shape[0],v.shape[1],None), compression="gzip", chunks=True)
            hf.create_dataset("dP_x", data=dP_x_list, maxshape=(u.shape[0],u.shape[1],None), compression="gzip", chunks=True)
            hf.create_dataset("dP_y", data=dP_y_list, maxshape=(v.shape[0],v.shape[1],None), compression="gzip", chunks=True)
            hf.create_dataset("P", data=P_list, maxshape=(u.shape[0],u.shape[1],None), compression="gzip", chunks=True)
            hf.create_dataset("t", data=t_list, maxshape=(None,), compression="gzip", chunks=True)

            hf.create_dataset("x", data=x_vals, maxshape=x_vals.shape, compression="gzip")
            hf.create_dataset("y", data=y_vals, maxshape=y_vals.shape, compression="gzip")
            hf.close()
            u_list = u_list[:,:,-1]
            v_list = v_list[:,:,-1]
            dP_x_list = dP_x_list[:,:,-1]
            dP_y_list = dP_y_list[:,:,-1]
            t_list = [-1]
        else:
            # --> This opens the file and appends to the current arrays
            hf = h5py.File(output_file, "a")

            # Pulling in and resizing the arrays
            hf["u"].resize(hf["u"].shape[2] + u_list.shape[2]-1, axis=2)
            hf["u"][:,:,-(u_list.shape[2]-1):] = u_list[:,:,1:]
            u_list = u_list[:,:,-1]

            hf["v"].resize(hf["v"].shape[2] + v_list.shape[2]-1, axis=2)
            hf["v"][:,:,-(v_list.shape[2]-1):] = v_list[:,:,1:]
            v_list = v_list[:,:,-1]

            hf["dP_x"].resize(hf["dP_x"].shape[2] + dP_x_list.shape[2]-1, axis=2)
            hf["dP_x"][:,:,-(dP_x_list.shape[2]-1):] = dP_x_list[:,:,1:]
            dP_x_list = dP_x_list[:,:,-1]

            hf["dP_y"].resize(hf["dP_y"].shape[2] + dP_y_list.shape[2]-1, axis=2)
            hf["dP_y"][:,:,-(dP_y_list.shape[2]-1):] = dP_y_list[:,:,1:]
            dP_y_list = dP_y_list[:,:,-1]

            hf["P"].resize(hf["P"].shape[2] + P_list.shape[2]-1, axis=2)
            hf["P"][:,:,-(P_list.shape[2]-1):] = P_list[:,:,1:]
            P_list = P_list[:,:,-1]

            hf["t"].resize(hf["t"].shape[0] + t_list.shape[0]-1, axis=0)
            hf["t"][-(t_list.shape[0]-1):] = t_list[1:]
            t_list = t_list[-1]

            hf.close()

        save_count += 1
        print("\n")
        print("\tWrote Output at t = ", round(t,5))
        print("\tReal Elapsed Time =", elapsed_time(real_start_time))
        print("\tdt = ", dc.dt)
        print("")

    # print(t) # + t_list.nbytes)
    # print(u_list[-1])
print("Simulation Completed")
exit()
