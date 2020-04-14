#
# ANDY JONES
# MAE 577
# Fluid Simulation Functions Code
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
import copy
# import cv2

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
@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def P_new_fun(C, P, rho, h, dt, u_ishift_star, v_ishift_star):
    return (1/C[1:-1,1:-1]) * (P[2:,1:-1] + P[:-2,1:-1] + P[1:-1,2:] + P[1:-1,:-2]
                  - (rho[1:-1,1:-1] * h/dt)*(
                    (u_ishift_star[1:-1,1:-1] - u_ishift_star[:-2,1:-1])
                  + (v_ishift_star[1:-1,1:-1] - v_ishift_star[1:-1,:-2]) ))

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def A_x_fun(h,u,v): # (i+1/2) = 1:-1, (i+3/2) = 2:, (i-1/2) = :-2
# THis function now is using SHIFTED velocity values
    return ((1.0/(4.0*h)) * ( (u[2:,1:-1] + u[1:-1,1:-1])**2.0 - (u[:-2,1:-1] + u[1:-1,1:-1])**2.0
                           + (u[1:-1,2:] + u[1:-1,1:-1]) * (v[2:,1:-1] + v[1:-1,1:-1]) #  + (v[1:-1,1:-1]+v[1:-1,2:]+v[:-2,1:-1]+v[:-2,2:])
                           - (u[1:-1,1:-1] + u[1:-1,:-2]) *(v[2:,:-2] + v[1:-1,:-2]) )) #+ (v[1:-1,1:-1]+v[1:-1,:-2]+v[:-2,1:-1]+v[:-2,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def A_y_fun(h,u,v):
    return (1.0/(4.0*h)) * ( (v[1:-1,2:] + v[1:-1,1:-1])**2.0 - (v[1:-1,:-2] + v[1:-1,1:-1])**2.0
                          + (v[2:,1:-1] + v[1:-1,1:-1]) * (u[1:-1,1:-1] + u[1:-1,2:] ) #+ (u[1:-1,1:-1]+u[1:-1,2:]+u[:-2,1:-1]+u[:-2,2:])
                          - (v[1:-1,1:-1] + v[:-2,1:-1]) *(u[:-2,2:] + u[:-2,1:-1] ) ) #+ (u[1:-1,1:-1]+u[1:-1,:-2]+u[:-2,1:-1]+u[:-2,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def D_fun(h,vel):
    return (1/h**2) * ( vel[2:,1:-1] + vel[:-2,1:-1] + vel[1:-1,2:] + vel[1:-1,:-2] - 4*vel[1:-1,1:-1])


@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," +
     data_type + "[:,:]," + data_type + "[:,:]," + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def vel_star(vel, dt, A, nu, D,F_gamma):
    return vel + dt * ( -A + nu * D + F_gamma )

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," + data_type + "[:,:]," + data_type + "," + data_type + "[:,:]," + data_type + ")"],
      nopython=True, target=my_target)
def vel_ishift_fun(vel_ishift_star, dt, rho, h, dP, F):
    return vel_ishift_star - (dt/(rho*h)) * dP * h + F # I took the h out of dP

@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def u_new_fun(vel_shift):
    return (1/2) * (vel_shift[:-2,1:-1] + vel_shift[1:-1,1:-1])

@jit(["" + data_type + "[:,:](" + data_type + "[:,:])"],
      nopython=True, target=my_target)
def v_new_fun(vel_shift):
    return (1/2) * (vel_shift[1:-1,1:-1] + vel_shift[1:-1,:-2])

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def vel_mag_fun(u,v):
    vel_mag = ((u**2 + v**2)**(0.5))/2
    return vel_mag

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "[:,:])"],nopython=True,target=my_target)
def calc_diff(past,current):
    return np.absolute(past - current)

@jit(["boolean(float64[:,:],float64[:,:],float64)"],nopython=True,target=my_target)
def check_conv(past, current, tol):
    diff = np.absolute(past - current) #calc_diff(past, current)
    if np.all(diff < tol):
        return True
    else:
        return False

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

    map[:,-1] = my_str(top)
    map[:,0] = my_str(bottom)
    map[0,:] = my_str(left)
    map[-1,:] = my_str(right)

    map = np.array(map)
    return map

# Checking to see if there are different options attached to a string
def get_num_option(cm,val_arr):
    # If an optional number is given with an underscore
    if len(cm.split("_")) > 1:
        w_type = cm.split("_")[1]
        w_type = int(w_type)
        # If there is an array of options
        if len(val_arr) != 0:
            bound_val = val_arr[w_type]
        else:
            bound_val = val_arr
    else:
        if len(val_arr) != 0:
            bound_val = val_arr[0]
        else:
            bound_val = val_arr
    return bound_val

def shift_dm(map,direction):

    if direction == "i":
        # Starts out as a smaller version of the actual map
        new_map = map[:-1,:]

        # Getting a top and bottom shifted map
        top_map = map[1:,:]
        bot_map = map[:-1,:]

        # Getting the logicals of where the walls are
        w_log_top = str_index(map[1:,:],"w")
        w_log_bot = str_index(map[:-1,:],"w")

        # Changing the value of the new map if the logical is true
        new_map[w_log_top] = top_map[w_log_top]
        new_map[w_log_bot] = bot_map[w_log_bot]

        # Getting the map edge to put on the new map
        edge_map = map[-1,:]
        new_map = np.row_stack((new_map,edge_map))
    elif direction == "j":
        # Starts out as a smaller version of the actual map
        new_map = map[:,:-1]

        # Getting a top and bottom shifted map
        top_map = map[:,1:]
        bot_map = map[:,:-1]

        # Getting the logicals of where the walls are
        w_log_top = str_index(map[:,1:],"w")
        w_log_bot = str_index(map[:,:-1],"w")

        # Changing the value of the new map if the logical is true
        new_map[w_log_top] = top_map[w_log_top]
        new_map[w_log_bot] = bot_map[w_log_bot]

        # Getting the map edge to put on the new map
        edge_map = map[:,-1]
        new_map = np.column_stack((new_map,edge_map))
    else:
        print("Please Enter a Correct Direction Option")
        return None

    return new_map

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

def find_fluid(map,loc,**kwargs):
    adj = kwargs.get("adj",0)
    bound_array = kwargs.get("bound_arry",bound_list(map,loc,adj=adj))
    #
    x = loc[0]
    y = loc[1]

    top_i = (x,y+1)
    bot_i = (x,y-1)
    right_i = (x+1,y)
    left_i = (x-1,y)

    index_array = np.array([top_i, bot_i, left_i, right_i])
    try:
        final_index = index_array[bound_array=="f"][0]
        final_index = (final_index[0]+(final_index[0]-x)*adj,final_index[1]+(final_index[1]-y)*adj)
    except:
        try:
            # This is for the corners
            # print(x,y,bound_array)
            # final_index = [("w" not in b and "" != b) for b in bound_array]
            # final_index = index_array[final_index][0]
            # final_index = (final_index[0]+(final_index[0]-x)*adj,final_index[1]+(final_index[1]-y)*adj)
            final_index = (x,y)
            # print("here",(x,y))
            # print(bound_array)
        except:
            # print("didnt Work")
            final_index = (x,y)

    return final_index

def extrap(map,ind,val0,mult=1):
    i = ind[0]
    j = ind[1]
    bound_array = bound_list(map,[i,j])
    f_ind1 = find_fluid(map,[i,j],bound_array=bound_array,adj=1) # One more in from the boundary
    f_ind2 = find_fluid(map,[i,j],bound_array=bound_array)
    # print((i,j),f_ind1,f_ind2)
    return mult*(val0[f_ind1] - val0[f_ind2]) + val0[f_ind2]

def set_ghost(map, val, u_B=0, **kwargs):

    type = kwargs.get("type","u")
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
    val0 = np.copy(val)

    # Checking to make sure the velocity array passed in is 2D
    if len(val.shape) == 2:
        num_i, num_j = val.shape
        num_i = num_i - 1
        num_j = num_j - 1

    # Getting the mininum and maximum x/y values
    x_max,y_max = map.shape
    x_min,y_min = 0,0

    x_max -= 1
    y_max -= 1

    # This pulls the indices of interest into arrays. That way only the non-fluid cells are checked
    voi_indices = np.where(np.logical_not(str_index(map,"f")))
    voi_i = voi_indices[0]
    voi_j = voi_indices[1]
    for ind in range(len(voi_i)):
        i = voi_i[ind]
        j = voi_j[ind]

        # Skipping corners
        if i == x_max and (j == y_min or j == y_max):
            # print(i,j)
            continue
        if j == y_max and (i == x_min or i == x_max):
            # print(i,j)
            continue

        # for j in voi_j:
        # Getting the current cell type
        cm = map[i,j]
        # print(i,j)

        if type.lower() == "p":
            # Pressure gradient in the wall will always be 0
            # NOTE THIS MAY NEED TO CHANGE

            # Setting periodic Boundaries
            if "p" in cm:
                # f_ind1 = find_fluid(map,[i,j],adj=1)
                # f_ind2 = find_fluid(map,[i,j])
                # val[i,j] = (val0[f_ind1] - val0[f_ind2])*h + val0[f_ind2]
                # This needs to change two pressure cells in order to be stable
                f_ind = find_fluid(map,[i,j],adj= -1)
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

                # print((i,j), "-> ", (x_i,y_i))
                val[i,j] = val0[x_i,y_i]

            # Setting Outflow Boundaries
            elif "o" in cm:
                # --> Linear Extrapolation approach
                val[i,j] = 101325 #- (0.02)*-2.4 #extrap(map,[i,j],val0,mult=1)

            elif not "f" in cm:
                val[i,j] = 0



            # if not "f" in cm:
            #     val[i,j] = 0

        elif type.lower() == "psi":
            # Setting periodic Boundaries
            if "p" in cm:# or "o" in cm:
                f_ind = find_fluid(map,[i,j],adj=-2)
                x_i,y_i = f_ind
                if x_i > num_i:
                    x_i = x_i-num_i
                if y_i > num_j:
                    y_i = y_i-num_j
                if x_i < 0:
                    x_i = x_i + num_i
                if y_i < 0:
                    y_i = y_i + num_j

                val[i,j] = val0[x_i,y_i]

            elif "w" in cm:
                f_ind = find_fluid(map,[i,j],adj=1)
                x_i,y_i = f_ind
                if x_i > num_i:
                    x_i = x_i-num_i
                if y_i > num_j:
                    y_i = y_i-num_j
                if x_i < 0:
                    x_i = x_i + num_i
                if y_i < 0:
                    y_i = y_i + num_j

                val[i,j] = val0[x_i,y_i]


        elif type.lower() == "u":
            # Setting the wall ghost cells
            if "w" in cm:
                # Getting the boundary velocity
                bound_vel = get_num_option(cm,u_B)

                # Getting the surrouding cell mao values
                bound_array = bound_list(map,[i,j]) #[top,bottom,left,right]

                # For u velocity, if the wall is on the left or right, the velicity is equal to the boundary
                if ("f" in bound_array[0] or "f" in bound_array[1]) and ("f" not in bound_array[2] and "f" not in bound_array[3]):
                    f_ind = find_fluid(map,[i,j],bound_array=bound_array)
                    val[i,j] = 2*bound_vel - val0[f_ind]
                else:
                    val[i,j] = bound_vel

            # Setting periodic Boundaries
            elif "p" in cm:# or "o" in cm:
                f_ind = find_fluid(map,[i,j],adj=-2)
                x_i,y_i = f_ind
                if x_i > num_i:
                    x_i = x_i-num_i
                if y_i > num_j:
                    y_i = y_i-num_j
                if x_i < 0:
                    x_i = x_i + num_i
                if y_i < 0:
                    y_i = y_i + num_j

                val[i,j] = val0[x_i,y_i]

            # Setting Outflow Boundaries
            elif "o" in cm:
                # --> Linear Extrapolation approach
                val[i,j] = extrap(map,[i,j],val0,mult=1)

            # Setting the source term ghost cells
            elif "s" in cm:
                bound_val = get_num_option(cm,source)
                val[i,j] = bound_val

        elif type.lower() == "v":
            # Setting the wall ghost cells
            if "w" in cm:
                # Getting the boundary velocity
                bound_vel = get_num_option(cm,u_B)

                # Getting the surrouding cell mao values
                bound_array = bound_list(map,[i,j]) #[top,bottom,left,right]

                # For u velocity, if the wall is on the left or right, the velicity is equal to the boundary
                if ("f" in bound_array[2] or "f" in bound_array[3]) and ("f" not in bound_array[0] and "f" not in bound_array[1]):
                    f_ind = find_fluid(map,[i,j],bound_array=bound_array)
                    val[i,j] = 2*bound_vel - val0[f_ind]
                else:
                    val[i,j] = bound_vel

            # Setting Outflow Boundaries
            elif "o" in cm:
                # --> Simple approach where the ghost cell equals the nearest fluid value
                # The Property on the boundary is equal to the fluid near the boundary
                # f_ind = find_fluid(map,[i,j])
                # x_i,y_i = f_ind
                # val[i,j] = val0[x_i,y_i]

                # --> Linear Extrapolation approach
                val[i,j] = extrap(map,[i,j],val0,mult=1)

            # Setting periodic Boundaries
            elif "p" in cm:# or "o" in cm:
                f_ind = find_fluid(map,[i,j],adj=-2)
                x_i,y_i = f_ind
                if x_i > num_i:
                    x_i = x_i-num_i
                if y_i > num_j:
                    y_i = y_i-num_j
                if x_i < 0:
                    x_i = x_i + num_i
                if y_i < 0:
                    y_i = y_i + num_j

                val[i,j] = val0[x_i,y_i]

            # Setting the source term ghost cells
            elif "s" in cm:
                bound_val = get_num_option(cm,source)
                val[i,j] = bound_val

            # # Checking to see if there are more than one source terms
            # if len(cm.split("_")) > 1:
            #     s_type = cm.split("_")[1]
            #     s_type = int(s_type)
            #     if len(source) != 0:
            #         val[i,j] = source[s_type]
            #         # if type.lower() == "pressure":
            #         #     print(i,j, source[s_type])
            #     else:
            #         val[i,j] = source
            # else:
            #     if len(u_B) != 0:
            #         val[i,j] = source[0]
            #     else:
            #         val[i,j] = source

    return val

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
            C[i,j] = np.sum(my_bool_f)  + np.sum(my_bool_p) + np.sum(my_bool_o) # + np.sum(my_bool_s)
    C[C==0] = 1
    return C

def calc_pressure_grad(dc, dP_x, dP_y, u_ishift_star, v_ishift_star):

    x_len, y_len = dc.domain_map.shape
    x_len -= 1
    y_len -= 1

    tol = 1e-6

    dP_x_new = copy.copy(dP_x)
    dP_y_new = copy.copy(dP_y)

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

@jit(["float64[:,:](float64,int32,int32,float64[:,:],float64[:,:],float64[:,:]," +
    "float64[:,:],float64[:,:],float64,float64,float64[:,:],float64[:,:])"],nopython=True,target=my_target)
def loop_Pressure(tol,max_loops,min_loops,wall_0s,P_old,P_new, C, rho, h, dt, u_ishift_star, v_jshift_star):
    # Values to initialize the loop
    conv = False
    count = 0

    while not conv and count < max_loops or count <= min_loops:
        # Calculating the new Pressure scalar field
        P_new[1:-1,1:-1] = P_new_fun(C, P_old, rho, h, dt, u_ishift_star, v_jshift_star)

        # Setting the pressure in the walls to 0
        P_new = np.absolute(P_new)*wall_0s

        # Checking if the current time step converges with the new step
        conv = check_conv(P_old[1:-1,1:-1], P_new[1:-1,1:-1], tol)

        # Normalizing the pressure back to absolute
        # P_new = np.absolute((P_new / np.amax(P_new)) * dc.P)
        P_old = np.copy(P_new[:,:])

        count += 1
    # if count > 1:
    #     print("pressure took ",count," to converge")

    return P_new

def calc_pressure(dc, oc, rho, P, u_ishift_star, v_jshift_star):

    # Tolerance for convergence
    min_loops=oc.min_Ploops
    max_loops=oc.max_Ploops
    tol = oc.Ptol

    # Setting a dummy matrix to be changed
    P_old = np.array(np.copy(P[:,:]))
    P_new = np.array(np.copy(P[:,:]))

    # Getting wall 0s to set the pressure values
    wall_0s = np.ones(P_old.shape)
    wall_index = str_index(dc.domain_map,"w")
    wall_0s[wall_index] = 0
    # plt.imshow(wall_0s)
    # plt.show()
    # exit()

    P_new = loop_Pressure(tol,max_loops, min_loops,wall_0s,P_old,P_new,
                dc.C, rho, dc.h, dc.dt, u_ishift_star, v_jshift_star)

    return np.copy(P_new)

def str_index(map,str):
    map = np.array(map,dtype="str")
    return np.core.defchararray.find(map,str)!=-1

def get_psi(X,Y,x_b,y_b,r_b):
    # Defining a distance field for one bubble
    psi = ( (X.T - x_b)**2 + (Y.T - y_b)**2)**0.5 - r_b
    return psi

def sgn_Mh(psi,M,h):
    val = np.zeros(psi.shape)
    val[psi>=(M*h)] = 1
    val[psi<=(-M*h)] = -1
    val[val==0] = psi[val==0]/(M*h) - (1/np.pi)*np.sin(np.pi*psi[val==0]/(M*h))
    return val

def sgn(psi):
    val = np.copy(psi[:,:])
    val[val!=0] = val[val!=0] / np.absolute(val[val!=0])
    return val

def update_properties(f,prop_l,prop_g):
    return f[:,:]*prop_l + (1-f[:,:])*prop_g
# Function for calculating the f(psi) using a variable M value
def get_f(M,h,psi,f):
    # Creating an array of logical statements used in the piecewise function
    cond1 = psi < -M*h
    cond2 = psi > M * h
    cond3 = ~np.logical_xor(cond1,cond2)

    # Using the conditionals to calulate an f(psi)
    f[cond1] = 0
    f[cond2] = 1
    f[cond3] = (0.5) * (1 + psi[cond3]/(M*h) + (1/np.pi)*np.sin(np.pi*(psi[cond3]/(M*h))))

    return f

class bubble_class:
    def __init__(self,u):
        self.psi_b = []
        self.psi_d = []
        self.x = []
        self.y = []
        self.r = []
        self.N_b = 0

        # Pre-allocation
        self.psi = np.zeros(u.shape)
        self.dpsi_dx = np.zeros(u.shape)
        self.dpsi_dy = np.zeros(u.shape)
        self.f = np.zeros(u.shape)
        self.f_ishift = np.zeros(u.shape)
        self.f_jshift = np.zeros(u.shape)
        self.df_dpsi_x = np.zeros(u.shape)
        self.df_dpsi_y = np.zeros(u.shape)
        self.kappa = np.zeros(u.shape)
        self.psi_x = np.zeros(u.shape)
        self.psi_y = np.zeros(u.shape)
        self.dpsi_dx_jshift = np.zeros(u.shape)
        self.dpsi_dy_ishift = np.zeros(u.shape)
        self.kappa_ishift = np.zeros(u.shape)
        self.kappa_jshift = np.zeros(u.shape)
        self.df_dpsi_ian = np.zeros(u.shape)
        self.df_dpsi_jan = np.zeros(u.shape)

    def add_bubble(self,dc,x_b,y_b,r_b):
        # Defining an initial distance field for one bubble
        self.psi_b.append(get_psi(dc.X,dc.Y,x_b,y_b,r_b))
        self.x.append(x_b)
        self.y.append(y_b)
        self.r.append(r_b)
        self.N_b += 1

    def add_droplet(self,dc,x_b,y_b,r_b):
        # Defining an initial distance field for one bubble
        self.psi_d.append(-1*get_psi(dc.X,dc.Y,x_b,y_b,r_b))
        self.x.append(x_b)
        self.y.append(y_b)
        self.r.append(r_b)
        self.N_b += 1

    def calc_all_psi_d(self):
        if self.psi_d == []:
            self.psi[:,:] = -1
        else:
            # This gets a continuous level set function for the entire domain
            self.psi = np.moveaxis(np.array(np.copy(self.psi_d[:])),0,-1)
            self.psi = np.amax(self.psi,axis=-1)

    def calc_all_psi_b(self):
        if self.psi_d == []:
            self.psi[:,:] = 1
        else:
            # This gets a continuous level set function for the entire domain
            self.psi = np.moveaxis(np.array(np.copy(self.psi_b[:])),0,-1)
            self.psi = np.amin(self.psi,axis=-1)

    def predict_psi(self,dc,fc):
        # This is the predictor step for the level set function
        self.Lpsi_fun(dc,fc)

        self.psi_star = self.psi + dc.dt * self.Lpsi

    def correct_psi(self,dc,fc):
        # This is the corrector step for the level set funcion
        self.Lpsi_star_fun(dc,fc)

        self.psi = self.psi + (dc.dt/2) * (self.Lpsi + self.Lpsi_star)

    def Lpsi_fun(self,dc,fc):
        # Calculating the neccessary values for Lpsi using the class funcitons
        self.D_psi_fun()
        self.M_switch()
        self.psi_shift_fun(fc)

        self.Lpsi = np.zeros(self.psi.shape)
        # u = np.zeros(u)
        # v = np.zeros(v)

        u = u_new_fun(fc.u_ishift)
        v = v_new_fun(fc.v_jshift)

        self.Lpsi[1:-1,1:-1] = ( -u * (1/dc.h) * (self.psi_ishift[1:-1,1:-1] - self.psi_ishift[:-2,1:-1]) -
                     v * (1/dc.h) * (self.psi_jshift[1:-1,1:-1] - self.psi_jshift[1:-1,:-2]) )

    def Lpsi_star_fun(self,dc,fc):
        # Calculating the neccessary values for Lpsi using the class funcitons
        self.D_psi_star_fun()
        self.M_switch_star()
        self.psi_star_shift_fun(fc)

        self.Lpsi_star = np.zeros(self.psi_star.shape)
        # u = np.zeros(u)
        # v = np.zeros(v)

        u = u_new_fun(fc.u_ishift)
        v = v_new_fun(fc.v_jshift)

        self.Lpsi_star[1:-1,1:-1] = ( -u * (1/dc.h) * (self.psi_star_ishift[1:-1,1:-1] - self.psi_star_ishift[:-2,1:-1]) -
                     v * (1/dc.h) * (self.psi_star_jshift[1:-1,1:-1] - self.psi_star_jshift[1:-1,:-2]) )

    def psi_shift_fun(self,fc):
        # --> Calculating psi shifted in the i direction (1+1/2,j)
        # Setting up an array to fill
        self.psi_ishift = np.zeros(self.psi.shape) + np.copy(self.psi[:,:])
        self.psi_jshift = np.zeros(self.psi.shape) + np.copy(self.psi[:,:])

        # Getting shifted psi values to be used for the extrapolation
        psi_ishift_temp = np.zeros(self.psi.shape)
        psi_jshift_temp = np.zeros(self.psi.shape)
        psi_ishift_temp[:-1,:] = np.copy(self.psi[1:,:])
        psi_jshift_temp[:,:-1] = np.copy(self.psi[:,1:])

        # Getting booleans for the velocity profiles
        bool_array_i = fc.u_ishift > 0
        bool_array_j = fc.v_jshift > 0

        # Filling the arrays for positive and negative velocities
        self.psi_ishift[bool_array_i] = self.psi[bool_array_i] + 0.5 * self.Mi[bool_array_i]
        self.psi_ishift[~bool_array_i] = psi_ishift_temp[~bool_array_i] - 0.5 * self.Mi_shift[~bool_array_i]

        self.psi_jshift[bool_array_j] = self.psi[bool_array_j] + 0.5 * self.Mj[bool_array_j]
        self.psi_jshift[~bool_array_j] = psi_jshift_temp[~bool_array_j] - 0.5 * self.Mj_shift[~bool_array_j]

    def psi_star_shift_fun(self,fc):
        # --> Calculating psi shifted in the i direction (1+1/2,j)
        # Setting up an array to fill
        self.psi_star_ishift = np.zeros(self.psi_star.shape) + self.psi_star[:,:]
        self.psi_star_jshift = np.zeros(self.psi_star.shape) + self.psi_star[:,:]

        # Getting shifted psi values to be used for the extrapolation
        psi_ishift_temp = np.zeros(self.psi_star.shape)
        psi_jshift_temp = np.zeros(self.psi_star.shape)
        psi_ishift_temp[:-1,:] = np.copy(self.psi_star[1:,:])
        psi_jshift_temp[:,:-1] = np.copy(self.psi_star[:,1:])

        # Getting booleans for the velocity profiles
        bool_array_i = fc.u_ishift > 0
        bool_array_j = fc.v_jshift > 0

        # Filling the arrays for positive and negative velocities
        self.psi_star_ishift[bool_array_i] = self.psi_star[bool_array_i] + 0.5 * self.Mi_star[bool_array_i]
        self.psi_star_ishift[~bool_array_i] = psi_ishift_temp[~bool_array_i] - 0.5 * self.Mi_shift_star[~bool_array_i]

        self.psi_star_jshift[bool_array_j] = self.psi_star[bool_array_j] + 0.5 * self.Mj_star[bool_array_j]
        self.psi_star_jshift[~bool_array_j] = psi_jshift_temp[~bool_array_j] - 0.5 * self.Mj_shift_star[~bool_array_j]

    def D_psi_fun(self):
        # Initializing the arrays to the correct dimentsions
        self.Dpi_psi       = np.zeros(self.psi.shape)# + np.nan
        self.Dni_psi       = np.zeros(self.psi.shape)# + np.nan
        self.Dpj_psi       = np.zeros(self.psi.shape)# + np.nan
        self.Dnj_psi       = np.zeros(self.psi.shape)# + np.nan
        self.Dpi_shift_psi = np.zeros(self.psi.shape)# + np.nan
        self.Dni_shift_psi = np.zeros(self.psi.shape)# + np.nan
        self.Dpj_shift_psi = np.zeros(self.psi.shape)# + np.nan
        self.Dnj_shift_psi = np.zeros(self.psi.shape)# + np.nan

        # Getting the D psi values centered around i,j
        self.Dpi_psi[1:-1,1:-1] = self.psi[2:,1:-1] - self.psi[1:-1,1:-1]
        self.Dni_psi[1:-1,1:-1] = self.psi[1:-1,1:-1] - self.psi[:-2,1:-1]
        self.Dpj_psi[1:-1,1:-1] = self.psi[1:-1,2:] - self.psi[1:-1,1:-1]
        self.Dnj_psi[1:-1,1:-1] = self.psi[1:-1,1:-1] - self.psi[1:-1,:-2]

        # Getting the D psi values centered around i+1,j
        self.Dpi_shift_psi[1:-2,1:-1] = self.psi[3:,1:-1] - self.psi[2:-1,1:-1]
        self.Dni_shift_psi[1:-1,1:-1] = self.psi[2:,1:-1] - self.psi[1:-1,1:-1]
        self.Dpj_shift_psi[1:-1,1:-2] = self.psi[1:-1,3:] - self.psi[1:-1,2:-1]
        self.Dnj_shift_psi[1:-1,1:-1] = self.psi[1:-1,2:] - self.psi[1:-1,1:-1]

        # Combining all of the positive and negative arrays
        self.Di_psi = np.array([self.Dni_psi,self.Dpi_psi])
        self.Dj_psi = np.array([self.Dnj_psi,self.Dpj_psi])
        self.Di_shift_psi = np.array([self.Dni_shift_psi,self.Dpi_shift_psi])
        self.Dj_shift_psi = np.array([self.Dnj_shift_psi,self.Dpj_shift_psi])

    def D_psi_star_fun(self):
        # Initializing the arrays to the correct dimentsions
        self.Dpi_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        self.Dni_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        self.Dpj_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        self.Dnj_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        self.Dpi_shift_psi_star = np.zeros(self.psi_star.shape)# + np.nan
        self.Dni_shift_psi_star = np.zeros(self.psi_star.shape)# + np.nan
        self.Dpj_shift_psi_star = np.zeros(self.psi_star.shape)# + np.nan
        self.Dnj_shift_psi_star = np.zeros(self.psi_star.shape)# + np.nan

        # Getting the D psi values centered around i,j
        self.Dpi_psi_star[1:-1,1:-1] = self.psi_star[2:,1:-1] - self.psi_star[1:-1,1:-1]
        self.Dni_psi_star[1:-1,1:-1] = self.psi_star[1:-1,1:-1] - self.psi_star[:-2,1:-1]
        self.Dpj_psi_star[1:-1,1:-1] = self.psi_star[1:-1,2:] - self.psi_star[1:-1,1:-1]
        self.Dnj_psi_star[1:-1,1:-1] = self.psi_star[1:-1,1:-1] - self.psi_star[1:-1,:-2]

        # Getting the D psi values centered around i+1,j
        self.Dpi_shift_psi_star[1:-2,1:-1] = self.psi_star[3:,1:-1] - self.psi_star[2:-1,1:-1]
        self.Dni_shift_psi_star[1:-1,1:-1] = self.psi_star[2:,1:-1] - self.psi_star[1:-1,1:-1]
        self.Dpj_shift_psi_star[1:-1,1:-2] = self.psi_star[1:-1,3:] - self.psi_star[1:-1,2:-1]
        self.Dnj_shift_psi_star[1:-1,1:-1] = self.psi_star[1:-1,2:] - self.psi_star[1:-1,1:-1]

        # Combining all of the positive and negative arrays
        self.Di_psi_star = np.array([self.Dni_psi_star,self.Dpi_psi_star])
        self.Dj_psi_star = np.array([self.Dnj_psi_star,self.Dpj_psi_star])
        self.Di_shift_psi_star = np.array([self.Dni_shift_psi_star,self.Dpi_shift_psi_star])
        self.Dj_shift_psi_star = np.array([self.Dnj_shift_psi_star,self.Dpj_shift_psi_star])

    def M_switch(self):
        # Establishing a bool array for indexing
        min_ind_i = np.array(np.zeros(self.Di_psi.shape),dtype=np.bool)
        min_ind_j = np.array(np.zeros(self.Dj_psi.shape),dtype=np.bool)
        min_ind_i_shift = np.array(np.zeros(self.Di_shift_psi.shape),dtype=np.bool)
        min_ind_j_shift = np.array(np.zeros(self.Dj_shift_psi.shape),dtype=np.bool)

        # Preallocating a switch function
        self.Mi = np.copy(self.Di_psi[0,:,:])
        self.Mj = np.copy(self.Dj_psi[0,:,:])
        self.Mi_shift = np.copy(self.Di_shift_psi[0,:,:])
        self.Mj_shift = np.copy(self.Dj_shift_psi[0,:,:])

        # Getting the absolute minimum values
        abs_min_i = np.nanmin(np.absolute(self.Di_psi[:,1:-1,1:-1]),axis=0)
        abs_min_j = np.nanmin(np.absolute(self.Dj_psi[:,1:-1,1:-1]),axis=0)
        abs_min_i_shift = np.nanmin(np.absolute(self.Di_shift_psi[:,1:-1,1:-1]),axis=0)
        abs_min_j_shift = np.nanmin(np.absolute(self.Dj_shift_psi[:,1:-1,1:-1]),axis=0)

        # Getting the indexes by using the absolute value of the minimums
        min_ind_i[:,1:-1,1:-1] = abs_min_i == np.absolute(self.Di_psi[:,1:-1,1:-1])
        min_ind_j[:,1:-1,1:-1] = abs_min_j == np.absolute(self.Dj_psi[:,1:-1,1:-1])
        min_ind_i_shift[:,1:-1,1:-1] = abs_min_i_shift == np.absolute(self.Di_shift_psi[:,1:-1,1:-1])
        min_ind_j_shift[:,1:-1,1:-1] = abs_min_j_shift == np.absolute(self.Dj_shift_psi[:,1:-1,1:-1])

        # Finally putting the final values in the switch function array
        self.Mi[min_ind_i[0,:,:]] = self.Di_psi[0,min_ind_i[0,:,:]]
        self.Mi[min_ind_i[1,:,:]] = self.Di_psi[1,min_ind_i[1,:,:]]

        self.Mj[min_ind_j[0,:,:]] = self.Dj_psi[0,min_ind_j[0,:,:]]
        self.Mj[min_ind_j[1,:,:]] = self.Dj_psi[1,min_ind_j[1,:,:]]

        self.Mi_shift[min_ind_i_shift[0,:,:]] = self.Di_shift_psi[0,min_ind_i_shift[0,:,:]]
        self.Mi_shift[min_ind_i_shift[1,:,:]] = self.Di_shift_psi[1,min_ind_i_shift[1,:,:]]

        self.Mj_shift[min_ind_j_shift[0,:,:]] = self.Dj_shift_psi[0,min_ind_j_shift[0,:,:]]
        self.Mj_shift[min_ind_j_shift[1,:,:]] = self.Dj_shift_psi[1,min_ind_j_shift[1,:,:]]

    def M_switch_star(self):
        # Establishing a bool array for indexing
        min_ind_i = np.array(np.zeros(self.Di_psi_star.shape),dtype=np.bool)
        min_ind_j = np.array(np.zeros(self.Dj_psi_star.shape),dtype=np.bool)
        min_ind_i_shift = np.array(np.zeros(self.Di_shift_psi_star.shape),dtype=np.bool)
        min_ind_j_shift = np.array(np.zeros(self.Dj_shift_psi_star.shape),dtype=np.bool)

        # Preallocating a switch function
        self.Mi_star = np.copy(self.Di_psi_star[0,:,:])
        self.Mj_star = np.copy(self.Dj_psi_star[0,:,:])
        self.Mi_shift_star = np.copy(self.Di_shift_psi_star[0,:,:])
        self.Mj_shift_star = np.copy(self.Dj_shift_psi_star[0,:,:])

        # Getting the absolute minimum values
        abs_min_i = np.nanmin(np.absolute(self.Di_psi_star[:,1:-1,1:-1]),axis=0)
        abs_min_j = np.nanmin(np.absolute(self.Dj_psi_star[:,1:-1,1:-1]),axis=0)
        abs_min_i_shift = np.nanmin(np.absolute(self.Di_shift_psi_star[:,1:-1,1:-1]),axis=0)
        abs_min_j_shift = np.nanmin(np.absolute(self.Dj_shift_psi_star[:,1:-1,1:-1]),axis=0)

        # Getting the indexes by using the absolute value of the minimums
        min_ind_i[:,1:-1,1:-1] = abs_min_i == np.absolute(self.Di_psi_star[:,1:-1,1:-1])
        min_ind_j[:,1:-1,1:-1] = abs_min_j == np.absolute(self.Dj_psi_star[:,1:-1,1:-1])
        min_ind_i_shift[:,1:-1,1:-1] = abs_min_i_shift == np.absolute(self.Di_shift_psi_star[:,1:-1,1:-1])
        min_ind_j_shift[:,1:-1,1:-1] = abs_min_j_shift == np.absolute(self.Dj_shift_psi_star[:,1:-1,1:-1])

        # Finally putting the final values in the switch function array
        self.Mi_star[min_ind_i[0,:,:]] = self.Di_psi_star[0,min_ind_i[0,:,:]]
        self.Mi_star[min_ind_i[1,:,:]] = self.Di_psi_star[1,min_ind_i[1,:,:]]

        self.Mj_star[min_ind_j[0,:,:]] = self.Dj_psi_star[0,min_ind_j[0,:,:]]
        self.Mj_star[min_ind_j[1,:,:]] = self.Dj_psi_star[1,min_ind_j[1,:,:]]

        self.Mi_shift_star[min_ind_i_shift[0,:,:]] = self.Di_shift_psi_star[0,min_ind_i_shift[0,:,:]]
        self.Mi_shift_star[min_ind_i_shift[1,:,:]] = self.Di_shift_psi_star[1,min_ind_i_shift[1,:,:]]

        self.Mj_shift_star[min_ind_j_shift[0,:,:]] = self.Dj_shift_psi_star[0,min_ind_j_shift[0,:,:]]
        self.Mj_shift_star[min_ind_j_shift[1,:,:]] = self.Dj_shift_psi_star[1,min_ind_j_shift[1,:,:]]

    def calc_f_arr(self,M,h):
        self.f[1:-1,1:-1] = get_f(M,h,self.psi[1:-1,1:-1],self.f[1:-1,1:-1])
        self.f_ishift[1:-1,1:-1] = get_f(M,h,self.psi_ishift[1:-1,1:-1],self.f_ishift[1:-1,1:-1])
        self.f_jshift[1:-1,1:-1] = get_f(M,h,self.psi_jshift[1:-1,1:-1],self.f_jshift[1:-1,1:-1])

    def calc_f_der_an(self,M,h):
        # Calculating the analytic derivative of df(psi)/dpsi
        cond1_i = np.logical_xor(self.f_ishift == 0, self.f_ishift ==1)
        self.df_dpsi_ian[cond1_i] = 0
        self.df_dpsi_ian[~cond1_i] = 0.5 * (1/(M * h) + (1/(M*h))*np.cos(np.pi*self.psi_ishift[~cond1_i]/(M*h)))

        cond1_j = np.logical_xor(self.f_jshift == 0, self.f_jshift ==1)
        self.df_dpsi_jan[cond1_j] = 0
        self.df_dpsi_jan[~cond1_j] = 0.5 * (1/(M * h) + (1/(M*h))*np.cos(np.pi*self.psi_jshift[~cond1_j]/(M*h)))

    def calc_f_der(self):
        # This gets the derivative of df/dpsi in the x and y directions centered
        #   about (i+1/2,j) and (i,j+1/2)
        self.df_dpsi_x[1:-1,:] = (self.f[2:,:] - self.f[1:-1,:])/(self.psi[2:,:] - self.psi[1:-1,:])
        self.df_dpsi_y[:,1:-1] = (self.f[:,2:] - self.f[:,1:-1])/(self.psi[:,2:] - self.psi[:,1:-1])

    def get_psi_grad(self,h):
        # This calculates the gradients of psi in the x and y direction
        #   NOTE: They are centered about (i+1/2,j) and (i,j+1/2)
        self.dpsi_dx[1:-1,:] = (self.psi[2:,:] - self.psi[1:-1,:])/h
        self.dpsi_dy[:,1:-1] = (self.psi[:,2:] - self.psi[:,1:-1])/h

    def calc_kappa(self,h):
        # Calculating the gradients centered about (i+1/2,j+1/2)
        self.dpsi_dx_jshift[:,1:-1] = 0.5*(self.dpsi_dx[:,2:] + self.dpsi_dx[:,1:-1])
        self.dpsi_dy_ishift[1:-1,:] = 0.5*(self.dpsi_dy[2:,:] + self.dpsi_dy[1:-1,:])

        # Calculating the magnitude of the shifted gradients centered at (i+1/2,j+1/2)
        mag_psi_grad = ( (self.dpsi_dx_jshift[:,:])**2 + (self.dpsi_dy_ishift[:,:])**2 )**0.5
        mag_psi_grad[mag_psi_grad==0] = np.nan

        # Calculating the psi_x and psi_y terms centered at (i+1/2,j+1/2)
        self.psi_x[1:-1,1:-1] = (self.psi[2:,2:] + self.psi[2:,1:-1] - self.psi[1:-1,2:] - self.psi[1:-1,1:-1])*(1/(2*h))
        self.psi_y[1:-1,1:-1] = (self.psi[2:,2:] - self.psi[2:,1:-1] + self.psi[1:-1,2:] - self.psi[1:-1,1:-1])*(1/(2*h))

        # Calculating Kappa
        self.kappa[1:-1,1:-1] = ( (self.psi_x[1:-1,1:-1]/mag_psi_grad[1:-1,1:-1])
                                + (self.psi_x[1:-1, :-2]/mag_psi_grad[1:-1, :-2])
                                - (self.psi_x[ :-2,1:-1]/mag_psi_grad[ :-2,1:-1])
                                - (self.psi_x[ :-2, :-2]/mag_psi_grad[ :-2, :-2])
                                + (self.psi_y[1:-1,1:-1]/mag_psi_grad[1:-1,1:-1])
                                - (self.psi_y[1:-1, :-2]/mag_psi_grad[1:-1, :-2])
                                + (self.psi_y[ :-2,1:-1]/mag_psi_grad[ :-2,1:-1])
                                - (self.psi_y[ :-2, :-2]/mag_psi_grad[ :-2, :-2]) ) * (1/(2*h))

        # Calculating the shifted kappa values
        self.kappa_ishift[1:-1,:] = 0.5*(self.kappa[2:,:] + self.kappa[1:-1,:])
        self.kappa_jshift[:,1:-1] = 0.5*(self.kappa[:,2:] + self.kappa[:,1:-1])

    def calc_F_gamma(self,gamma,M,h):
        # Calculating a f array for determining fluid properties
        # self.calc_f_arr(M,h)

        # Calculating a gradient in both directions for psi
        self.get_psi_grad(h)

        # Calculating the derivative of f
        # self.calc_f_der()
        self.calc_f_der_an(M,h)

        # Calculating Kappa
        self.calc_kappa(h)

        # Getting the final Force values
        self.F_gamma_x = -gamma*self.kappa_ishift*self.df_dpsi_ian[:,:]*self.dpsi_dx[:,:]
        self.F_gamma_x[np.isnan(self.F_gamma_x)] = 0
        self.F_gamma_x[np.logical_xor(self.f == 0,self.f==1)] = 0
        self.F_gamma_y = -gamma*self.kappa_jshift*self.df_dpsi_jan[:,:]*self.dpsi_dy[:,:]
        self.F_gamma_y[np.isnan(self.F_gamma_y)] = 0
        self.F_gamma_y[np.logical_xor(self.f == 0,self.f==1)] = 0

class distance_field:
    def __init__(self,M,bc,dc):
        self.M = M
        self.psi = np.copy(bc.psi[:,:])
        self.tou = 0
        self.dtou = dc.h/2
        self.num_loops = 2*self.M

        # Initializing the sgn and sgn_Mh arrays
        self.sgn_psi = sgn(self.psi[:,:])
        self.sgn_Mh_psi = sgn_Mh(self.psi[:,:],M,dc.h)

        # Initializing the arrays to the correct dimentsions
        self.DDi_psi       = np.zeros(self.psi.shape)# + np.nan
        self.DDj_psi       = np.zeros(self.psi.shape)# + np.nan
        self.DDi_pshift_psi = np.zeros(self.psi.shape)# + np.nan
        self.DDi_nshift_psi = np.zeros(self.psi.shape)# + np.nan
        self.DDj_pshift_psi = np.zeros(self.psi.shape)# + np.nan
        self.DDj_nshift_psi = np.zeros(self.psi.shape)# + np.nan

        # Preallocating a switch function
        self.Mi_p = np.zeros(self.psi.shape)
        self.Mj_p = np.zeros(self.psi.shape)
        self.Mi_n = np.zeros(self.psi.shape)
        self.Mj_n = np.zeros(self.psi.shape)

        # Preallocating the D tilda array
        self.Di_ptilda = np.zeros(self.psi.shape)
        self.Dj_ptilda = np.zeros(self.psi.shape)
        self.Di_ntilda = np.zeros(self.psi.shape)
        self.Dj_ntilda = np.zeros(self.psi.shape)

        self.Di_tilda = np.zeros(self.psi.shape)
        self.Dj_tilda = np.zeros(self.psi.shape)

        self.Lpsi = np.zeros(self.psi.shape)

        #-->  Preallocation of Star arrays
        self.psi_star = np.zeros(self.psi.shape)

        # Initializing the sgn and sgn_Mh arrays
        self.sgn_psi_star = sgn(self.psi[:,:])
        self.sgn_Mh_psi_star = sgn_Mh(self.psi[:,:],M,dc.h)

        self.DDi_psi_star       = np.zeros(self.psi.shape)# + np.nan
        self.DDj_psi_star       = np.zeros(self.psi.shape)# + np.nan
        self.DDi_pshift_psi_star = np.zeros(self.psi.shape)# + np.nan
        self.DDi_nshift_psi_star = np.zeros(self.psi.shape)# + np.nan
        self.DDj_pshift_psi_star = np.zeros(self.psi.shape)# + np.nan
        self.DDj_nshift_psi_star = np.zeros(self.psi.shape)# + np.nan

        # Preallocating a switch function
        self.Mi_p_star = np.zeros(self.psi.shape)
        self.Mj_p_star = np.zeros(self.psi.shape)
        self.Mi_n_star = np.zeros(self.psi.shape)
        self.Mj_n_star = np.zeros(self.psi.shape)

        # Preallocating the D tilda array
        self.Di_ptilda_star = np.zeros(self.psi.shape)
        self.Dj_ptilda_star = np.zeros(self.psi.shape)
        self.Di_ntilda_star = np.zeros(self.psi.shape)
        self.Dj_ntilda_star = np.zeros(self.psi.shape)

        self.Di_tilda_star = np.zeros(self.psi.shape)
        self.Dj_tilda_star = np.zeros(self.psi.shape)

        self.Lpsi_star = np.zeros(self.psi.shape)

        # Pre-allocation
        self.dpsi_dx = np.zeros(self.psi.shape)
        self.dpsi_dy = np.zeros(self.psi.shape)
        self.f = np.zeros(self.psi.shape)
        self.f_ishift = np.zeros(self.psi.shape)
        self.f_jshift = np.zeros(self.psi.shape)
        self.df_dpsi_x = np.zeros(self.psi.shape)
        self.df_dpsi_y = np.zeros(self.psi.shape)
        self.df_dpsi_ian = np.zeros(self.psi.shape)
        self.df_dpsi_jan = np.zeros(self.psi.shape)
        self.kappa = np.zeros(self.psi.shape)
        self.psi_x = np.zeros(self.psi.shape)
        self.psi_y = np.zeros(self.psi.shape)
        self.dpsi_dx_jshift = np.zeros(self.psi.shape)
        self.dpsi_dy_ishift = np.zeros(self.psi.shape)
        self.kappa_ishift = np.zeros(self.psi.shape)
        self.kappa_jshift = np.zeros(self.psi.shape)

    def predict_psi(self,dc,fc):
        # This is the predictor step for the level set function
        self.Lpsi_fun(dc,fc)

        # Predicting the level set distance function
        self.psi_star[:,:] = self.psi[:,:] + self.dtou * self.Lpsi[:,:]

    def correct_psi(self,dc,fc):
        # Calculating the lpsi from the predictor level set function
        self.Lpsi_star_fun(dc,fc)

        # Correcting the level set distance function
        self.psi[:,:] = self.psi[:,:] + (self.dtou/2) * (self.Lpsi[:,:] + self.Lpsi_star[:,:])

        # Calculating the shifted psi values
        self.psi_ishift = np.copy(self.psi[:,:])
        self.psi_jshift = np.copy(self.psi[:,:])
        self.psi_ishift[1:-1,1:-1] = 0.5 * (self.psi_ishift[2:,1:-1] + self.psi_ishift[1:-1,1:-1])
        self.psi_jshift[1:-1,1:-1] = 0.5 * (self.psi_jshift[1:-1,2:] + self.psi_jshift[1:-1,1:-1])

    def Lpsi_fun(self,dc,fc):
        # Calculating the neccessary values for Lpsi using the class funcitons
        self.sgn_psi[:,:] = sgn(self.psi[:,:])
        self.sgn_Mh_psi[:,:] = sgn_Mh(self.psi[:,:],self.M,dc.h)
        self.D_psi_fun()
        self.DD_psi_fun()
        self.M_switch()
        self.D_stilda_fun()
        self.D_tilda_fun()

        # Calculating the L_psi value
        self.Lpsi[:,:] = self.sgn_Mh_psi[:,:] * (1 - ( (self.Di_tilda[:,:]/dc.h)**2 + (self.Dj_tilda[:,:]/dc.h)**2 )**0.5)

    def Lpsi_star_fun(self,dc,fc):
        # Calculating the neccessary values for Lpsi using the class funcitons
        self.sgn_psi_star[:,:] = sgn(self.psi_star[:,:])
        self.sgn_Mh_psi_star[:,:] = sgn_Mh(self.psi_star[:,:],self.M,dc.h)
        self.D_psi_star_fun()
        self.DD_psi_star_fun()
        self.M_switch_star()
        self.D_stilda_star_fun()
        self.D_tilda_star_fun()

        # Calculating the L_psi value
        self.Lpsi_star[:,:] = self.sgn_Mh_psi_star[:,:] * (1 - ( (self.Di_tilda_star[:,:]/dc.h)**2 + (self.Dj_tilda_star[:,:]/dc.h)**2 )**0.5)

    def D_psi_fun(self):
        # Initializing the arrays to the correct dimentsions
        Dpi_psi       = np.zeros(self.psi.shape)# + np.nan
        Dni_psi       = np.zeros(self.psi.shape)# + np.nan
        Dpj_psi       = np.zeros(self.psi.shape)# + np.nan
        Dnj_psi       = np.zeros(self.psi.shape)# + np.nan
        # Dpi_shift_psi = np.zeros(self.psi.shape)# + np.nan
        # Dni_shift_psi = np.zeros(self.psi.shape)# + np.nan
        # Dpj_shift_psi = np.zeros(self.psi.shape)# + np.nan
        # Dnj_shift_psi = np.zeros(self.psi.shape)# + np.nan

        # Getting the D psi values centered around i,j
        Dpi_psi[1:-1,1:-1] = self.psi[2:,1:-1] - self.psi[1:-1,1:-1]
        Dni_psi[1:-1,1:-1] = self.psi[1:-1,1:-1] - self.psi[:-2,1:-1]
        Dpj_psi[1:-1,1:-1] = self.psi[1:-1,2:] - self.psi[1:-1,1:-1]
        Dnj_psi[1:-1,1:-1] = self.psi[1:-1,1:-1] - self.psi[1:-1,:-2]

        # Getting the D psi values centered around i+1,j
        # Dpi_shift_psi[1:-2,1:-1] = self.psi[3:,1:-1] - self.psi[2:-1,1:-1]
        # Dni_shift_psi[1:-1,1:-1] = self.psi[2:,1:-1] - self.psi[1:-1,1:-1]
        # Dpj_shift_psi[1:-1,1:-2] = self.psi[1:-1,3:] - self.psi[1:-1,2:-1]
        # Dnj_shift_psi[1:-1,1:-1] = self.psi[1:-1,2:] - self.psi[1:-1,1:-1]

        # Combining all of the positive and negative arrays
        #   Note: The output has negative in the [0,:,:], positive in the [1,:,:]
        # self.Di_shift_psi = np.array([Dni_shift_psi,Dpi_shift_psi])
        # self.Dj_shift_psi = np.array([Dnj_shift_psi,Dpj_shift_psi])
        self.Di_psi = np.array([Dni_psi,Dpi_psi])
        self.Dj_psi = np.array([Dnj_psi,Dpj_psi])

    def D_psi_star_fun(self):
        # Initializing the arrays to the correct dimentsions
        Dpi_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        Dni_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        Dpj_psi_star       = np.zeros(self.psi_star.shape)# + np.nan
        Dnj_psi_star       = np.zeros(self.psi_star.shape)# + np.nan

        # Getting the D psi values centered around i,j
        Dpi_psi_star[1:-1,1:-1] = self.psi_star[2:,1:-1] - self.psi_star[1:-1,1:-1]
        Dni_psi_star[1:-1,1:-1] = self.psi_star[1:-1,1:-1] - self.psi_star[:-2,1:-1]
        Dpj_psi_star[1:-1,1:-1] = self.psi_star[1:-1,2:] - self.psi_star[1:-1,1:-1]
        Dnj_psi_star[1:-1,1:-1] = self.psi_star[1:-1,1:-1] - self.psi_star[1:-1,:-2]


        # Combining all of the positive and negative arrays
        #   Note: The output has negative in the [0,:,:], positive in the [1,:,:]
        self.Di_psi_star = np.array([Dni_psi_star,Dpi_psi_star])
        self.Dj_psi_star = np.array([Dnj_psi_star,Dpj_psi_star])

    def DD_psi_fun(self):

        # Getting the D psi values centered around i,j
        self.DDi_psi[1:-1,1:-1] = self.psi[2:,1:-1] + self.psi[:-2,1:-1] - 2*self.psi[1:-1,1:-1]
        self.DDj_psi[1:-1,1:-1] = self.psi[1:-1,2:] + self.psi[1:-1,:-2] - 2*self.psi[1:-1,1:-1]

        # Getting the D psi values centered around i+1,j
        self.DDi_pshift_psi[1:-2,1:-1] = self.psi[3:,1:-1] + self.psi[1:-2,1:-1] - 2*self.psi[2:-1,1:-1]
        self.DDi_nshift_psi[1:-2,1:-1] = self.psi[2:-1,1:-1] + self.psi[:-3,1:-1] - 2*self.psi[1:-2,1:-1]
        self.DDj_pshift_psi[1:-1,1:-2] = self.psi[1:-1,3:] + self.psi[1:-1,1:-2] - 2*self.psi[1:-1,2:-1]
        self.DDj_nshift_psi[1:-1,1:-2] = self.psi[1:-1,2:-1] + self.psi[1:-1,:-3] - 2*self.psi[1:-1,1:-2]

        # Applying Boundaries
        def set_bnd_temp(arr):
            arr[:,1] = arr[:,2]
            arr[:,0] = arr[:,2]
            arr[:,-1] = arr[:,-3]
            arr[:,-2] = arr[:,-3]
            return arr

        self.DDi_psi = set_bnd_temp(self.DDi_psi)
        self.DDj_psi = set_bnd_temp(self.DDj_psi)
        self.DDi_pshift_psi = set_bnd_temp(self.DDi_pshift_psi)
        self.DDi_nshift_psi = set_bnd_temp(self.DDi_nshift_psi)
        self.DDj_pshift_psi = set_bnd_temp(self.DDj_pshift_psi)
        self.DDj_nshift_psi = set_bnd_temp(self.DDj_nshift_psi)

    def DD_psi_star_fun(self):

        # Getting the D psi values centered around i,j
        self.DDi_psi_star[1:-1,1:-1] = self.psi_star[2:,1:-1] + self.psi_star[:-2,1:-1] - 2*self.psi_star[1:-1,1:-1]
        self.DDj_psi_star[1:-1,1:-1] = self.psi_star[1:-1,2:] + self.psi_star[1:-1,:-2] - 2*self.psi_star[1:-1,1:-1]

        # Getting the D psi values centered around i+1,j
        self.DDi_pshift_psi_star[1:-2,1:-1] = self.psi_star[3:,1:-1] + self.psi_star[1:-2,1:-1] - 2*self.psi_star[2:-1,1:-1]
        self.DDi_nshift_psi_star[1:-2,1:-1] = self.psi_star[2:-1,1:-1] + self.psi_star[:-3,1:-1] - 2*self.psi_star[1:-2,1:-1]
        self.DDj_pshift_psi_star[1:-1,1:-2] = self.psi_star[1:-1,3:] + self.psi_star[1:-1,1:-2] - 2*self.psi_star[1:-1,2:-1]
        self.DDj_nshift_psi_star[1:-1,1:-2] = self.psi_star[1:-1,2:-1] + self.psi_star[1:-1,:-3] - 2*self.psi_star[1:-1,1:-2]

        # Applying Boundaries
        def set_bnd_temp(arr):
            arr[:,1] = arr[:,2]
            arr[:,0] = arr[:,2]
            arr[:,-1] = arr[:,-3]
            arr[:,-2] = arr[:,-3]
            return arr

        self.DDi_psi_star = set_bnd_temp(self.DDi_psi_star)
        self.DDj_psi_star = set_bnd_temp(self.DDj_psi_star)
        self.DDi_pshift_psi_star = set_bnd_temp(self.DDi_pshift_psi_star)
        self.DDi_nshift_psi_star = set_bnd_temp(self.DDi_nshift_psi_star)
        self.DDj_pshift_psi_star = set_bnd_temp(self.DDj_pshift_psi_star)
        self.DDj_nshift_psi_star = set_bnd_temp(self.DDj_nshift_psi_star)

    def M_switch(self):
        # Combining the DD functions for positive and negative
        DDi_p = np.array([self.DDi_psi,self.DDi_pshift_psi])
        DDj_p = np.array([self.DDj_psi,self.DDj_pshift_psi])
        DDi_n = np.array([self.DDi_psi,self.DDi_nshift_psi])
        DDj_n = np.array([self.DDj_psi,self.DDj_nshift_psi])

        # Establishing a bool array for indexing
        min_ind_pi = np.array(np.zeros(DDi_p.shape),dtype=np.bool)
        min_ind_pj = np.array(np.zeros(DDj_p.shape),dtype=np.bool)
        min_ind_ni = np.array(np.zeros(DDi_n.shape),dtype=np.bool)
        min_ind_nj = np.array(np.zeros(DDj_n.shape),dtype=np.bool)

        # Getting the absolute minimum values
        abs_min_pi = np.nanmin(np.absolute(DDi_p[:,1:-1,1:-1]),axis=0)
        abs_min_pj = np.nanmin(np.absolute(DDj_p[:,1:-1,1:-1]),axis=0)
        abs_min_ni = np.nanmin(np.absolute(DDi_n[:,1:-1,1:-1]),axis=0)
        abs_min_nj = np.nanmin(np.absolute(DDj_n[:,1:-1,1:-1]),axis=0)

        # Getting the indexes by using the absolute value of the minimums
        min_ind_pi[:,1:-1,1:-1] = abs_min_pi == np.absolute(DDi_p[:,1:-1,1:-1])
        min_ind_pj[:,1:-1,1:-1] = abs_min_pj == np.absolute(DDj_p[:,1:-1,1:-1])
        min_ind_ni[:,1:-1,1:-1] = abs_min_ni == np.absolute(DDi_n[:,1:-1,1:-1])
        min_ind_nj[:,1:-1,1:-1] = abs_min_nj == np.absolute(DDj_n[:,1:-1,1:-1])

        # Finally putting the final values in the switch function array
        self.Mi_p[min_ind_pi[0,:,:]] = DDi_p[0,min_ind_pi[0,:,:]]
        self.Mi_p[min_ind_pi[1,:,:]] = DDi_p[1,min_ind_pi[1,:,:]]

        self.Mj_p[min_ind_pj[0,:,:]] = DDj_p[0,min_ind_pj[0,:,:]]
        self.Mj_p[min_ind_pj[1,:,:]] = DDj_p[1,min_ind_pj[1,:,:]]

        self.Mi_n[min_ind_ni[0,:,:]] = DDi_n[0,min_ind_ni[0,:,:]]
        self.Mi_n[min_ind_ni[1,:,:]] = DDi_n[1,min_ind_ni[1,:,:]]

        self.Mj_n[min_ind_nj[0,:,:]] = DDj_n[0,min_ind_nj[0,:,:]]
        self.Mj_n[min_ind_nj[1,:,:]] = DDj_n[1,min_ind_nj[1,:,:]]

    def M_switch_star(self):
        # Combining the DD functions for positive and negative
        DDi_p = np.array([self.DDi_psi_star,self.DDi_pshift_psi_star])
        DDj_p = np.array([self.DDj_psi_star,self.DDj_pshift_psi_star])
        DDi_n = np.array([self.DDi_psi_star,self.DDi_nshift_psi_star])
        DDj_n = np.array([self.DDj_psi_star,self.DDj_nshift_psi_star])

        # Establishing a bool array for indexing
        min_ind_pi = np.array(np.zeros(DDi_p.shape),dtype=np.bool)
        min_ind_pj = np.array(np.zeros(DDj_p.shape),dtype=np.bool)
        min_ind_ni = np.array(np.zeros(DDi_n.shape),dtype=np.bool)
        min_ind_nj = np.array(np.zeros(DDj_n.shape),dtype=np.bool)

        # Getting the absolute minimum values
        abs_min_pi = np.nanmin(np.absolute(DDi_p[:,1:-1,1:-1]),axis=0)
        abs_min_pj = np.nanmin(np.absolute(DDj_p[:,1:-1,1:-1]),axis=0)
        abs_min_ni = np.nanmin(np.absolute(DDi_n[:,1:-1,1:-1]),axis=0)
        abs_min_nj = np.nanmin(np.absolute(DDj_n[:,1:-1,1:-1]),axis=0)

        # Getting the indexes by using the absolute value of the minimums
        min_ind_pi[:,1:-1,1:-1] = abs_min_pi == np.absolute(DDi_p[:,1:-1,1:-1])
        min_ind_pj[:,1:-1,1:-1] = abs_min_pj == np.absolute(DDj_p[:,1:-1,1:-1])
        min_ind_ni[:,1:-1,1:-1] = abs_min_ni == np.absolute(DDi_n[:,1:-1,1:-1])
        min_ind_nj[:,1:-1,1:-1] = abs_min_nj == np.absolute(DDj_n[:,1:-1,1:-1])

        # Finally putting the final values in the switch function array
        self.Mi_p_star[min_ind_pi[0,:,:]] = DDi_p[0,min_ind_pi[0,:,:]]
        self.Mi_p_star[min_ind_pi[1,:,:]] = DDi_p[1,min_ind_pi[1,:,:]]

        self.Mj_p_star[min_ind_pj[0,:,:]] = DDj_p[0,min_ind_pj[0,:,:]]
        self.Mj_p_star[min_ind_pj[1,:,:]] = DDj_p[1,min_ind_pj[1,:,:]]

        self.Mi_n_star[min_ind_ni[0,:,:]] = DDi_n[0,min_ind_ni[0,:,:]]
        self.Mi_n_star[min_ind_ni[1,:,:]] = DDi_n[1,min_ind_ni[1,:,:]]

        self.Mj_n_star[min_ind_nj[0,:,:]] = DDj_n[0,min_ind_nj[0,:,:]]
        self.Mj_n_star[min_ind_nj[1,:,:]] = DDj_n[1,min_ind_nj[1,:,:]]

    def D_stilda_fun(self):
        self.Di_ptilda = self.Di_psi[1,:,:] - (1/2) * self.Mi_p
        self.Di_ntilda = self.Di_psi[0,:,:] - (1/2) * self.Mi_n
        self.Dj_ptilda = self.Dj_psi[1,:,:] - (1/2) * self.Mj_p
        self.Dj_ntilda = self.Dj_psi[0,:,:] - (1/2) * self.Mj_n

    def D_stilda_star_fun(self):
        self.Di_ptilda_star = self.Di_psi_star[1,:,:] - (1/2) * self.Mi_p_star
        self.Di_ntilda_star = self.Di_psi_star[0,:,:] - (1/2) * self.Mi_n_star
        self.Dj_ptilda_star = self.Dj_psi_star[1,:,:] - (1/2) * self.Mj_p_star
        self.Dj_ntilda_star = self.Dj_psi_star[0,:,:] - (1/2) * self.Mj_n_star

    def D_tilda_fun(self):
        # Handling the otherwise condition
        self.Di_tilda[:,:] = 0.5*(self.Di_ptilda + self.Di_ntilda)
        self.Dj_tilda[:,:] = 0.5*(self.Dj_ptilda + self.Dj_ntilda)

        # Condition 1
        cond1i = np.logical_and(self.sgn_psi*self.Di_psi[1,:,:] < 0,
                               self.sgn_psi*self.Di_psi[0,:,:] < -self.sgn_psi*self.Di_psi[1,:,:])
        cond2i = np.logical_and(self.sgn_psi*self.Di_psi[0,:,:] > 0,
                               self.sgn_psi*self.Di_psi[1,:,:] > -self.sgn_psi*self.Di_psi[0,:,:])

        cond1j = np.logical_and(self.sgn_psi*self.Dj_psi[1,:,:] < 0,
                               self.sgn_psi*self.Dj_psi[0,:,:] < -self.sgn_psi*self.Dj_psi[1,:,:])
        cond2j = np.logical_and(self.sgn_psi*self.Dj_psi[0,:,:] > 0,
                               self.sgn_psi*self.Dj_psi[1,:,:] > -self.sgn_psi*self.Dj_psi[0,:,:])

        # Using the defined conditions to set the values of the array
        self.Di_tilda[cond1i] = self.Di_ptilda[cond1i]
        self.Di_tilda[cond2i] = self.Di_ntilda[cond2i]

        self.Dj_tilda[cond1j] = self.Dj_ptilda[cond1j]
        self.Dj_tilda[cond2j] = self.Dj_ntilda[cond2j]

    def D_tilda_star_fun(self):
        # Handling the otherwise condition
        self.Di_tilda_star[:,:] = 0.5*(self.Di_ptilda_star + self.Di_ntilda_star)
        self.Dj_tilda_star[:,:] = 0.5*(self.Dj_ptilda_star + self.Dj_ntilda_star)

        # Condition 1
        cond1i = np.logical_and(self.sgn_psi_star*self.Di_psi_star[1,:,:] < 0,
                                self.sgn_psi_star*self.Di_psi_star[0,:,:] < -self.sgn_psi_star*self.Di_psi_star[1,:,:])
        cond2i = np.logical_and(self.sgn_psi_star*self.Di_psi_star[0,:,:] > 0,
                                self.sgn_psi_star*self.Di_psi_star[1,:,:] > -self.sgn_psi_star*self.Di_psi_star[0,:,:])

        cond1j = np.logical_and(self.sgn_psi_star*self.Dj_psi_star[1,:,:] < 0,
                                self.sgn_psi_star*self.Dj_psi_star[0,:,:] < -self.sgn_psi_star*self.Dj_psi_star[1,:,:])
        cond2j = np.logical_and(self.sgn_psi_star*self.Dj_psi_star[0,:,:] > 0,
                                self.sgn_psi_star*self.Dj_psi_star[1,:,:] > -self.sgn_psi_star*self.Dj_psi_star[0,:,:])

        # Using the defined conditions to set the values of the array
        self.Di_tilda_star[cond1i] = self.Di_ptilda_star[cond1i]
        self.Di_tilda_star[cond2i] = self.Di_ntilda_star[cond2i]

        self.Dj_tilda_star[cond1j] = self.Dj_ptilda_star[cond1j]
        self.Dj_tilda_star[cond2j] = self.Dj_ntilda_star[cond2j]

    def calc_f_arr(self,M,h):
        self.f[1:-1,1:-1] = get_f(M,h,self.psi[1:-1,1:-1],self.f[1:-1,1:-1])
        self.f_ishift[1:-1,1:-1] = get_f(M,h,self.psi_ishift[1:-1,1:-1],self.f_ishift[1:-1,1:-1])
        self.f_jshift[1:-1,1:-1] = get_f(M,h,self.psi_jshift[1:-1,1:-1],self.f_jshift[1:-1,1:-1])

    # def calc_f_der(self):
    #     # This gets the derivative of df/dpsi in the x and y directions centered
    #     #   about (i+1/2,j) and (i,j+1/2)
    #     self.df_dpsi_x[1:-1,:] = (self.f[2:,:] - self.f[1:-1,:])/(self.psi[2:,:] - self.psi[1:-1,:])
    #     self.df_dpsi_y[:,1:-1] = (self.f[:,2:] - self.f[:,1:-1])/(self.psi[:,2:] - self.psi[:,1:-1])

    def calc_f_der_an(self,h):
        # Calculating the analytic derivative of df(psi)/dpsi
        cond1_i = np.logical_xor(self.f_ishift == 0, self.f_ishift ==1)
        self.df_dpsi_ian[cond1_i] = 0
        self.df_dpsi_ian[~cond1_i] = 0.5 * (1/(self.M * h) + (1/(self.M*h))*np.cos(np.pi*self.psi_ishift[~cond1_i]/(self.M*h)))

        cond1_j = np.logical_xor(self.f_jshift == 0, self.f_jshift ==1)
        self.df_dpsi_jan[cond1_j] = 0
        self.df_dpsi_jan[~cond1_j] = 0.5 * (1/(self.M * h) + (1/(self.M*h))*np.cos(np.pi*self.psi_jshift[~cond1_j]/(self.M*h)))

    def get_psi_grad(self,h):
        # This calculates the gradients of psi in the x and y direction
        #   NOTE: They are centered about (i+1/2,j) and (i,j+1/2)
        self.dpsi_dx[1:-1,1:-1] = (self.psi[2:,1:-1] - self.psi[1:-1,1:-1])/h
        self.dpsi_dy[1:-1,1:-1] = (self.psi[1:-1,2:] - self.psi[1:-1,1:-1])/h

    def calc_kappa(self,h):
        # Calculating the gradients centered about (i+1/2,j+1/2)
        self.dpsi_dx_jshift[1:-1,1:-1] = 0.5*(self.dpsi_dx[1:-1,2:] + self.dpsi_dx[1:-1,1:-1])
        self.dpsi_dy_ishift[1:-1,1:-1] = 0.5*(self.dpsi_dy[2:,1:-1] + self.dpsi_dy[1:-1,1:-1])

        # Calculating the magnitude of the shifted gradients centered at (i+1/2,j+1/2)
        mag_psi_grad = ( (self.dpsi_dx_jshift[:,:])**2 + (self.dpsi_dy_ishift[:,:])**2 )**0.5
        mag_psi_grad[mag_psi_grad==0] = np.nan

        # Calculating the psi_x and psi_y terms centered at (i+1/2,j+1/2)
        self.psi_x[1:-1,1:-1] = (self.psi[2:,2:] + self.psi[2:,1:-1] - self.psi[1:-1,2:] - self.psi[1:-1,1:-1])*(1/(2*h))
        self.psi_y[1:-1,1:-1] = (self.psi[2:,2:] - self.psi[2:,1:-1] + self.psi[1:-1,2:] - self.psi[1:-1,1:-1])*(1/(2*h))

        # Calculating Kappa
        self.kappa[1:-1,1:-1] = ( (self.psi_x[1:-1,1:-1]/mag_psi_grad[1:-1,1:-1])
                                + (self.psi_x[1:-1, :-2]/mag_psi_grad[1:-1, :-2])
                                - (self.psi_x[ :-2,1:-1]/mag_psi_grad[ :-2,1:-1])
                                - (self.psi_x[ :-2, :-2]/mag_psi_grad[ :-2, :-2])
                                + (self.psi_y[1:-1,1:-1]/mag_psi_grad[1:-1,1:-1])
                                - (self.psi_y[1:-1, :-2]/mag_psi_grad[1:-1, :-2])
                                + (self.psi_y[ :-2,1:-1]/mag_psi_grad[ :-2,1:-1])
                                - (self.psi_y[ :-2, :-2]/mag_psi_grad[ :-2, :-2]) ) * (1/(2*h))

        # Calculating the shifted kappa values
        self.kappa_ishift[1:-1,:] = 0.5*(self.kappa[2:,:] + self.kappa[1:-1,:])
        self.kappa_jshift[:,1:-1] = 0.5*(self.kappa[:,2:] + self.kappa[:,1:-1])

    def calc_F_gamma(self,gamma,M,h):
        # Calculating a f array for determining fluid properties
        # self.calc_f_arr(M,h)

        # Calculating a gradient in both directions for psi
        self.get_psi_grad(h)

        # Calculating the derivative of f
        # self.calc_f_der()
        self.calc_f_der_an(h)

        # Calculating Kappa
        self.calc_kappa(h)

        # Getting the final Force values
        self.F_gamma_x = -gamma*self.kappa_ishift*self.df_dpsi_ian[:,:]*self.dpsi_dx[:,:]
        self.F_gamma_x[np.isnan(self.F_gamma_x)] = 0
        self.F_gamma_x[np.logical_xor(self.f == 0,self.f==1)] = 0
        self.F_gamma_y = -gamma*self.kappa_jshift*self.df_dpsi_jan[:,:]*self.dpsi_dy[:,:]
        self.F_gamma_y[np.isnan(self.F_gamma_y)] = 0
        self.F_gamma_y[np.logical_xor(self.f == 0,self.f==1)] = 0

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
        self.X,self.Y = np.meshgrid(self.x_grid,self.y_grid)
        # self.X = X
        # self.Y = Y

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

        self.rho   = 1e3   # kg/m^3
        self.nu    = 1e-6  # m^2/s
        self.gamma = 0.006
        self.rho_l = 1000.
        self.rho_g = 1000.#1.0
        self.mu_l  = 1e-3#1e-3
        self.mu_g  = 1e-3#1e-8
        self.check_dt = True

    def set_bounds(self):
        # Creating a domain map
        self.domain_map = set_boundary([self.N_x,self.N_y],
                                       top   =self.top,
                                       bottom=self.bottom,
                                       left  =self.left,
                                       right =self.right)
        self.update_bound_vals()

    def update_bound_vals(self):
        # Creating an array of C's for the pressure calculation
        self.C = calc_C(self.domain_map)

        # Getting shifted domain maps for the i and j dimensions for the walls
        self.update_dm_shift()

    def update_dm_shift(self):
        # Getting shifted domain maps for the i and j dimensions for the walls
        dum_map = np.copy(self.domain_map)
        self.dm_ishift = shift_dm(dum_map,"i")
        self.dm_jshift = shift_dm(dum_map,"j")

    def draw_box(self,w,h, angle, origin,**kwargs):
        # This function is designed to draw a box on the domain with a given heighth.
        #   width, and angle. The origin of the box is the bottom left of the box
        letter = kwargs.get("letter","w")
        origin_point = kwargs.get("origin_point","bl") # Bottom Left is the standard

        # Converting the angle to radians
        theta = (angle/360) * 2*np.pi

        # Changing the origin point if the bottom left origin is not desired
        if origin_point.lower() == "br":
            origin[0] = origin[0] - w*np.sin(theta)
            origin[1] = origin[1] + w*np.cos(theta)
        if origin_point.lower() == "tl":
            origin[0] = origin[0] - h*np.cos(theta)
            origin[1] = origin[1] - h*np.sin(theta)
        if origin_point.lower() == "tr":
            origin[0] = origin[0] - h*np.cos(theta) - w*np.sin(theta)
            origin[1] = origin[1] - h*np.sin(theta) + w*np.cos(theta)


        # Defining the width and height in terms of the 'pixelated' domain
        w = w/self.h
        h = h/self.h
        # print(w,h)

        # Defining the 4 corners of the box
        P0 = origin/self.h
        P1 = P0 + [h*np.cos(theta), h * np.sin(theta)]
        P3 = P0 + [w*np.cos(theta - np.pi/2),  w * np.sin(theta - np.pi/2)]
        P2 = -P0 + P1 + P3
        P = np.array([P0,P1,P2,P3])

        L = []
        slp = []
        if angle < 90 and angle > 0:
            for i in range(len(P)):
                # Defining slopes
                slp.append( (P[i,1] - P[i-1,1])/(P[i,0] - P[i-1,0]))

                # This makes an equation to get the line x,y with any given index
                dum_eq = lambda x,y, i=i: np.array([P[i-1,0] + (y-P[i-1,1]) * (1/slp[i]), P[i-1,1] + (x-P[i-1,0])*slp[i]])

                # Output of the line is [x_inp,y_inp,(x,y)]
                #   -> L[0,:,:] = all x values
                #   -> L[1,:,:] = all y values
                L.append(dum_eq)
            # Getting the logic for the block
            grid = np.indices(self.domain_map.shape)
            x = grid[0]
            y = grid[1]

            logic0 = np.logical_and(y >= L[0](x,y)[1,:,:], x >= L[0](x,y)[0,:,:])
            logic1 = np.logical_and(y <= L[1](x,y)[1,:,:], x >= L[1](x,y)[0,:,:])
            logic2 = np.logical_and(y <= L[2](x,y)[1,:,:], x <= L[2](x,y)[0,:,:])
            logic3 = np.logical_and(y >= L[3](x,y)[1,:,:], x <= L[3](x,y)[0,:,:])
        elif angle > -90 and angle < 0:
            for i in range(len(P)):
                # Defining slopes
                slp.append( (P[i,1] - P[i-1,1])/(P[i,0] - P[i-1,0]))

                # This makes an equation to get the line x,y with any given index
                dum_eq = lambda x,y, i=i: np.array([P[i-1,0] + (y-P[i-1,1]) * (1/slp[i]), P[i-1,1] + (x-P[i-1,0])*slp[i]])

                # Output of the line is [x_inp,y_inp,(x,y)]
                #   -> L[0,:,:] = all x values
                #   -> L[1,:,:] = all y values
                L.append(dum_eq)

            # Getting the logic for the block
            grid = np.indices(self.domain_map.shape)
            x = grid[0]
            y = grid[1]

            logic0 = np.logical_and(y <= L[0](x,y)[1,:,:], x >= L[0](x,y)[0,:,:])
            logic1 = np.logical_and(y <= L[1](x,y)[1,:,:], x <= L[1](x,y)[0,:,:])
            logic2 = np.logical_and(y >= L[2](x,y)[1,:,:], x <= L[2](x,y)[0,:,:])
            logic3 = np.logical_and(y >= L[3](x,y)[1,:,:], x >= L[3](x,y)[0,:,:])
        elif angle == 90:
            for i in range(len(P)):
                if i == 0 or i == 2:
                    dum_eq = lambda x,y,i=i: np.array([x, P[i-1,1] + x * 0.])
                elif i == 1 or i == 3:
                    dum_eq = lambda x,y,i=i: np.array([P[i-1,0] + y * 0., y])
                # This makes an equation to get the line x,y with any given index
                # dum_eq = lambda x,y, i=i: np.array([P[i-1,0] + (y-P[i-1,1]) * (1/slp[i]), P[i-1,1] + (x-P[i-1,0])*slp[i]])

                # Output of the line is [x_inp,y_inp,(x,y)]
                #   -> L[0,:,:] = all x values
                #   -> L[1,:,:] = all y values
                L.append(dum_eq)
            # Getting the logic for the block
            grid = np.indices(self.domain_map.shape)
            x = grid[0]
            y = grid[1]

            logic0 = np.logical_and(y >= L[0](x,y)[1,:,:], x >= L[0](x,y)[0,:,:])
            logic1 = np.logical_and(y <= L[1](x,y)[1,:,:], x >= L[1](x,y)[0,:,:])
            logic2 = np.logical_and(y <= L[2](x,y)[1,:,:], x <= L[2](x,y)[0,:,:])
            logic3 = np.logical_and(y >= L[3](x,y)[1,:,:], x <= L[3](x,y)[0,:,:])
        elif angle == 0:
            for i in range(len(P)):
                if i == 1 or i == 3:
                    dum_eq = lambda x,y,i=i: np.array([x, P[i-1,1] + x * 0.])
                elif i == 0 or i == 2:
                    dum_eq = lambda x,y,i=i: np.array([P[i-1,0] + y * 0., y])
                # This makes an equation to get the line x,y with any given index
                # dum_eq = lambda x,y, i=i: np.array([P[i-1,0] + (y-P[i-1,1]) * (1/slp[i]), P[i-1,1] + (x-P[i-1,0])*slp[i]])

                # Output of the line is [x_inp,y_inp,(x,y)]
                #   -> L[0,:,:] = all x values
                #   -> L[1,:,:] = all y values
                L.append(dum_eq)
            # Getting the logic for the block
            grid = np.indices(self.domain_map.shape)
            x = grid[0]
            y = grid[1]

            logic0 = np.logical_and(y >= L[0](x,y)[1,:,:], x >= L[0](x,y)[0,:,:])
            logic1 = np.logical_and(y <= L[1](x,y)[1,:,:], x >= L[1](x,y)[0,:,:])
            logic2 = np.logical_and(y <= L[2](x,y)[1,:,:], x <= L[2](x,y)[0,:,:])
            logic3 = np.logical_and(y >= L[3](x,y)[1,:,:], x <= L[3](x,y)[0,:,:])


        # Getting a mesh of the grid indices flip([1].T) = y, [1].T = x

        logic_all = np.logical_and(np.logical_and(np.logical_and(
                             logic0,
                             logic1),
                             logic2),
                             logic3)

        # Changing the domain map
        self.domain_map[logic_all] = letter


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

        # Initializing Property Arrays
        self.rho = np.zeros(dc.domain_map.shape) + dc.rho_l
        self.rho_ishift = np.copy(self.rho[:,:])
        self.rho_jshift = np.copy(self.rho[:,:])
        self.mu  = np.zeros(dc.domain_map.shape) + dc.mu_l
        self.mu_ishift  = np.copy(self.mu[:,:])
        self.mu_jshift  = np.copy(self.mu[:,:])

        # Intializing the velocity Array
        self.u = np.zeros((dc.N_x + 2, dc.N_y + 2))
        self.v = np.zeros((dc.N_x + 2, dc.N_y + 2))

        # Setting the time variable
        self.t = dc.t

        # Setting the Pressure Gradients
        self.dP_x = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.dP_x
        self.dP_y = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.dP_y

        # Setting the intial pressure values
        self.P = np.zeros((dc.N_x + 2, dc.N_y + 2)) + dc.P

        # Setting the boundaries on the initial Pressure Arrays
        self.dP_x = set_ghost(dc.domain_map, self.dP_x, dc.u_B, type="p",source=dc.dP_x_S)
        self.dP_y = set_ghost(dc.domain_map, self.dP_y, dc.v_B, type="p",source=dc.dP_y_S)
        self.P = set_ghost(dc.domain_map, self.P, dc.v_B, type="p",source=dc.P_S)

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

        # Creating the temporary flow value Arrays
        # Preallocating arrays
        numpy_dtype = np.float64
        self.A_x = np.zeros(self.u.shape, dtype=numpy_dtype)
        self.A_y = np.zeros(self.v.shape, dtype=numpy_dtype)
        self.D_x = np.zeros(self.u.shape, dtype=numpy_dtype)
        self.D_y = np.zeros(self.v.shape, dtype=numpy_dtype)
        self.u_star = np.zeros(self.u.shape, dtype=numpy_dtype)
        self.v_star = np.zeros(self.v.shape, dtype=numpy_dtype)
        self.u_ishift = np.zeros(self.u.shape, dtype=numpy_dtype) + dc.u_init
        self.v_jshift = np.zeros(self.v.shape, dtype=numpy_dtype) + dc.v_init
        self.u_ishift_star = np.zeros(self.u.shape, dtype=numpy_dtype)
        self.v_jshift_star = np.zeros(self.v.shape, dtype=numpy_dtype)
        self.u_new = np.zeros(self.u.shape, dtype=numpy_dtype)
        self.v_new = np.zeros(self.v.shape, dtype=numpy_dtype)

        # Setting the boundaries on the initial velocity array
        self.u_ishift = set_ghost(dc.dm_ishift, self.u_ishift, dc.u_B,type="u",source=dc.u_S)
        self.v_jshift = set_ghost(dc.dm_jshift, self.v_jshift, dc.v_B,type="v",source=dc.v_S)

        # Calculating the cell velocities
        self.u[1:-1,1:-1] = u_new_fun(self.u_ishift)
        self.v[1:-1,1:-1] = v_new_fun(self.v_jshift)

class save_class:
    def __init__(self,flow_class, bubble_class):
        # Creating lists to store all of the variables
        numpy_dtype = np.float32
        self.u_list = np.array(flow_class.u, dtype=numpy_dtype)
        self.v_list = np.array(flow_class.v, dtype=numpy_dtype)
        self.t_list = np.array(flow_class.t, dtype=numpy_dtype)
        self.dP_x_list = np.array(flow_class.dP_x, dtype=numpy_dtype)
        self.dP_y_list = np.array(flow_class.dP_y, dtype=numpy_dtype)
        self.P_list = np.array(flow_class.P, dtype=numpy_dtype)
        self.psi_list = np.array(bubble_class.psi, dtype=numpy_dtype)


        # This is a counting variable for saving hdf5 file
        self.save_count = 0

    def save_values(self,elapsed_time, dc,fc,bc,oc):
        # Appending values to the lists for storage
        self.u_list = np.dstack((self.u_list, fc.u))
        self.v_list = np.dstack((self.v_list, fc.v))
        self.t_list = np.append(self.t_list, fc.t)
        self.dP_x_list = np.dstack((self.dP_x_list, fc.dP_x))
        self.dP_y_list = np.dstack((self.dP_y_list, fc.dP_y))
        self.P_list = np.dstack((self.P_list, fc.P))
        self.psi_list = np.dstack((self.psi_list, bc.psi))

        # Calculating the size of all of the arrays
        current_size = (self.u_list.nbytes + self.v_list.nbytes +
                        self.dP_x_list.nbytes + self.dP_y_list.nbytes + self.P_list.nbytes + self.psi_list.nbytes)

        # Saving an HDF5 File if the size of the arrays reaches a certain number of bytes
        #   or the write interval is reached
        if self.save_count * oc.write_interval <= fc.t or fc.t >= dc.T or current_size >= oc.max_size:
            if self.save_count == 0:
                # --> This occurs on the first save and creates the file
                hf = h5py.File(oc.output_file, "w")
                # These change every time loop
                hf.create_dataset("u", data=self.u_list, maxshape=(fc.u.shape[0],fc.u.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("v", data=self.v_list, maxshape=(fc.v.shape[0],fc.v.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("dP_x", data=self.dP_x_list, maxshape=(fc.u.shape[0],fc.u.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("dP_y", data=self.dP_y_list, maxshape=(fc.v.shape[0],fc.v.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("P", data=self.P_list, maxshape=(fc.u.shape[0],fc.u.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("psi", data=self.psi_list, maxshape=(fc.u.shape[0],fc.u.shape[1],None), compression="gzip", chunks=True)
                hf.create_dataset("t", data=self.t_list, maxshape=(None,), compression="gzip", chunks=True)

                hf.create_dataset("x", data=dc.x_grid, maxshape=dc.x_grid.shape, compression="gzip")
                hf.create_dataset("y", data=dc.y_grid, maxshape=dc.y_grid.shape, compression="gzip")
                hf.close()
                self.u_list = self.u_list[:,:,-1]
                self.v_list = self.v_list[:,:,-1]
                self.dP_x_list = self.dP_x_list[:,:,-1]
                self.dP_y_list = self.dP_y_list[:,:,-1]
                self.P_list = self.P_list[:,:,-1]
                self.psi_list = self.psi_list[:,:,-1]
                self.t_list = [-1]
            else:
                # --> This opens the file and appends to the current arrays
                hf = h5py.File(oc.output_file, "a")

                # Pulling in and resizing the arrays
                hf["u"].resize(hf["u"].shape[2] + self.u_list.shape[2]-1, axis=2)
                hf["u"][:,:,-(self.u_list.shape[2]-1):] = self.u_list[:,:,1:]
                self.u_list = self.u_list[:,:,-1]

                hf["v"].resize(hf["v"].shape[2] + self.v_list.shape[2]-1, axis=2)
                hf["v"][:,:,-(self.v_list.shape[2]-1):] = self.v_list[:,:,1:]
                self.v_list = self.v_list[:,:,-1]

                hf["dP_x"].resize(hf["dP_x"].shape[2] + self.dP_x_list.shape[2]-1, axis=2)
                hf["dP_x"][:,:,-(self.dP_x_list.shape[2]-1):] = self.dP_x_list[:,:,1:]
                self.dP_x_list = self.dP_x_list[:,:,-1]

                hf["dP_y"].resize(hf["dP_y"].shape[2] + self.dP_y_list.shape[2]-1, axis=2)
                hf["dP_y"][:,:,-(self.dP_y_list.shape[2]-1):] = self.dP_y_list[:,:,1:]
                self.dP_y_list = self.dP_y_list[:,:,-1]

                hf["P"].resize(hf["P"].shape[2] + self.P_list.shape[2]-1, axis=2)
                hf["P"][:,:,-(self.P_list.shape[2]-1):] = self.P_list[:,:,1:]
                self.P_list = self.P_list[:,:,-1]

                hf["psi"].resize(hf["psi"].shape[2] + self.psi_list.shape[2]-1, axis=2)
                hf["psi"][:,:,-(self.psi_list.shape[2]-1):] = self.psi_list[:,:,1:]
                self.psi_list = self.psi_list[:,:,-1]

                hf["t"].resize(hf["t"].shape[0] + self.t_list.shape[0]-1, axis=0)
                hf["t"][-(self.t_list.shape[0]-1):] = self.t_list[1:]
                self.t_list = self.t_list[-1]

                hf.close()

            self.save_count += 1
            print("\n")
            print("\tWrote Output at t = ", round(fc.t,5))
            print("\tReal Elapsed Time =", elapsed_time(oc.real_start_time))
            print("\tdt = ", dc.dt)
            print("")

class option_class:
    def __init__(self):
        self.pressure_solve = "value"
        self.output_file = "./Output/testing/run.h5"
        self.show_progress = False
        self.write_interval = 120
        self.dt_multiplier = 0.50

        # Minimum and Maximum Time Step
        self.dt_max = 0.05
        self.dt_min = 1e-30

        # Number of bytes that is allowed to be stored locally
        self.max_size = 10e6

        self.real_start_time = time.time()

        # Set number of loops for the pressure convergence
        self.min_Ploops = int(10)
        self.max_Ploops = int(1000)
        self.Ptol = 1e-1

        self.M = 3
