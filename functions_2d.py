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


@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," + data_type + "[:,:]," + data_type + "," + data_type + "[:,:])"],
      nopython=True, target=my_target)
def vel_star(vel, dt, A, nu, D):
    return vel + dt * ( -A + nu * D )

@jit(["" + data_type + "[:,:](" + data_type + "[:,:]," + data_type + "," + data_type + "," + data_type + "," + data_type + "[:,:]," + data_type + ")"],
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
            "float64[:,:],float64,float64,float64,float64[:,:],float64[:,:])"],nopython=True,target=my_target)
def loop_Pressure(tol,max_loops,min_loops,wall_0s,P_old,P_new,
                    C, rho, h, dt, u_ishift_star, v_jshift_star):
    conv = False
    count = 0

    while not conv and count < max_loops or count <= min_loops:
        # Calculating the new Pressure scalar field
        P_new[1:-1,1:-1] = P_new_fun(C, P_old, rho, h, dt, u_ishift_star, v_jshift_star)
        # P_new = set_ghost(dc.domain_map, P_new, dc.u_B, h=dc.h, type="p",source=dc.P_S)
        # plt.figure()
        # plt.contourf(P_new)
        P_new = P_new*wall_0s
        # plt.figure()
        # plt.contourf(P_new)
        # plt.show()
        # exit()
        # Checking if the current time step converges with the new step
        conv = check_conv(P_old[1:-1,1:-1], P_new[1:-1,1:-1], tol)

        # Normalizing the pressure back to absolute
        # P_new = np.absolute((P_new / np.amax(P_new)) * dc.P)
        P_old = np.copy(P_new[:,:])

        count += 1
    # if count > 1:
    #     print("pressure took ",count," to converge")

    return P_new

def calc_pressure(dc, oc, P, u_ishift_star, v_jshift_star):

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
                dc.C, dc.rho, dc.h, dc.dt, u_ishift_star, v_jshift_star)

    return np.copy(P_new)

def str_index(map,str):
    map = np.array(map,dtype="str")
    return np.core.defchararray.find(map,str)!=-1

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

        self.rho   = 1e3   # kg/m^3
        self.nu    = 1e-6  # m^2/s
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
    def __init__(self,flow_class):
        # Creating lists to store all of the variables
        numpy_dtype = np.float32
        self.u_list = np.array(flow_class.u, dtype=numpy_dtype)
        self.v_list = np.array(flow_class.v, dtype=numpy_dtype)
        self.t_list = np.array(flow_class.t, dtype=numpy_dtype)
        self.dP_x_list = np.array(flow_class.dP_x, dtype=numpy_dtype)
        self.dP_y_list = np.array(flow_class.dP_y, dtype=numpy_dtype)
        self.P_list = np.array(flow_class.P, dtype=numpy_dtype)

        # This is a counting variable for saving hdf5 file
        self.save_count = 0

    def save_values(self,elapsed_time, dc,fc,oc):
        # Appending values to the lists for storage
        self.u_list = np.dstack((self.u_list, fc.u))
        self.v_list = np.dstack((self.v_list, fc.v))
        self.t_list = np.append(self.t_list, fc.t)
        self.dP_x_list = np.dstack((self.dP_x_list, fc.dP_x))
        self.dP_y_list = np.dstack((self.dP_y_list, fc.dP_y))
        self.P_list = np.dstack((self.P_list, fc.P))

        # Calculating the size of all of the arrays
        current_size = (self.u_list.nbytes + self.v_list.nbytes +
                        self.dP_x_list.nbytes + self.dP_y_list.nbytes + self.P_list.nbytes)

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
                hf.create_dataset("t", data=self.t_list, maxshape=(None,), compression="gzip", chunks=True)

                hf.create_dataset("x", data=dc.x_grid, maxshape=dc.x_grid.shape, compression="gzip")
                hf.create_dataset("y", data=dc.y_grid, maxshape=dc.y_grid.shape, compression="gzip")
                hf.close()
                self.u_list = self.u_list[:,:,-1]
                self.v_list = self.v_list[:,:,-1]
                self.dP_x_list = self.dP_x_list[:,:,-1]
                self.dP_y_list = self.dP_y_list[:,:,-1]
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
