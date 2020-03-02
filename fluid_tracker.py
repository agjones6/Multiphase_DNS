# The goal of this script is to track specific 'fluid particles'

import scipy
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import h5py
import numpy as np

x = [1, 2, 1, 2]
y = [1, 1, 2, 2]
u = [6, -6, 6, -6]
v = [6, 6, -6, -6]

f_u = interp2d(x,y,u,kind='linear')
f_v = interp2d(x,y,v,kind='linear')

x_1, y_1 = 1.5,1.51
u_1 = f_u(x_1, y_1)
v_1 = f_v(x_1, y_1)


# Defining the hdf5 file
mb_num = 9
skip_num = 1
my_dpi = 400
my_fps = 25
my_file    = "./Output/testing/run_" + str(mb_num) + ".h5"
video_name = "./Videos/testing/run_" + str(mb_num) + ".mp4"

show_fig = True
save_fig = False

# Importing the h5 file
hf = h5py.File(my_file, "r")

# Getting all of the object names in the hdf5 file
data_names = hf.keys()

# Pulling the data
    # NOTE: the matrices are 3-D pulled in as (x, y, t)
u = hf["u"]
v = hf["v"]
t = hf["t"]
dP_x = hf["dP_x"]
dP_y = hf["dP_y"]
x = hf["x"]
y = hf["y"]

P_loc = [0.01,0.005]
P_loc_array = np.array(P_loc)

for k in range(len(t)-1):

    interp_u = interp2d(x, y, u[:,:,k],kind="linear")
    interp_v = interp2d(x, y, v[:,:,k],kind="linear")

    new_x = interp_u(P_loc[0],P_loc[1]) * (t[k+1] - t[k] ) + P_loc[0]
    new_y = interp_v(P_loc[0],P_loc[1]) * (t[k+1] - t[k] ) + P_loc[1]

    if new_x > x[-1] or new_x < x[0] or new_y > y[-1] or new_y < y[0]:
        continue
    else:
        P_loc = np.array((new_x[0], new_y[0]))

    P_loc_array = np.dstack((P_loc_array, P_loc))

    # print(k)
    # print(P_loc)
    plt.cla()
    plt.plot(P_loc[0],P_loc[1],'.k')
    plt.xlim((x[0],x[-1]))
    plt.ylim((y[0],y[-1]))
    plt.title(t[k])
    plt.pause(0.00001)
    # exit()
