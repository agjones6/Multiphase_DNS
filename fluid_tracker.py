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
mb_num = 13
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

P_loc = [[0.01,0.005],
         [0.02,0.005]]
# P_x = np.array([0.01 , 0.02])
# P_y = np.array([0.005, 0.005])
P_x = np.linspace(x[2],x[-2],20)
P_y = np.linspace(y[2],y[-2],20)
# P_x = np.array([0.001,0.02])
# P_y = np.array([0.001,0.02])
P_x, P_y = np.meshgrid(P_x,P_y)
P_x = P_x.flatten()
P_y = P_y.flatten()
# print(P_x)
# print(P_y)
# exit()
# P_x = np.flatten(P_x)
# P_y = np.flatten(P_y)
P_x_array = np.array(P_x)
P_y_array = np.array(P_y)
# print(P_x_array.shape)
# exit()
print(x.shape,y.shape,u.shape)
k_prev = 0
t_P = [t[k_prev]]
dx = P_x*0

dy = P_y*0
factor = 1
for k in range(k_prev,(len(t)-1),50):

    interp_u = interp2d(x, y, u[:,:,k].T,kind="linear")
    interp_v = interp2d(x, y, v[:,:,k].T,kind="linear")
    # interp_u = scipy.interpolate.RectBivariateSpline(x, y, u[:,:,k])
    # interp_v = scipy.interpolate.RectBivariateSpline(x, y, v[:,:,k])

    # Grabbing the interpolated values using the functions made.
        # NOTE: The functions return ALL combinations of the intput values.
        # Currently, I will just use the primary combination in the first row
    # print(interp_u(P_x[0],P_y[0])[0,:])

    # dx_test = interp_u(P_x, P_y)[0,:] * (t[k] - t[k_prev] )
    # dy = interp_v(P_x, P_y)[:,0] * (t[k] - t[k_prev] )
    # dx_test = np.fliplr(interp_u(P_x, P_y)).T.diagonal() * (t[k] - t[k_prev] )
    # dy = interp_v(P_x, P_y).diagonal() * (t[k] - t[k_prev] )
    for i in range(len(P_x)):
        dx[i] = interp_u(P_x[i], P_y[i]) * (t[k] - t[k_prev] ) * factor
        dy[i] = interp_v(P_x[i], P_y[i]) * (t[k] - t[k_prev] ) * factor



    # print("")
    # print(dx)
    # print(dy)
    # print("")

    # Updating the position arrays
    P_x = P_x + dx
    P_y = P_y + dy
    # Fixing the boundaries so that whatever goes out one side comes in the other
    P_x[P_x > x[-2]] = x[2] #x[len(x)//2]
    P_x[P_x < x[1]]  = x[-3] #x[len(x)//2]


    P_y[P_y > y[-2]] = y[2]#y[len(y)//2]
    P_y[P_y < y[1]]  = y[-3]#y[len(y)//2]

    # if new_x > x[-2] or new_x < x[1] or new_y > y[-2] or new_y < y[1]:
    #     break
    # else:
    #     P_loc = np.array((new_x[0], new_y[0]))


    # print(P_x.shape)
    # print(P_x_array.shape)
    P_x_array = np.dstack((P_x_array, P_x))
    P_y_array = np.dstack((P_y_array, P_y))
    t_P.append(t[k+1])
    k_prev = k

    # print(k)
    # print(P_loc)
    plt.cla()
    plt.plot(P_x,P_y,'.')
    plt.xlim((x[0],x[-1]))
    plt.ylim((y[0],y[-1]))
    plt.title(t[k])
    plt.pause(0.00001)
    # plt.show()
    # exit()
# P_loc_array = np.squeeze(P_loc_array)
t_P = np.array(t_P)

P_x_array = np.squeeze(P_x_array)
P_y_array = np.squeeze(P_y_array)

print(P_x_array)
print(P_x_array.shape)
plt.figure()
plt.plot(t_P,P_x_array[:,:].T)
plt.show()
