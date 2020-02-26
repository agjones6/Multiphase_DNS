#
# This file is made to read the hdf5 file output from the fluid simulation code
#   written by Andy Jones
#

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def make_plot(mp, x, y, u, v, **kwargs):
    # Handling optional arguments
    LB = kwargs.get("LB",[])
    UB = kwargs.get("UB",[])
    plot_type = kwargs.get("plot_type","profile")
    sub_type = kwargs.get("sub_type",["u","y"])
    show_0 = kwargs.get("show_0","")
    clear_plot = kwargs.get("clear_plot",True)

    # Plotting the velocity profile
    # --> Plotting
    if clear_plot:
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
        mp.quiver(x,y,u,v,M,linewidth=0.1,edgecolor=(0,0,0),cmap="jet")#,scale_units="xy")

    elif plot_type.lower() == "surf":
        x,y = np.meshgrid(x,y)
        mag = (u**2 + v**2)**0.5
        mp.plot_surface(x,y,mag,cmap="jet")

# --> Defining the hdf5 file reading variables
mb_num = 30
skip_num = 1
my_dpi = 400
my_fps = 25
my_file    = "./Output/hwk3/N_" + str(mb_num) + "_fromPeak.h5"
video_name = "./Videos/hwk3/N_" + str(mb_num) + ".mp4"
# my_file    = "./Output/MB_" + str(mb_num) + ".h5"
# video_name = "./Videos/MB_" + str(mb_num) + ".mp4"

# Options to show a video or save a video
show_fig = True
save_fig = False

# Importing the h5 file
hf = h5py.File(my_file, "r")

# Getting all of the object names in the hdf5 file
data_names = hf.keys()

# Pulling the data
    # NOTE: the matrices are 3-D pulled in as (x, y, t)
u = np.array(hf["u"])
v = np.array(hf["v"])
t = np.array(hf["t"])
dP_x = np.array(hf["dP_x"])
dP_y = np.array(hf["dP_y"])
x = np.array(hf["x"])
y = np.array(hf["y"])
# Initializing Writer
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

# --> Finding the time of being steady state
tol = 1e-6

# --> Getting time derivative a specific x location in the flow to find steady state
u_der = (u[3, :, 1:] - u[3, :, :-1])/(t[1:] - t[:-1])
for i in range(u_der.shape[1]):
    if np.all(u_der[:,i] <= tol):
        t_max_index = i
        t_max = t[i]
        break

# Making the plots for Homework 3
# Fluid Properties
u_max = 0.03    # m/s
y_min = 0.005   # m
rho   = 1e3   # kg/m^3
nu    = 1e-6  # m^2/s
dP_analytic  = -u_max * 2 * rho * nu / y_min**2 # [Pa/m]

# Analytic Solution
u_fun = lambda y: (-u_max/y_min**2)*y**2 + u_max

# Calculating the Staedy State Velocity Field Distribution
u_vals = u_fun(y-0.005)
u_analytic_mean = np.mean(u_vals)

# Function for plotting
def hwk3_plt(plt_i,i):
    plt_i.plot(u_vals, y, '--k')
    make_plot(plt_i, x[1:-1], y[1:-1],
              u[3,1:-1,i].T,
              v[3,1:-1,i].T,
              plot_type="profile",
              sub_type=["u","y"],
              show_0="x",
              clear_plot=False
              )
    # Getting the error
    L2 = ( (1/mb_num)*np.sum((u[3,:,i] - u_vals)**2))**0.5

    plt_i.set_title("t = " + str(round(t[i],4)) + "s | L^2 = " + str(round(L2,8)))
    plt_i.set_xlim([0,0.035])

# Creating subplots
fig = plt.figure()
plt_1 = fig.add_subplot(4,2,1)
plt_2 = fig.add_subplot(4,2,2)
plt_3 = fig.add_subplot(4,2,3)
plt_4 = fig.add_subplot(4,2,4)
plt_5 = fig.add_subplot(4,2,5)
plt_6 = fig.add_subplot(4,2,6)
plt_7 = fig.add_subplot(4,2,7)
plt_8 = fig.add_subplot(4,2,8)

# Plotting different steps in a single figure
hwk3_plt(plt_1, 0)
hwk3_plt(plt_2, 6)
hwk3_plt(plt_3, t_max_index//60)
hwk3_plt(plt_4, t_max_index//30)
hwk3_plt(plt_5, t_max_index//15)
hwk3_plt(plt_6, t_max_index//8)
hwk3_plt(plt_7, int(t_max_index//1.5)) #int(len(t)//1.5))
hwk3_plt(plt_8, t_max_index)

# Setting the Labels
plt_1.set_ylabel("y (m)")
plt_3.set_ylabel("y (m)")
plt_5.set_ylabel("y (m)")
plt_7.set_ylabel("y (m)")
plt_7.set_xlabel("u (m/s)")
plt_8.set_xlabel("u (m/s)")

plt.show()

