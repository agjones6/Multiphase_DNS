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


def make_plot(fig, mp, x, y, u, v, **kwargs):
    # Handling optional arguments
    LB = kwargs.get("LB",[])
    UB = kwargs.get("UB",[])
    plot_type = kwargs.get("plot_type","profile")
    sub_type = kwargs.get("sub_type",["u","y"])
    show_0 = kwargs.get("show_0","")
    clear_plot = kwargs.get("clear_plot",True)
    sl_density = kwargs.get("sl_density",1.0)

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

    elif plot_type.lower() == "streamline":
        X,Y = np.meshgrid(x,y)
        M = np.hypot(u, v)
        speed = np.sqrt(u**2 + v**2)
        # lw = 10 * (speed / speed.max())
        # mp.streamplot(X,Y, u, v, density=sl_density, color=M, cmap="jet")
        # print(M.shape)
        # strm = mp.streamplot(x, y, u, v, linewidth=lw, cmap="jet")
        strm = mp.streamplot(x, y, u, v, density=sl_density, color=M, cmap="jet")
        # fig.colorbar(strm.lines)
        mp.set_xlim([x[0],x[-1]])
        mp.set_ylim([y[0],y[-1]])

    elif plot_type.lower() == "surf":
        x,y = np.meshgrid(x,y)
        mag = (u**2 + v**2)**0.5
        mp.plot_surface(x,y,mag,cmap="jet")

    elif plot_type.lower() == "contourf":

        mag = (u**2 + v**2)**0.5
        mp.contourf(x,y,mag)

# --> Defining the hdf5 file reading variables
mb_num = 40
skip_num = 0
my_dpi = 400
my_fps = 1
my_file    = "./Output/hwk4/run_" + str(mb_num) + ".h5"
video_name = "./Videos/hwk4/run_" + str(mb_num) + ".mp4"
# my_file    = "./Output/MB_" + str(mb_num) + ".h5"
# video_name = "./Videos/MB_" + str(mb_num) + ".mp4"

# Options to show a video or save a video
show_fig = True
save_fig = False

# Making sure the file opens
num_tries = 0
e = 1
while e == 1 and num_tries < 10:
    try:
        # Importing the h5 file
        hf = h5py.File(my_file, "r")
        e = 0
    except:
        e = 1
        num_tries += 1
        print(num_tries)
        pass

# Initializing Writer
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

# Getting all of the object names in the hdf5 file
data_names = hf.keys()

# Pulling the data
# NOTE: the matrices are 3-D pulled in as (x, y, t)
u = hf["u"]
v = hf["v"]
t = hf["t"]
dP_x = hf["dP_x"]
dP_y = hf["dP_y"]
P = hf["P"]
x = hf["x"]
y = hf["y"]

# Choosing the starting index
i_start = len(t)//1.2
i_start = 0

# print(dP_x[5,1:-1,1])
# Plotting a specific x location of u Velocity at a ceratin time
# plt.figure()
# plt.plot(u[1:-1,1:-1,-1].T,y)
# plt.plot(np.mean(u[1:-1,1:-1,-1].T,axis=1),y,'k--',linewidth=3)
# # plt.figure()
# # plt.plot(x,dP_x[1:-1,1:-1,-1])
# plt.show()
# exit()

# fig = plt.figure()
# my_plot1 = fig.add_subplot(2,1,1)#,projection='3d')
# my_plot4 = fig.add_subplot(2,1,2)#,projection='3d')
# my_plot2 = fig.add_subplot(2,2,3)
# my_plot3 = fig.add_subplot(2,2,4)
def animate(i):
    # Calculating the index
    i = i + (i) * skip_num + i_start
    if i >= len(t)-1:
        i = len(t)-1
    # if i <= 1:
    #     i = 1

    u_i = u[1:-1,1:-1,i]#/np.max(u[:,:,i])
    v_i = v[1:-1,1:-1,i]#/np.max(v[1:-1,1:-1,i])
    dP_x_i = dP_x[1:-1,1:-1,i]
    dP_y_i = dP_y[1:-1,1:-1,i]
    max_P = np.amax(P[1:-1,1:-1,i])
    print(max_P)
    if max_P != 0:
        P_i = P[1:-1,1:-1,i]/max_P
    else:
        P_i = P[1:-1,1:-1,i]

    my_plot1.clear()
    try:
        make_plot(fig, my_plot1, x[1:-1], y[1:-1],
              u_i.T,
              v_i.T,
              plot_type="field",
              sl_density=[1.0, 0.5],
              sub_type=["u","y"],
              show_0="x"
              )
    except:
        pass
    # M = np.hypot(u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T)
    # my_plot1.quiver(x, y, u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T, M, linewidth=0.1, edgecolor=(0,0,0),cmap="jet")
    my_plot1.set_title(str(round(t[i],6)))

    # --> PRESSURE GRADIENT FIELD
    my_plot4.clear()
    try:
        make_plot(fig, my_plot4, x[1:-1], y[1:-1],
              P_i.T,
              P_i.T,
              plot_type="contourf",
              sl_density=[0.9,1.0],
              sub_type=[]
              )
    except Exception as e:
        print(e)
        exit()
        pass

    # --> PRESSURE GRADIENT FIELD
    # my_plot4.clear()
    # try:
    #     make_plot(fig, my_plot4, x, y,
    #           dP_x_i.T,
    #           dP_y_i.T,
    #           plot_type="field",
    #           sl_density=[0.9,1.0],
    #           sub_type=[]
    #           )
    # except:
    #     pass

    # my_plot4.clear()
    # make_plot(my_plot4, x, y,
    #           u[:,:,i].T,
    #           v[:,:,i].T,
    #           plot_type="profile",
    #           sub_type=["u","y"],
    #           show_0="x"
    #           )
    #
    # make_plot(my_plot3, x, y,
    #           u[i,1:-1,1:-1],
    #           v[i,1:-1,1:-1],
    #           plot_type="profile",
    #           sub_type=["x","v"],
    #           show_0="y"
    #           )
    # return u[i,1:-1,1:-1]


# -15
# x_vls = np.argwhere(x < 0.02)[-1][0]

# x_vls = -1
#
# t_i = len(t)//20
# # t_i = -1
# make_plot(fig, my_plot1,  x[1:x_vls], y[1:-1],
#       u[1:x_vls,1:-1,t_i].T,
#       v[1:x_vls,1:-1,t_i].T,
#       plot_type="streamline",
#       sl_density=[1.0, 0.5],
#       sub_type=["u","y"]
#       # show_0="x"
#       )
# make_plot(fig, my_plot4, x[1:x_vls], y[1:-1],
#       P[1:x_vls,1:-1,t_i].T,
#       P[1:x_vls,1:-1,t_i].T,
#       plot_type="contourf",
#       sl_density=[1.0, 0.5],
#       sub_type=["x", "u"]
#       # show_0="x"
#       )
# # my_plot1.plot(np.mean(u[1:x_vls,1:-1,t_i].T,axis=1),y,'k--',linewidth=3)
# my_plot1.set_title("t = " + str(round(t[t_i],4)) + "s")
# plt.show()
# exit()

tol = 1e-10
# --> Getting time derivative a specific x location in the flow to find steady state
u_der = (u[3, :, 1:] - u[3, :, :-1])/(t[1:] - t[:-1])
for i in range(len(t)//2,u_der.shape[1]):
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
u_vals = u_fun(y[:]-y_min)
u_analytic_mean = np.mean(u_vals)

# Function for plotting
def hwk3_plt(fig,plt_i,i):
    plt_i.plot(u_vals, y, '--k')
    make_plot(fig,plt_i, x[1:-1], y[1:-1],
              u[3,1:-1,i].T,
              v[3,1:-1,i].T,
              plot_type="profile",
              sub_type=["u","y"],
              show_0="x",
              clear_plot=False
              )
    # Getting the error
    L2 = ( (1/mb_num)*np.sum((u[3,:,i] - u_vals)**2))**0.5

    plt_i.set_title("t = " + str(round(t[i],4)) + "s")# | L^2 = " + str(round(L2,8)))
    plt_i.set_xlim([-0.012,0.035])

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
hwk3_plt(fig,plt_1, 0)
hwk3_plt(fig,plt_2, 15)
hwk3_plt(fig,plt_3, t_max_index//60)
hwk3_plt(fig,plt_4, t_max_index//30)
hwk3_plt(fig,plt_5, t_max_index//15)
hwk3_plt(fig,plt_6, t_max_index//8)
hwk3_plt(fig,plt_7, int(t_max_index//1.5)) #int(len(t)//1.5))
hwk3_plt(fig,plt_8, t_max_index)

# Setting the Labels
plt_1.set_ylabel("y (m)")
plt_3.set_ylabel("y (m)")
plt_5.set_ylabel("y (m)")
plt_7.set_ylabel("y (m)")
plt_7.set_xlabel("u (m/s)")
plt_8.set_xlabel("u (m/s)")

plt.show()
exit()

ani = FuncAnimation(fig,animate,frames=(len(t)//skip_num))

if show_fig:
    plt.show()
if save_fig:
    ani.save(video_name,writer="ffmpeg", dpi=my_dpi, fps=my_fps)#,writer='matplotlib.animation.PillowWriter')#,writer=writer,dpi=100)
# exit()
