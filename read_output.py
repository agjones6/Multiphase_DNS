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
        mp.quiver(x,y,u,v,M,linewidth=0.1,edgecolor=(0,0,0),cmap="jet")#,scale_units="xy")

    elif plot_type.lower() == "surf":
        x,y = np.meshgrid(x,y)
        mag = (u**2 + v**2)**0.5
        mp.plot_surface(x,y,mag,cmap="jet")

# Defining the hdf5 file
mb_num = 21
skip_num = 1
my_dpi = 400
my_fps = 25
my_file    = "./Output/MB_" + str(mb_num) + ".h5"
video_name = "./Videos/MB_" + str(mb_num) + ".mp4"

# Importing the h5 file
hf = h5py.File(my_file, "r")

# Getting all of the object names in the hdf5 file
data_names = hf.keys()

# Pulling the data
    # NOTE: the matrices are 3-D pulled in as (t, x, y)
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

fig = plt.figure()
my_plot1 = fig.add_subplot(2,1,1,projection='3d')
my_plot4 = fig.add_subplot(2,1,2,projection='3d')
# my_plot2 = fig.add_subplot(2,2,3)
# my_plot3 = fig.add_subplot(2,2,4)
def animate(i):
    i = i + (i) * skip_num
    if i >= len(t)-1:
        i = len(t)-1
    my_plot1.clear()
    make_plot(my_plot1, x, y,
              u[i,1:-1,1:-1].T,
              v[i,1:-1,1:-1].T,
              plot_type="surf",
              sub_type=[]
              )
    # M = np.hypot(u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T)
    # my_plot1.quiver(x, y, u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T, M, linewidth=0.1, edgecolor=(0,0,0),cmap="jet")
    my_plot1.set_title(str(round(t[i],6)))

    # --> PRESSURE GRADIENT FIELD
    my_plot4.clear()
    make_plot(my_plot4, x, y,
              dP_x[i,1:-1,1:-1].T,
              dP_y[i,1:-1,1:-1].T,
              plot_type="surf",
              sub_type=[]
              )

    # make_plot(my_plot2, x, y,
    #           u[i,1:-1,1:-1].T,
    #           v[i,1:-1,1:-1].T,
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

ani = FuncAnimation(fig,animate,frames=(len(t)//skip_num))

plt.show()
# ani.save(video_name,writer="ffmpeg", dpi=my_dpi, fps=my_fps)#,writer='matplotlib.animation.PillowWriter')#,writer=writer,dpi=100)
exit()
