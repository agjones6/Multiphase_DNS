#
# This file is made to read the hdf5 file output from the fluid simulation code
#   written by Andy Jones
#

import h5py
import numpy as np
import matplotlib as mpl
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
    norm = kwargs.get("norm",False)
    show_cbar = kwargs.get("show_cbar",False)
    show_block = kwargs.get("show_block",None)

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
        if not show_block is None:
            # u = np.ma.array(u,mask=show_block)
            u[show_block.T] = np.nan
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
        if norm:
            max_u = np.nanmax(u)
            u = u #max_u

        CS = mp.contourf(x,y,u)


        if show_cbar:
            # print("\t", (np.amin(mag),np.amax(mag)))
            norm = mpl.colors.Normalize(vmin=np.nanmin(u),vmax=np.nanmax(u))
            cmap = plt.get_cmap("viridis")
            my_cm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
            try:
                global cbar
                cbar.update_normal(my_cm)
                # cbar.set_label(str(round(max_u,3)))
                # cbar.config_axis()
                # cbar.draw_all()
                # cbar.update_ticks()
            except Exception as e:
                print(e)
                cbar = fig.colorbar(my_cm,ax=mp)
                # cb1 = mpl.colorbar(mp, cmap=cmap,norm=norm)

    # if show block is wanted
    if not show_block is None:

        # x_block = X[show_block]
        # y_block = Y[show_block]
        # print(x_block)
        # (disc_xlb,disc_xub,disc_ylb,disc_yub)
        # (np.amin(x_block),np.amax(x_block),np.amin(y_block),np.amax(y_block))
        mp.imshow(~show_block.T, alpha=0.5, extent=(np.amin(x),np.amax(x),np.amin(y),np.amax(y)), origin='lower',
                interpolation='nearest', cmap='gray', aspect='auto')
        # mp.set_aspect('equal')


# --> Defining the hdf5 file reading variables
mb_num = 1
skip_num = 10
my_dpi = 100
my_fps = 30
my_file    = "./Output/testing4/run_" + str(mb_num) + ".h5"
video_name = "./Videos/testing4/run_" + str(mb_num) + ".mp4"

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
psi = hf["psi"]
x = hf["x"]
y = hf["y"]

# Choosing the starting index
i_start = len(t)//1.5
i_start = 0

# print(dP_x[5,1:-1,1])
# Plotting a specific x location of u Velocity at a ceratin time
# plt.plot(np.mean(u[1:-1,1:-1,-1].T,axis=1),y,'k--',linewidth=3)
# # plt.figure()
# # plt.plot(x,dP_x[1:-1,1:-1,-1])
# plt.figure()
# plt.plot(u[:,:,0].T,y)
# plt.show()
# exit()

fig = plt.figure()
my_plot1 = fig.add_subplot(3,1,1)#,projection='3d')
my_plot3 = fig.add_subplot(3,1,2)#,projection='3d')
my_plot4 = fig.add_subplot(3,1,3)#,projection='3d')

# max_P = np.amax(P[1:-1,1:-1,:])
# min_P = np.amin(P[1:-1,1:-1,:])
# my_block = np.isnan(P_i[1:-1,1:-1,2])
my_block = None
def animate(i):

    # Calculating the index
    i = i + (i) * skip_num + i_start
    if i >= len(t)-1:
        i = len(t)-1
    # if i <= 1:
    #     i = 1

    u_i = u[1:-1,1:-1,i]#/np.max(u[:,:,i])
    v_i = v[1:-1,1:-1,i]#/np.max(v[1:-1,1:-1,i])
    psi_i = psi[1:-1,1:-1,i]
    P_i = P[1:-1,1:-1,i]
    P_i[P_i == 0 ] = np.nan
    max_P = np.nanmax(P_i)
    min_P = np.nanmin(P_i)
    print(i,str(round(t[i],8)).ljust(12),max_P,min_P)
    if max_P != 0:
        P_i = P_i/101325#max_P
        # P_i[P_i==0] = np.nan
    else:
        P_i = P[1:-1,1:-1,i]

    my_plot1.clear()
    try:
        make_plot(fig, my_plot1, x[1:-1], y[1:-1],
              u_i.T,
              v_i.T,
              plot_type="streamline",
              sl_density=[5.0, 0.6],
              sub_type=["u","y"],
              show_0="x",
              show_block=my_block
              )
    except Exception as e:
        print(e)
        pass
    # M = np.hypot(u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T)
    # my_plot1.quiver(x, y, u[i,1:-1,1:-1].T,v[i,1:-1,1:-1].T, M, linewidth=0.1, edgecolor=(0,0,0),cmap="jet")
    my_plot1.set_title(str(round(t[i],6)))

    # --> PRESSURE FIELD
    my_plot3.clear()
    try:
        make_plot(fig, my_plot3, x[1:-1], y[1:-1],
              P_i.T,
              P_i.T,
              plot_type="contourf",
              sl_density=[0.9,1.0],
              sub_type=[],
              norm=True,
              show_cbar=True,
              show_block=np.isnan(P_i)
              )
    except Exception as e:
        print(e)
        exit()
        pass

    # --> Level Set Field
    my_plot4.clear()
    psi_i2 = np.copy(psi_i[:,:])
    psi_i2[(psi_i < 0)] = np.nan
    # psi_i2[(psi_i2 > 0)] = np.nan
    try:
        # my_plot4.contour(x[1:-1], y[1:-1],psi_i.T)
        make_plot(fig, my_plot4, x[1:-1], y[1:-1],
              psi_i2.T,
              psi_i2.T,
              plot_type="contourf",
              clear_plot=False,
              sl_density=[0.9,1.0],
              sub_type=[],
              norm=False,
              show_cbar=True,
              show_block=None ##np.isnan(P_i)
              )

    except Exception as e:
        print(e)
        exit()
        pass

num_frames = (len(t)//(skip_num+1))+1
ani = FuncAnimation(fig,animate,frames=num_frames)


if show_fig:
    plt.show()
if save_fig:
    ani.save(video_name,writer="ffmpeg", dpi=my_dpi, fps=my_fps)#,writer='matplotlib.animation.PillowWriter')#,writer=writer,dpi=100)
# exit()
