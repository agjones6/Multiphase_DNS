#
# ANDY JONES
# MAE 577 | HOMEWORK 4
# MARCH 23, 2020
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Determinining the number of droplets and bubbles
N_b = 5
N_d = 5

# Setting the domain size
L_x = 10
L_y = 5
x = np.linspace(0,L_x,1000)
h = x[1] - x[0]
y = np.arange(0,L_y,h)


# Setting the radius of the bubbles
R = L_y * 0.1

# Y loction of the liquid
y_w = 2.0

# Determinining the locations of the bubbles and droplets relative to the domain size
x_b = np.random.uniform(low=0,high=L_x,size=(N_b))
y_b = np.random.uniform(low=0,high=y_w,size=(N_b))
r_b = np.zeros(x_b.size) + R

x_d = np.random.uniform(low=0,high=L_x,size=(N_d))
y_d = np.random.uniform(low=y_w,high=L_y,size=(N_d))
r_d = np.zeros(x_d.size) + R

# Getting a radius as a funciton of a position in the domain
r_fun_d = lambda x,y: ( (x - x_d)**2 + (y - y_d)**2)**0.5
r_fun_b = lambda x,y: ( (x - x_b)**2 + (y - y_b)**2)**0.5

# Going through the domain
psi = np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        if y[j] < y_w:
            if np.any(r_fun_b(x[i],y[j]) < R):
                psi[i,j] = -1
            else:
                psi[i,j] = 1
        else:
            if np.any(r_fun_d(x[i],y[j]) < R):
                psi[i,j] = 1
            else:
                psi[i,j] = -1

X,Y = np.meshgrid(x,y)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x,y,psi)
# ax.plot_surface(X,Y,psi)
# plt.imshow(psi.T,interpolation="bicubic")

# fig,ax = plt.subplots(1,1)
plt.contourf(x,y,psi.T,colors=[(1,1,1),[0/255, 30/255, 222/255]])#cmap="summer")#colors=["white","blue"])
plt.show()
