#
# ANDY JONES
# MAE 577 | HOMEWORK 4
# MARCH 23, 2020
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Determinining the number of droplets and bubbles
N_b = 1
N_d = 1

# Setting the domain size
L_x = 10
L_y = 5
x = np.linspace(0,L_x,1000)
h = x[1] - x[0]
y = np.arange(0,L_y,h)


# Setting the radius of the bubbles
R = L_y * 0.1

# Y loction of the liquid
y_w = L_y

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

def r_fun_b2(x,y,x_b,y_b):
    return ( (x - x_b)**2 + (y - y_b)**2)**0.5

X,Y = np.meshgrid(x,y)
psi_all = []
for i in range(len(x_b)):
    psi_all.append(r_fun_b2(X,Y,x_b[i],y_b[i]) - R )
psi_all = np.array(psi_all)
psi_d2 = np.min(psi_all,axis=0)
# psi_d2 = np.minimum(psi_d2,y_w-Y)
# print(phi_all.shape)
# print()
# exit()
# Going through the domain
psi = np.zeros((len(x),len(y)))
psi_d = np.zeros((len(x),len(y)))

# for i in range(len(x)):
#     for j in range(len(y)):
#         if y[j] < y_w:
#             if np.any(r_fun_b(x[i],y[j]) < R):
#                 psi[i,j] = -1
#             else:
#                 psi[i,j] = 1
#
#             dum_val = np.amin(r_fun_b(x[i],y[j])-R)
#             dum_val = np.amin([dum_val, y_w - y[j]] )
#             psi_d[i,j] = dum_val
#
#         else:
#             if np.any(r_fun_d(x[i],y[j]) < R):
#                 psi[i,j] = 1
#             else:
#                 psi[i,j] = -1
#
#             dum_val = np.amax(R - r_fun_d(x[i],y[j]))
#             dum_val = np.amax([dum_val, y_w - y[j]] )
#             psi_d[i,j] = dum_val

X,Y = np.meshgrid(x,y)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x,y,psi)
# ax.plot_surface(X,Y,psi)
# plt.imshow(psi.T,interpolation="bicubic")

# fig,ax = plt.subplots(1,1)
# plt.contour(x,y,psi.T,colors="black")#,colors=[(1,1,1),[0/255, 30/255, 222/255]])#cmap="summer")#colors=["white","blue"])
plt.contourf(x,y,psi_d2)#,colors=[(1,1,1),[0/255, 30/255, 222/255]])#cmap="summer")#colors=["white","blue"])
plt.show()
