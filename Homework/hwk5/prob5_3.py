#
# ANDY JONES
# MAE 577 | HOMEWORK 5
# MARCH 23, 2020
#
# Problem 3
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp as CP
from CoolProp.CoolProp import PropsSI

# Function for calculating level set
def get_psi(X,Y,x_b,y_b,r_b):
    # Defining a distance field for one bubble
    psi = r_b - ( (X.T - x_b)**2 + (Y.T - y_b)**2)**0.5
    return psi

# Bubble class I pulled from my main code
class bubble_class:
    def __init__(self):
        self.psi_b = []
        self.x = []
        self.y = []
        self.r = []
        self.N_b = 0
    def add_bubble(self,x,y,x_b,y_b,r_b):
        # Defining an initial distance field for one bubble
        X, Y = np.meshgrid(x,y)
        self.psi_b.append(get_psi(X,Y,x_b,y_b,r_b))
        self.x.append(x_b)
        self.y.append(y_b)
        self.r.append(r_b)
        self.N_b += 1

    def calc_all_psi(self):
        # This gets a continuous level set function for the entire domain
        self.psi = np.moveaxis(np.array(np.copy(self.psi_b[:])),0,-1)
        self.psi = np.amin(self.psi,axis=-1)

#--> Defining both fluid densities
# P_M = np.array([101325,101325,1013,1013]) # [Pa]
# Liquid
rho_l = 1000. #PropsSI("D","Q",0,"P",P_M,"Water")
# Gas
rho_g = 1.0 #PropsSI("D","Q",1,"P",P_M,"Water")

# Defining the domain
L_x = 0.01
L_y = 0.01 # This value doesn't matter
N_x = 60
x = np.linspace(0,L_x,N_x)
h = x[1]-x[0]
y = np.arange(0,L_y,h)

# Initializing the bubble class
bc = bubble_class()

# Adding a bubble to the bubble class
r = L_x/4
x_d = L_x/2
y_d = L_y/2
bc.add_bubble(x,y,x_d,y_d,r)
bc.calc_all_psi()

# Pulling out the 1 dimensional level set function
psi = np.copy(bc.psi[:,len(y)//2])

# Function for calculating the f(psi) using a variable M value
def get_f(M):
    # Creating an array of logical statements used in the piecewise function
    cond1 = psi < -M*h
    cond2 = psi > M * h
    cond3 = ~np.logical_xor(cond1,cond2)

    # Using the conditionals to calulate an f(psi)
    f = np.zeros(psi.shape)
    f[cond1] = 0
    f[cond2] = 1
    f[cond3] = (0.5) * (1 + psi[cond3]/(M*h) + (1/np.pi)*np.sin(np.pi*(psi[cond3]/(M*h))))

    return f

# Defining the M values
M1 = 1
M2 = 2
M3 = 3

# Getting a different f(psi) for each M
f1 = get_f(M1)
f2 = get_f(M2)
f3 = get_f(M3)

# --> ANSWER TO PART A
# Calculating a density with the continuous f(psi)
rho1 = rho_l*f1 + rho_g*f1
rho2 = rho_l*f2 + rho_g*f2
rho3 = rho_l*f3 + rho_g*f3

# --> PART B
m_drop_opt1 = []
m_drop_opt2 = []
m_drop_opt3 = []
x_droplet = x[(psi > 0)]
dx_droplet = x_droplet[-1] - x_droplet[0]

# Cycling through every M's density function
for rho in [rho1,rho2,rho3]:
    m_drop_opt1.append(np.sum(rho_l * h) * len(x[(psi > 0)]) )
    m_drop_opt2.append( np.sum(rho[(psi > 0)] * h ) )
    density_weight = (rho - rho_g) / (rho_l - rho_g)
    m_drop_opt3.append( np.sum( rho_l * density_weight * h ) )

# Calculating the analytic value for mass of the droplet
dx_analytic = 2*r
m_drop_analytic = dx_analytic * rho_l

# Printing the mass values
print(m_drop_opt1)
print(m_drop_opt2)
print(m_drop_opt3)
print(m_drop_analytic)

# Plotting
show_plots = False
if show_plots:
    fig = plt.figure()
    # Plotting M = 1
    ax1 = fig.add_subplot(3,1,1)
    ax1.plot(x,rho1,label="density",linewidth=2)
    ax1.plot([x_d+r, x_d+r],[np.amin(rho1),np.amax(rho1)],"--k",label="analytic interface")
    ax1.plot([x_d-r, x_d-r],[np.amin(rho1),np.amax(rho1)],"--k")
    ax1.set_title("M = 1")
    ax1.set_ylabel("rho (kg/m^3)")
    ax1.legend()

    # Plotting M = 2
    ax2 = fig.add_subplot(3,1,2)
    ax2.plot(x,rho2,label="density",linewidth=2)
    ax2.plot([x_d+r, x_d+r],[np.amin(rho2),np.amax(rho2)],"--k",label="analytic interface")
    ax2.plot([x_d-r, x_d-r],[np.amin(rho2),np.amax(rho2)],"--k")
    ax2.set_title("M = 2")
    ax2.set_ylabel("rho (kg/m^3)")
    ax2.legend()

    # Plotting M = 3
    ax3 = fig.add_subplot(3,1,3)
    ax3.plot(x,rho3,label="density",linewidth=2)
    ax3.plot([x_d+r, x_d+r],[np.amin(rho3),np.amax(rho3)],"--k",label="analytic interface")
    ax3.plot([x_d-r, x_d-r],[np.amin(rho3),np.amax(rho3)],"--k")
    ax3.set_title("M = 3")
    ax3.legend()
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("rho (kg/m^3)")
    plt.show()
