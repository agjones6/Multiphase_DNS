#
# ANDY JONES
# MAE 577 | HOMEWORK 3
#

# importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Defining the known analytic equation
u_max = 0.03    # m/s
y_min = 0.005   # m
rho   = 10**3   # kg/m
nu    = 10**-6  # m^2/s
dP    = -u_max * 2 * rho * nu / y_min**2
print(dP)
# exit()
u_fun = lambda y: (-u_max/y_min**2)*y**2 + u_max

# Defining a y distribution and calculating the velocity at each y point
y_dist = np.linspace(-y_min,y_min,1000)
u_vals = u_fun(y_dist)

plt.plot(u_vals,y_dist)
plt.show()
