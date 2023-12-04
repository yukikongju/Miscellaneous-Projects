import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y, t):
    # Unpack variables
    xwt_sens, xwt_tol, xM_sens, xM_tol = y
    
    # Define the differential equations
    dydt = [
        (l_wt_sens * xwt_sens + a_wt_sens * xwt_sens * (xM_sens + xM_tol))
        * (1 - (xwt_sens + xwt_tol) / Kwt) - v_sens * xwt_sens,
        
        (l_wt_tol * xwt_tol + a_wt_tol * xwt_tol * (xM_sens + xM_tol))
        * (1 - (xwt_sens + xwt_tol) / Kwt) + vtol * xwt_sens,
        
        (l_M_sens * xM_sens + a_M_sens * xM_sens * (xwt_sens + xwt_tol))
        * (1 - (xM_sens + xM_tol) / Km) - v_sens * xM_sens,
        
        (l_M_tol * xM_tol + a_M_tol * xM_tol * (xwt_sens + xwt_tol))
        * (1 - (xM_sens + xM_tol) / Km) + vtol * xM_sens
    ]
    
    return dydt


# Set parameter values
l_wt_sens, l_wt_tol, l_M_sens, l_M_tol = -3.6519, 0.07, -0.3975, 1.0494
a_wt_sens, a_wt_tol, a_M_sens, a_M_tol = 10, 2.0331, 0.131, -0.3031
Kwt, Km = 10, 10
v_sens, vtol = 5.0103, 0.0142

# Set initial conditions
initial_xwt_sens, initial_xwt_tol, initial_xM_sens, initial_xM_tol = 1, 1, 1, 1
y0 = [initial_xwt_sens, initial_xwt_tol, initial_xM_sens, initial_xM_tol]

# Set time points
t_start, t_end, num_points = 0, 6, 1000
t = np.linspace(t_start, t_end, num_points)

# Solve the system of differential equations
solution = odeint(model, y0, t)

# Extract results
xwt_sens_solution = solution[:, 0]
xwt_tol_solution = solution[:, 1]
xM_sens_solution = solution[:, 2]
xM_tol_solution = solution[:, 3]

# Plot the results
plt.plot(t, xwt_sens_solution, label='xwt_sens')
plt.plot(t, xwt_tol_solution, label='xwt_tol')
plt.plot(t, xM_sens_solution, label='xM_sens')
plt.plot(t, xM_tol_solution, label='xM_tol')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()
