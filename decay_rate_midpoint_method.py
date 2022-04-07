# -*- coding: utf-8 -*-
"""
Radioactive Decay - Midpoint Method

Radioactive decay occurs when unstable nuclei spontaneously breakdown into more
stable nuclei in a process that releases energy and matter from the nucleus.
The radioactive decay rate describes the average number of decays that occur
per unit time. The formula for the decay rate is as follows:

                                dN/dt = -λ * N

Where:
    
dN/dt - The first order differential of the number of nuclei with respect to time
λ - The decay constant
N - The number of nuclei

This programme will use the midpoint method of differentiation and matplotlib
to calculate and plot the decay rate equation. The expected values will also be
calculated and graphed as a comparison.
"""

# Import libraries
import numpy as np
from matplotlib import pyplot as plt

# Constants
N0 = 2            # Initial quantity
decay_const = 2   # Decay constant (lambda)
t_min = 0         # Start time
t_max = 10        # End time
steps = 100       # Number of steps
t_step = (t_max - t_min) / steps    # Step intervals for time
t_mp = t_step/2   # t midpoint value

# Predeclare lists for plotting
t_list = [] # x component (Time)
N_list = [] # y component (Calculated quantity)
N_ana = []  # y component (Expected quantity)

def slope(t, N):
    '''
    Calculate the slope for the radioactive decay equation
    
    ...
    
    Parameters
    ----------
    t : TYPE: Float
        DESCRIPTION: Time (Redundant)
    N : TYPE: Float
        DESCRIPTION: Quantity

    Returns
    -------
    TYPE: Float
    DESCRIPTION: Calculates and returns the slope of the 
                 graph at specified point
    '''
    return -decay_const * N

""" Midpoint Method: """
# Set initial N value
N = N0
# Loop from t_min to t_max at intervals of t_step
for t in np.arange(t_min, t_max + t_step, t_step):
    # Append current t value (x component) to t_list
    t_list.append(t)
    # Append current N value (y component) to N_list
    N_list.append(N)
    # Calculate and append expected N value to N_ana list
    N_ana.append(N0*np.exp(-decay_const * t))
    # Calculate N midpoint value
    N_mp = N + slope(t, N) * t_mp
    # Calculate new N1 value using the midpoint slope
    N1 = N + slope(t, N_mp) * t_step
    # Set new N value to N1
    N = N1

""" Plot """
# Set x-axis label
plt.xlabel('Time (t)')
# Set y-axis label
plt.ylabel('Count (N)')
# Plot midpoint method in red
plt.plot(t_list, N_list, 'r')
# Plot algebraic solution method in green
plt.plot(t_list, N_ana, 'g')
