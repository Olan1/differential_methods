# -*- coding: utf-8 -*-
"""
Radioactive Decay

Radioactive decay occurs when unstable nuclei spontaneously breakdown into more
stable nuclei in a process that releases energy and matter from the nucleus.
The radioactive decay rate describes the average number of decays that occur
per unit time. The formula for the decay rate is as follows:

                                dN/dt = -λ * N

Where:
    
dN/dt - The first order differential of the number of nuclei with respect to time
λ - The decay constant
N - The number of nuclei

This programme will use odeint to differentiate the equation, and matplotlib to
plot the decay rate.
"""

# Import libraries/modules
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np

# Variables:
lmbda = 0.5                # Decay constant (lambda)
N0 = 1000                  # Initial number of atoms

ti = 0                     # Start time
tf = 100                   # End time
t = np.arange(ti, tf, 1)   # List of time values from ti to tf in steps of 1

# Slope function
def slope(N, t, lmbda):
    '''
    Parameters
    ----------
    N : TYPE - Integar
        DESCRIPTION - Number of undecayed nuclei
    t : TYPE - Integar
        DESCRIPTION - Time variable
    lmbda : TYPE - Float
        DESCRIPTION - Decay constant

    Returns
    -------
    TYPE - Float
        DESCRIPTION - The rate of decay
    '''
    # Calculate and return the slope
    return -lmbda * N

# Differentiate using odeint
y = odeint(slope, N0, t, args=(lmbda,))

# Pyplot graph:
# Set x-axis label
plt.xlabel('Time')
# Set y-axis label
plt.ylabel('N')
# Set graph title
plt.title('Radioactive Decay')
# Plot N vs t
plt.plot(t, y)
# Display legend
plt.legend(loc='best')
