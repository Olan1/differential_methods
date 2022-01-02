"""
Computational Physics - Damped Harmonic Oscillator

This program will use odeint to calculate the damped harmonic oscillator.
The damping effect will be adjusted via the damping ratio (R). The 
greater R, the more damped the oscillator will become.
i.e:
R = 0 - Not damped
R < 1 - Underdamped
R = 1 - Critically damped
R > 1 - Overdamped
The oscillations will be graphed and displayed using matplotlib.
The equation used to calculate the oscillations is displayed below:
                d^2y/dt^2 = -2.R.w0.dy/dt - w0^2.y
Where:
d^2y/dt^2 - 2nd order derivative of y (position) with respect to t (time)
R - Damping ratio
w0 - Angular frequency
dy/dt - First order derivative of y (position) with respect to t (time)
y - Position
"""

# Import libraries/modules
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

# Constants
w0 = 1      # Initial angular frequency
y0 = 0      # Initial displacement
v0 = 1      # Initial dy/dt
ics = [y0, v0]  # Place initial conditions in a list
R_start = 0     # Initial damping ratio
R_end = 1.5     # Final damping ratio
R_step = 0.5   # Damping ratio step

# Create list of R values(damping ratios) from R_start - R_end in steps of R_step
R_list = np.arange(R_start, R_end + R_step, R_step)

# Construct a list of time variables from 0 to 25 in steps of 0.1 seconds
t = np.arange(0, 25, 0.1)

def slope(ics, t, R, w0):
    '''
    Parameters
    ----------
    ics : TYPE: List
        DESCRIPTION: List of initial conditions
    t : TYPE: Float
        DESCRIPTION: Time variable
    R : TYPE: Float
        DESCRIPTION: Damping ratio
    w0 : TYPE: Float
        DESCRIPTION: Angular frequency
    Returns
    -------
    TYPE: List
    DESCRIPTION: List[0] = dy/dt (First order derivative of y)
                 List[1] = d^2y/dt^2 (Second order derivative of y)
    '''
    # d^2y/dt^2 = -2 * R * w0 * dy/dt - w0^2 * y
    # d/dt dy/dt = -2 * R * w0 * dy/dt - w0^2 * y
    # dy/dt = v
    # d^2y/dt^2 = dv/dt = -2 * R * w0 * v - w0^2 * y
    y = ics[0]    # = y
    v = ics[1]    # = y'
    dv_dt = -2 * R * w0 * v - w0**2 * y  # Calculate derivative of v (v' = y'')
    # Return [1st y derivative (y', or v), 2nd y derivative (y'', or v')]
    return [v, dv_dt]

def plot_graph(R, Y_list, t):
    """
    Parameters
    ----------
    R : TYPE: Float
        DESCRIPTION: Damping ratio
    Y_list : TYPE: List
        DESCRIPTION: List of y values to be plotted
    t : TYPE: List
        DESCRIPTION: List of t values to be plotted
    Returns
    -------
    None.
    
    Plot a graph of Y_list vs t with the graph label determined by R value
    """
    plt.xlabel('Time (s)')  # Set x-axis label
    plt.ylabel('y (m)')     # Set y-axis label
    plt.title('Damped Harmonic Oscillator')   # Set graph title
    
    # Plot graph and determine label based on damping ratio (R)
    # If R = 0:
    if R == 0:
        # Plot graph with label displaying R value and Not Damped
        plt.plot(t, Y_list, label=f'R = {R} - Not Damped')
    # Else if 0 < R < 1:
    elif R > 0 and R < 1:
        # Plot graph with label displaying R value and Underdamped
        plt.plot(t, Y_list, label=f'R = {R} - Underdamped')
    # Else if R = 1:
    elif R == 1:
        # Plot graph with label displaying R value and Critically Damped
        plt.plot(t, Y_list, label=f'R = {R} - Critically Damped')
    # Else if R > 1:
    elif R > 1:
        # Plot graph with label displaying R value and Overdamped
        plt.plot(t, Y_list, label=f'R = {R} - Overdamped')
        
    # Display the graph legend in the upper right corner. Font size set small
    plt.legend(loc='upper right', prop={'size': 'small'})
    
# Iterate through R values in R_list
for R in R_list:
    # Differentiate using slope function, initial conditions, range = t,
    # and constants R (Damping ratio) and w0 (Angular frequency)
    Y = odeint(slope, ics, t, args=(R, w0))
    # Create a list of all rows, 2nd column from the Y 2d array
    Y_list = Y[:, 1]
    
    # Pyplot graph:
    plot_graph(R, Y_list, t)
