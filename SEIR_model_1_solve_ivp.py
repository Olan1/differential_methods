# -*- coding: utf-8 -*-
"""
SEIR Model applied to COVID-19:

The SEIR model is used to model the spread of infectious disease. It is
described by:

                        N = S + E + I + R

Where:
N - Total population
S - Susceptable population
E - Exposed population
I - Infected population
R - Recovered population

This program will use solve_ivp to calculate the rate of change for each of
these variables. These 1st order derivatives are:
    
                dS/dt = rb.N - rd.S - beta.I.S/N
                
                dE/dt = beta.I.S/N - (rd - alpha).E
                
                dI/dt = alpha.E - (gamma + rd).I
                
                dR/dt = gamma.I - rd.R
    
Where:
dS/dt - 1st order derivative of S (Susceptable population)
rb - Birth rate
rd - Death rate
beta - Number of close contacts per person per day
dE/dt - - 1st order derivative of E (Exposed population)
alpha - Rate exposed people become infectious per day
dI/dt - 1st order derivative of I (Infected population)
gamma - Rate infected people recover per day
dR/dt - 1st order derivative of R (Recovered population)

"""

# Import libraries/modules
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

# Constants
N = 6556300.0         # Population of island of Ireland - North + Republic
rb = 0.008535/365     # Birth rate per day
rd = 0.004845/365     # Death rate per day

alpha = 0.0115        # Rate exposed people become infectious per day
gamma = 0.05          # Rate infected people recover per day
# Data source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7376536/

beta = 2.0      # Number of close contacts per person per day (post lockdown)
# Data source: https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-Contract-tracing-scale-up.pdf

S = N       # Number of susceptable people (Initially = total population)
E = 5077    # Number exposed to disease
I = 736     # Number currently infected
R = 32      # Number recovered
# Data source: https://link.springer.com/article/10.1007/s11071-020-05743-y

ics = [S, E, I, R]  # Place initial conditions in a list

ti = 0      # Start day
tf = 365    # End day
# Create list of time variables from ti to tf with 10,000 increments
t = np.linspace(ti, tf, 10000)

def slope(t, ics, N, rb, rd, alpha, beta, gamma):
    """
    Parameters
    ----------
    t : TYPE: Float
        DESCRIPTION: Current time variable
    ics : TYPE: List
        DESCRIPTION: Contains initial conditions
    N : TYPE: Float
        DESCRIPTION: Population
    rb : TYPE: Float
        DESCRIPTION: Daily birth rate
    rd : TYPE: Float
        DESCRIPTION: Daily death rate
    alpha : TYPE: Float
        DESCRIPTION: Rate exposed people become infectious per day
    beta : TYPE: Float
        DESCRIPTION: Number of close contacts per person per day
    gamma : TYPE: Float
        DESCRIPTION: Rate infected people recover per day

    Returns
    -------
    list
        DESCRIPTION: First order derivatives of S, E, I, and R

    """
    # Unpack ics list
    S, E, I, R = ics
    # Calculate rate of change of susceptable population
    dSdt = rb*N - rd*S - (beta*I*S)/N
    # Calculate rate of change of exposed population
    dEdt = (beta*I*S)/N - (rd + alpha)*E
    # Calculate rate of change of infected population
    dIdt = alpha*E - (gamma+rd)*I
    # Calculate rate of change of recovered population
    dRdt = gamma*I - rd*R
    # Return each derivative
    return [dSdt, dEdt, dIdt, dRdt]

# Differentiate using slope function, from ti to tf, with constants N, rb, rd, alpha,
# beta and gamma, specify t points, set integration method to LSODA (ODEINT method)
sol = solve_ivp(slope, [ti, tf], ics,args=(N, rb, rd, alpha, beta, gamma),
                t_eval=t, method='LSODA')

# Calculate and print reproduction number
R0 = (alpha * beta)/((rd+alpha)*(rd+gamma))
print(R0)

# Plot pyplot graph:
plt.xlabel('Days')      # Set x-axis label
plt.ylabel('People')    # Set y-axis label
plt.title('SEIR Model for COVID-19')    # Set graph title
plt.plot(sol.t, sol.y[0], 'r', label='Susceptible') # Plot S vs t
plt.plot(sol.t, sol.y[1], 'c', label='Exposed')     # Plot E vs t
plt.plot(sol.t, sol.y[2], 'g', label='Infected')    # Plot I vs t
plt.plot(sol.t, sol.y[3], 'b', label='Recovered')   # Plot R vs t
plt.legend(loc='best')  # Display legend
