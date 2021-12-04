# -*- coding: utf-8 -*-
"""
SEIR Model applied to COVID-19:

The SEIR model is used to model the spread of infectious disease. It is
described by:

                N = S + E + Ip + Ia + Ii + It1 + It2 + In + R

Where:
N - Total population
S - Susceptable population
E - Exposed population
Ip -  Pre-symptomatic infected population
Ia -  Asymptomatic infected population
Ii - Symptomatic and self-isolating (without testing) population
It1 - Symptomatic and waiting for testing population
It2 - Post-test self-isolation population
In - Symptomatic and not isolating population
R - Recovered population

This program will use solve_ivp to calculate the rate of change for each of
these variables. These 1st order derivatives are:
    
                dS/dt = −βS(Ip + hIa + iIi + It1 + jIt2 + In) /N
                
                dE/dt = βS(Ip + hIa + iIi + It1 + jIt2 + In) /N − (1/L).E
                
                dIp/dt = ((1 − f)/L).E - (1/(C - L)).Ip
                
                dIa/dt = (f/L).E - (1/D).Ip
                
                dIi/dt = (q/(C - L)).Ip - (1/(D - C + L)).Ii
                
                dIt1/dt = (τ/(C - L)).Ip - (1/T).It1
                
                dIt2/dt = (1/T).It1 - (1/(D - C + L - T)).It2
                
                dIn/dt = ((1 - q - τ)/(C - L)).Ip - (1/(D - C + L)).In
                
                dR/dt = (1/D).Ia + (1/(D - C + L)).Ii + (1/(D - C + L - T)).It2 + (1/(D - C + L)).In
    
Where:
dS/dt - 1st order derivative of S
β - Transmission rate
h - multiplicative factor for reduction of effective transmission from the
Asymptomatic Infected compartment, relative to Symptomatic Infected (assumed 0.01-0.5)
i -  multiplicative factor for reduction of effective transmission from the
Immediate Isolation compartment relative to Symptomatic Infected (assumed 0-0.1)
j - multiplicative factor for reduction of effective transmission from the
Post-test isolation compartment, relative to Symptomatic Infected (Assumed 0-0.1)
dE/dt - - 1st order derivative of E
L - Average latent period (Assumed 3.9 - 5.9)
dIp/dt - 1st order derivative of Ip
f - Fraction of infected who are Asymptomatic (range assumed is 0.18 to 0.82)
C - Average incubation period (Assumed L - 6.8)
dIa/dt - 1st order derivative of Ia
D - average infectious period (Assumed C-L to 9.0)
dIi/dt - 1st order derivative of Ii
q - Fraction of symptomatic cases that self-quarantine (assumed  0 to 1 − τ)
dIt1/dt - 1st order derivative of It1
τ - Fraction of symptomatic cases that are tested (Assumed 0.5 - 1.0)
T -  Average wait for test results (assumed 1 to 5)
dIt2/dt - 1st order derivative of It2
dIn/dt - 1st order derivative of In
dR/dt - 1st order derivative of R
"""

# Import libraries/modules
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

# Constants
N = 6556300.0         # Population of island of Ireland - North + Republic
# rb = 0.008535/365     # Birth rate per day
# rd = 0.004845/365     # Death rate per day

# alpha = 0.0115        # Rate exposed people become infectious per day
# gamma = 0.05          # Rate infected people recover per day
# Data source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7376536/

# beta = 2.0      # Number of close contacts per person per day (post lockdown)
# Data source: https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-Contract-tracing-scale-up.pdf

β = 2
L = (5.9 + 3.9)/2
f = (0.82 + 0.18)/2
C = (6.8 + L)/2
D = (9 + (C-L))/2
τ = (1 + 0.5)/2
q = (1 - τ)/2
T = (1 + 5)/2
h= (0.01 + 0.5)/2
i = 0.1/2
j = 0.1/2

S = N       # Number of susceptable people (Initially = total population)
E = 5077    # Number exposed to disease
Ip = 200      # Pre-symptomatic infected population
Ia = 200      # Asymptomatic infected population
Ii = 736      # Symptomatic and self-isolating (without testing) population
It1 = 736     # Symptomatic and waiting for testing population
It2 = 736     # Post-test self-isolation population
In = 200      # Symptomatic and not isolating population
R = 32      # Number recovered

ics = [S, E, Ip, Ia, Ii, It1, It2, In, R]  # Place initial conditions in a list

ti = 0      # Start day
tf = 365    # End day
# Create list of time variables from ti to tf with 10,000 increments
t = np.linspace(ti, tf, 10000)

def slope(t, ics, N, β, L, f, C, D, τ, q, T, h, i, j):
    """
    Parameters
    ----------
    t : TYPE: Float
        DESCRIPTION: Time variable
    ics : TYPE: List
        DESCRIPTION: Initial conditions
    N : TYPE: Float
        DESCRIPTION: Population
    β : TYPE: Float
        DESCRIPTION:Transmission rate
    L : TYPE: Float
        DESCRIPTION: Average latent period
    f : TYPE: Float
        DESCRIPTION: Fraction of infected who are Asymptomatic
    C : TYPE: Float
        DESCRIPTION: Average incubation period
    D : TYPE: Float
        DESCRIPTION: Average infectious period
    τ : TYPE: Float
        DESCRIPTION: Fraction of symptomatic cases that are tested
    q : TYPE: Float
        DESCRIPTION: Fraction of symptomatic cases that self-quarantine
    T : TYPE: Float
        DESCRIPTION: Average wait for test results
    h : TYPE: Float
        DESCRIPTION: Multiplicative factor for reduction of effective transmission
                    from the Asymptomatic Infected compartment
    i : TYPE: Float
        DESCRIPTION: Multiplicative factor for reduction of effective transmission
                    from the Immediate Isolation compartment
    j : TYPE: Float
        DESCRIPTION: multiplicative factor for reduction of effective transmission
                    from the Post-test isolation compartment

    Returns
    -------
    List.
    """
    # Unpack ics list
    S, E, Ip, Ia, Ii, It1, It2, In, R = ics
    # Calculate rate of change of susceptable population
    dSdt = -β*S*(Ip + h*Ia + i*Ii + It1 + j*It2 + In)/N
    # Calculate rate of change of exposed population
    dEdt = β*S*(Ip + h*Ia + i*Ii + It1 + j*It2 + In)/N - (1/L)*E
    # Calculate rate of change of infected population
    dIpdt = ((1 - f)/L)*E - (1/(C - L))*Ip
    # Calculate rate of change of 
    dIadt = (f/L)*E - (1/D)*Ia
    # Calculate rate of change of 
    dIidt = (q/(C - L))*Ip - (1/(D - C + L))*Ii
    # Calculate rate of change of 
    dIt1dt = (τ/(C - L))*Ip - (1/T)*It1
    # Calculate rate of change of 
    dIt2dt = (1/T)*It1 - (1/(D - C + L - T))*It2
    # Calculate rate of change of 
    dIndt = ((1 - q - τ)/(C - L))*Ip - (1/(D - C + L))*In
    # Calculate rate of change of recovered population
    dRdt = (1/D)*Ia + (1/(D - C + L))*Ii + (1/(D - C + L - T))*It2 + (1/(D - C + L))*In
    # Return each derivative
    return [dSdt, dEdt, dIpdt, dIadt, dIidt, dIt1dt, dIt2dt, dIndt, dRdt]

# Differentiate using slope function, from ti to tf, with constants N, rb, rd, alpha,
# beta and gamma, specify t points, set integration method to LSODA (ODEINT method)
sol = solve_ivp(slope, [ti, tf], ics,args=(N, β, L, f, C, D, τ, q, T, h, i, j),
                t_eval=t, method='LSODA')

# Plot pyplot graph:
plt.xlabel('Days')      # Set x-axis label
plt.ylabel('People')    # Set y-axis label
plt.title('SEIR Model for COVID-19')    # Set graph title
plt.plot(sol.t, sol.y[0], label='S')             # Plot S vs t
plt.plot(sol.t, sol.y[1], label='E')             # Plot E vs t
plt.plot(sol.t, sol.y[2], label='Ip')            # Plot Ip vs t
plt.plot(sol.t, sol.y[3], label='Ia')            # Plot Ia vs t
plt.plot(sol.t, sol.y[4], label='Ii')            # Plot Ii vs t
plt.plot(sol.t, sol.y[5], label='It1')           # Plot It1 vs t
plt.plot(sol.t, sol.y[6], label='It2')           # Plot It2 vs t
plt.plot(sol.t, sol.y[7], label='In')            # Plot In vs t
plt.plot(sol.t, sol.y[8], label='R')     # Plot R vs t
plt.legend(loc='best', prop={'size': 'small'})  # Display legend
