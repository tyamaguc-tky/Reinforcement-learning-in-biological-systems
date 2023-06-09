#Two components change the numbers by stochastic increase and decrease.
#Simulation of the regulation with reinforcement learning (Fig 1b).
#For the deterministic regulation (Fig 1c), change the values of amp, add, 
import numpy as np
xAxB = np.array([1, 1])#initial values of x_a and x_b
target = np.array([1, 2])
target_ratio = target/np.sum(target)#Target ratio is [1/3, 2/3]
tmax = 10**5#the number of repetitions
data_all = np.zeros((2, tmax + 1), int)#for recording
xratio = xAxB/np.sum(xAxB)#initial value = [0.5, 0.5]
amp = 0.1#0 in the determinstic regulation
add = 0.001#0.1 in the deterministic regulation
dec_min = 10**(-4)#minimum value of decay probability
for t in range(tmax):#repeat for tmax times
    data_all[:, t] = xAxB#record
    if np.random.rand() < amp:#competitive amplification
        if np.sum(xAxB) > 0:#xa + xb > 0
            rc = np.random.choice((0, 1), p=(xAxB)/np.sum(xAxB))#current ratio
            xAxB[rc] += 1#the selected xa or xb increases by one
    if np.random.rand() < add:#additive increase
        if amp != 0:#regulation with reinforcement learning
            rc = np.random.choice((0, 1), p=(0.5, 0.5))# 1:1 ratio
        else:#determinstic regulation
            rc = np.random.choice((0, 1), p=target_ratio)#correct 1:2 ratio
        xAxB[rc] += 1
    if np.sum(xAxB) > 0:#decay
        xratio = xAxB/np.sum(xAxB)#current ratio
        if amp != 0:#regulation with reinforcement learning
            dec = np.max([np.sum((xratio - target_ratio)**2)/2/10, dec_min])#MSE/10
        else:#determinstic regulation
            dec = dec_min
        xAxB = np.random.binomial(xAxB, 1 - dec)#binomial distribution with the probability of 1-dec
data_all[:, -1] = xAxB#record the last numbers

import matplotlib.pyplot as plt#for presentation as figure
plt.plot(data_all[0,:])#xA
plt.plot(data_all[1,:])#xB
plt.show()