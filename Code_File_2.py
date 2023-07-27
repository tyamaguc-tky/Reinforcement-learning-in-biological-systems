#Code for Figure 2; predator and prey simulation
import numpy as np
xAxB = np.array([10, 10])#initial numbers of prey xa and predator xb 
target = np.array([2, 1])
target_ratio = target/np.sum(target)#Target ratio is [2/3, 1/3]
tmax = 10**5#number of repetitions. 10**6 in SummaryFig.
xratio = xAxB/np.sum(xAxB)#initial ratio is 0.5:0.5
dec_min = 1E-4#minimal value of decay probability, 10**(-4)

def pp_model(pair = xAxB, km=100, vmax=1, hunt=0):
    a = 0.1#proliferation
    b = 0.01#migration
    b0 = target_ratio[0]#0.667
    dec = dec_min#decay probability E
    dec_h0 = 1.5E-4#decay of xa due to hunting by human
    dec_h1 = hunt#set to 0 or 1.5E-4 or 1E-4
    if np.random.rand() < b:#additive increase
       rc = np.random.choice((0, 1), p=(b0, 1 - b0))#p=[2/3, 1/3]
       pair[rc] += 1#increases by 1
    if np.random.rand() < a:#competitive amplification of prey
        if np.random.rand() < pair[0]/(pair[0] + pair[1] + 1E-7):
            pair[0] += 1#prey increase by one
    if np.random.rand() < a:#proliferation of predator in Michaelis-Menten eq.
        if np.random.rand() < pair[1]/(pair[1] + km) * vmax:
            pair[1] += 1#predator increase by one
    
    if np.sum(pair) > 0:#decay
        xratio = pair/np.sum(pair)#current ratio
        if xratio[0] - target_ratio[0] < 0:#if prey is relatively few
            dec = max(sum((xratio - target_ratio)**2)/2/100, dec_min)#MSE/100
        else:
            dec = dec_min
        if hunt == 0:# without hunting
            pair = np.random.binomial(pair, 1 - dec)#binomial distribution
        else:#for Fig 2F,G, and SummaryFig
            pair[0] = np.random.binomial(pair[0], 1 - dec - dec_h0)
            pair[1] = np.random.binomial(pair[1], 1 - dec - dec_h1)
    return pair

def test100(km100=100, vmax100=1, tmax=10**5):#for Fig 2B
    result_array = np.zeros((2, 100), dtype=int)
    for k in range(100):
        xAxB = np.array([10, 10])#initial numbers
        for t in range(tmax):#repeat for tmax times
            xAxB = pp_model(pair=xAxB, km=km100, vmax=vmax100, hunt=0)#change to km=xAxB[0] to set km=xa
        result_array[:, k] = xAxB#record the number at tmax
    return result_array

data_all = np.zeros((2, tmax + 1),int)#for recording
for t in range(tmax):#repeat for tmax times
    data_all[:, t] = xAxB#recording the number at each time point
    xAxB = pp_model(pair=xAxB, vmax=1,km=xAxB[0], hunt=0)#parameters should be changed to vmax=1.3,km=200, or vmax=0.8,km=400, or vmax=1,km=xAxB[0]
    #xAxB = pp_model(pair=xAxB, vmax=1, km=xAxB[0], hunt=1E-4)#parameters should be changed to hunt=0, 1E-4,or 1.5E-4
data_all[:, -1] = xAxB#record the number at tmax
"""#use the followings for Fig 2B
data = np.zeros((2, 100, 15),dtype=int) 
for v in range(15):
    data[:, :, v] = test100(km100=200, vmax100=(1+v)/10, tmax=10**4)#change the parameter km100
medians = np.median(np.squeeze(data[1,:,:]),axis=0).T
print(medians)
"""
import matplotlib.pyplot as plt#for visualization
plt.plot(data_all[0,:])#the number of prey
plt.plot(data_all[1,:])#the number of predator
plt.show()