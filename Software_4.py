import numpy as np
import matplotlib.pyplot as plt
#Code for visualization of Figure 3 are written in coment mode
#Using Code_file_3, simulation results are obtained.
"""
for test in range(test_n):
    result[test] = random_walk_1D(init_pos=10, tmax=100)
traj = random_walk_2D(sl_p=0.1,init_pos=10,exp_n=1,tmax=10**7,border=0)
traj_b = random_walk_2D(sl_p=0.1,init_pos=10,exp_n=1,tmax=10**6,border=500)
#np.savez_compressed('data0', traj_s=traj,traj_b_s=traj_b, distance_data=result)
"""
loaded = np.load('data0.npz')
result = loaded['distance_data']
traj = loaded['traj_s']#[:,t0:t1+1]
traj_b = loaded['traj_b_s']#[:,t0:t1+1]

#To make histogram in Figure 3B-D
result10 = np.log10(result)
(hist, bins, _) = plt.hist(result10, bins=1000,range=(-2,3))#range=(-2,3) or (-2,4)
plt.show()
print([np.sum(result < 10**-2), np.sum(result > 10**3)])#np.sum(result>10**4)])

#To make fitting curve in Fig 3F
"""
# To estimate parameter values of fitting curves, curve fitting package from scipy was used
#import curve fitting package from scipy
from scipy.optimize import curve_fit
pars2, cov2 = curve_fit(f=truncated_power, xdata=sort_dist[:x_len], ydata=x,p0=[1e-14, 0.05, 10**15, 2*10**5] )
a, b, c, d = pars2
"""
def truncated_power(x, a=10**-16,b=0.05,c=10**14,n=10**6):
    d = n * a**b#158489
    return d * (x + a)**(-b) * np.exp(-x/c)
#The following can be considered for better fitting.
def truncated_power2(x, a=10**-1,b=0.08,c=10**14,aa= 10**-28,bb=0.015,n=10**6):
    d = n * a**b * aa**bb
    return d * (x + a)**(-b) * (x + aa)**(-bb)* np.exp(-x/c)

step_length = np.ones(10**6 - 1)#without boundary#np.ones(len(traj[0,:]) - 1)
for t in range(len(step_length)):
    step_length[t] = ((traj[0,t] - traj[0,t+1])**2 + (traj[1,t] - traj[1,t+1])**2)**(1/2)
sort_sl = np.sort(step_length)[::-1]
x_len = np.sum(sort_sl > 0)
x = np.arange(1, x_len + 1)#range(len(distance))

step_length_b = np.ones(10**6 - 1)#with boundary#np.ones(len(traj_b[0,:])-1)
for t in range(len(step_length_b)):
    step_length_b[t] = ((traj_b[0,t] - traj_b[0,t+1])**2 + (traj_b[1,t] - traj_b[1,t+1])**2)**(1/2)
sort_sl_b = np.sort(step_length_b)[::-1]
x_len_b = np.sum(sort_sl_b > 0)
x_b = np.arange(1, x_len_b + 1)#range(len(distance))

b = 0.05
a = 10**(-16)

c1 = 10**14#for trajectory without boundary
fit_curve_red = truncated_power(sort_sl[:x_len],a=a,b=b,c=c1,n=x_len)
c2 = 10#for trajectory with boundary
fit_curve_green = truncated_power(sort_sl_b[:x_len_b],a=a,b=b,c=c2,n=x_len_b)

plt.plot(sort_sl_b[:x_len_b], x_b,'ob')
plt.plot(sort_sl[:x_len], x,'ok')
plt.plot(sort_sl_b[:x_len_b], fit_curve_green,'--y',linewidth='5')
plt.plot(sort_sl[:x_len], fit_curve_red,'-r',linewidth='3')
plt.ylim(10**4, 2*10**6)
plt.xscale('log')
plt.yscale('log')
plt.show()


#trajectory Figure
#To make Figure 3E, G # for Fig 3G blocks should be activated
#block1=np.array([[-200, -300], [300,-300],[300,300],[-300,300],[-300,-300],[-250,-300]]).T
#block2=np.array([[0, 100], [-100,100],[-100,-100],[100,-100],[100,100],[50,100]]).T
#block3=np.array([[-10, -50], [50,-50],[50,50],[-50,50],[-50,-50],[-30,-50]]).T
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
plt.rcParams["font.size"] = 18
t0=0#13000#change to the indicated starting time point
t1 =t0 + 10000#3000#tmaxchange to the indicated end time point#1000 for Summary Fig
plt.plot(traj[0,t0:t1], traj[1,t0:t1],linewidth=0.5)
plt.plot(traj[0,t0], traj[1,t0], 'kx')#'gs')
plt.plot(traj[0,t1-1], traj[1,t1-1], 'ro')
#plt.plot(traj1[0,t0:t1], traj1[1,t0:t1],linewidth=0.5)
#plt.plot(traj1[0,t1-1], traj1[1,t1-1], 'ro')
#plt.plot(traj2[0,t0:t1], traj2[1,t0:t1],linewidth=0.5)
#plt.plot(traj2[0,t1-1], traj2[1,t1-1], 'ro')
#plt.plot(block1[0,:],block1[1,:], '-k',linewidth=3)
#plt.plot(block2[0,:],block2[1,:], '-k',linewidth=3)
#plt.plot(block3[0,:],block3[1,:], '-k',linewidth=3)
plt.xlim(-500, 500)#can be changed or removed
plt.ylim(-500, 500)#can be changed or removed
plt.show()

#Animation for Movie 1-2
import matplotlib.animation as animation
block1=np.array([[-200, -300], [300,-300],[300,300],[-300,300],[-300,-300],[-250,-300]]).T
block2=np.array([[0, 100], [-100,100],[-100,-100],[100,-100],[100,100],[50,100]]).T
block3=np.array([[-10, -50], [50,-50],[50,50],[-50,50],[-50,-50],[-30,-50]]).T
graphs = []
fig = plt.figure()
fig.set_figwidth(4)
fig.set_figheight(4)
plt.xlim(-500, 500)
plt.ylim(-500, 500)
plt.plot(block1[0,:],block1[1,:], '-k',linewidth=3)
plt.plot(block2[0,:],block2[1,:], '-k',linewidth=3)
plt.plot(block3[0,:],block3[1,:], '-k',linewidth=3)
g = plt.plot(traj[0,0], traj[1,0], 'ro')
graphs.append(g)#(g)
for t in range(40001):#5001)#len(traj[0,:])):
    if t%10 == 0:
        g0 = plt.plot(traj[0,max(t-10,0):t+1], traj[1,max(t-10,0):t+1],'b--')
        g = plt.plot(traj[0,t], traj[1,t], 'ro')
        graphs.append(g0 + g)
anim = animation.ArtistAnimation(fig, graphs, interval=20, repeat=False)
plt.show()
#anim.save('movie0.gif', writer='pillow')
#gif files are coverted to MP4 in Adobe web page