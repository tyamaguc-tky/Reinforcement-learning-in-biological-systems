#Code for random walk controlling step-length.
#Simulations in Figure 3. Use also Code_File_4 for visualization. 
import numpy as np
#For 1-dimensional s-rw in Fig 3B
#Change init_pos and tmax 
def random_walk_1D(init_pos=10, tmax=100):
    sl_p = 0.1
    target = 0
    rand_d = 1 - np.random.rand(tmax) * 2 
    xt = init_pos
    traj = np.ones(tmax, dtype=float)
    traj[0] = xt
    for t in range(tmax - 1):
        if np.abs(xt) < 10**20:
            step_length = np.abs(xt - target) * sl_p
            xt += step_length * rand_d[t]            
        else:
            break
        traj[t + 1] = xt
    return xt#traj

#for setting boundary or obstacles
#To determine the next postion
def inside_out(x0, x1, y0, y1, block=50, gap=-20):
    xc = x1#next position
    yc = y1
    if np.abs(x0) < block:
        if np.abs(y0) < block:#current position is inside the block
            if np.abs(x1) >= block:#crossing the block
                    xc = np.sign(x1) * (block - 1)#stay near the wall
                    yc = y0 + (y1 - y0) * (xc - x0) / (x1 - x0)#intesection point
                    if np.abs(yc) >= block:#correct the intersction point
                        yc = np.sign(y1) * (block - 1)
                        xc = x0 + (x1 - x0) * (yc - y0) / (y1 - y0)
                        if np.abs(xc) >= block:
                            xc = np.sign(x1) * (block - 1)
                        if np.sign(yc) == np.sign(gap):#Check whether the point is in the gap
                            if np.sign(xc) == np.sign(gap):
                                if np.abs(xc) > block - 2 * np.abs(gap):
                                    if np.abs(xc) < block - np.abs(gap):
                                        yc += 2*np.sign(gap)#Cross over the wall       
            elif np.abs(y1) >= block:#crossing the block
                yc = np.sign(y1) * (block - 1)
                xc = x0 + (x1 - x0) * (yc - y0) / (y1 - y0)
                if np.abs(xc) >= block:
                    xc = np.sign(x1) * (block - 1)
                if np.sign(yc) == np.sign(gap):
                        if np.sign(xc) == np.sign(gap):
                            if np.abs(xc) > block - 2*np.abs(gap):
                                if np.abs(xc) < block - np.abs(gap):
                                    yc += 2*np.sign(gap)
    return [xc, yc]#output the next postition

def outside_in(x0, x1, y0, y1, block=50, gap=-20):
    xc = x1
    yc = y1
    if np.abs(x0) > block:
        if (np.abs(x1) <= block) or (np.sign(x0) != np.sign(x1)):
            xc = np.sign(x0) * (block + 1)
            yc = y0 + (y1 - y0) * (xc - x0) / (x1 - x0)
            if np.abs(yc) >= block:
                if (np.abs(y1) <= block) or (np.sign(yc) != np.sign(y1)):
                    yc = np.sign(y0) * (block + 1)
                    xc = x0 + (x1 - x0) * (yc - y0) / (y1 - y0)
                    if np.sign(yc) == np.sign(gap):
                        if np.sign(xc) == np.sign(gap):
                            if np.abs(xc) > block - 2*np.abs(gap):
                                if np.abs(xc) < block - np.abs(gap):
                                    yc += -2*np.sign(gap)
                else:
                    xc = x1
                    yc = y1
    elif np.abs(y0) > block:
        if (np.abs(y1) <= block) or (np.sign(y0) != np.sign(y1)):
            yc = np.sign(y0) * (block + 1)
            xc = x0 + (x1 - x0) * (yc - y0) / (y1 - y0)
            if np.sign(yc) == np.sign(gap):
                if xc < 1 * (block - np.abs(gap)):
                    if np.sign(xc) == np.sign(gap):
                        if np.abs(xc) > block - 2*np.abs(gap):
                            if np.abs(xc) < block - np.abs(gap):
                                yc += -2*np.sign(gap)
            if np.abs(xc) >= block:
                xc = x1
                yc = y1
    return [xc, yc]#output the next postition

# for 2-dimensional s-rw in Fig 3C-I, SupFig 3, SummaryFig
# for Fig 2E,F,  activate and return "[traj_x, traj_y]"
# To set obstacles for Fig 3G-I, activate """--""" parts
def random_walk_2D(sl_p=0.1, init_pos=10/np.sqrt(2), tar_pos=0, pow_n=1, tmax=10**5, border=0):
    target = [tar_pos, tar_pos]
    random_direction = np.random.rand(tmax) * 2 * np.pi
    drx = np.cos(random_direction)
    dry = np.sin(random_direction)
    xt0 = init_pos
    yt0 = init_pos
    r_distance = ((xt0 - target[0])**2 + (yt0 - target[1])**2) ** (1/2)
    for t in range(tmax - 1):
        if pow_n == -1:
            step_length = (np.log(r_distance + 1)) * sl_p
        else:
            step_length = r_distance**pow_n * sl_p#usually pow_n=1

        if border == 0:#without boundary
            xt1 = xt0 + drx[t] * step_length 
            yt1 = yt0 + dry[t] * step_length
        else:#with boundary
            xt1 = xt0 + drx[t] * step_length 
            yt1 = yt0 + dry[t] * step_length
            [xt1, yt1] = inside_out(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=border, gap=0)
            """#To set obstacles, the previous 3 lines are replaced by the followings
            if max(np.abs(xt0),np.abs(yt0)) > 300:
                [xt1, yt1] = outside_in(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=300, gap=-50)
                [xt1, yt1] = inside_out(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=500, gap=0)
            elif max(np.abs(xt0),np.abs(yt0)) > 100:
                [xt1, yt1] = outside_in(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=100, gap=50)
                [xt1, yt1] = inside_out(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=300, gap=-50)
            elif max(np.abs(xt0),np.abs(yt0)) > 50:
                [xt1, yt1] = outside_in(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=50, gap=-20)
                [xt1, yt1] = inside_out(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=100, gap=50)
            else:
                [xt1, yt1] = inside_out(x0=xt0, x1=xt1, y0=yt0, y1=yt1, block=50, gap=-20)        
            """
        xt0 = xt1
        yt0 = yt1

        if max(np.abs(xt0), np.abs(yt0)) < 10**17:
            r_distance = ((xt0 - target[0])**2 + (yt0 - target[1])**2) ** (1/2)
        else:
            break
    return r_distance

#Output the trajectory in random_walk_2D
def random_walk_2D_traj(sl_p=0.1, init_pos=10/np.sqrt(2), tar_pos=0, pow_n=1, tmax=10**5, border=0):
    target = [tar_pos, tar_pos]
    random_direction = np.random.rand(tmax) * 2 * np.pi
    drx = np.cos(random_direction)
    dry = np.sin(random_direction)
    traj_x = np.zeros(tmax)
    traj_y = np.zeros(tmax)
    traj_x[0] = init_pos
    traj_y[0] = init_pos
    r_distance = ((traj_x[0] - target[0])**2 + (traj_y[0] - target[1])**2) ** (1/2)
    for t in range(tmax - 1):
        if pow_n == -1:
            step_length = (np.log(r_distance + 1)) * sl_p
        else:
            step_length = r_distance**pow_n * sl_p

        if border == 0:
            xt1 = traj_x[t] + drx[t] * step_length 
            yt1 = traj_y[t] + dry[t] * step_length
        else:
            xt1 = traj_x[t] + drx[t] * step_length 
            yt1 = traj_y[t] + dry[t] * step_length
            [xt1, yt1] = inside_out(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=border, gap=0)
            """#To set obstacles, the previous 3 lines are replaced by the followings
            if max(np.abs(traj_x[t]),np.abs(traj_y[t])) > 300:
                [xt1, yt1] = outside_in(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=300, gap=-50)
                [xt1, yt1] = inside_out(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=500, gap=0)
            elif max(np.abs(traj_x[t]),np.abs(traj_y[t])) > 100:
                [xt1, yt1] = outside_in(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=100, gap=50)
                [xt1, yt1] = inside_out(x0=traj_y[t], x1=xt1, y0=traj_y[t], y1=yt1, block=300, gap=-50)
            elif max(np.abs(traj_x[t]),np.abs(traj_y[t])) > 50:
                [xt1, yt1] = outside_in(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=50, gap=-20)
                [xt1, yt1] = inside_out(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=100, gap=50)
            else:
                [xt1, yt1] = inside_out(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=50, gap=-20)        
            """
        traj_x[t + 1] = xt1
        traj_y[t + 1] = yt1

        if max(np.abs(xt1), np.abs(yt1)) < 10**17:
            r_distance = ((xt1 - target[0])**2 + (yt1 - target[1])**2) ** (1/2)
        else:
            break
    #distance_traj = ((traj_x-target[0])**2 + (traj_y-target[0])**2)**(1/2)
    return np.array([traj_x, traj_y])#distance_traj

# for SupFig 2
def xr_rw(sl_p=0.1, init_pos=10/np.sqrt(2),tar_pos=0, tmax=10**4):
    target = [tar_pos, tar_pos]
    random_size = 1 - 2 * np.random.rand(tmax)
    random_angle = (1 - np.random.rand(tmax) * 2) * np.pi
    dny = np.sin(random_angle)
    xt0 = init_pos#x-coordinate of the position
    yt0 = init_pos#y-coordinate of the position
    for t in range(tmax -1):
        distance = ((xt0 - target[0])**2 + (yt0 - target[1])**2) ** (1/2)
        if distance < 10**20:
            xt0 += xt0 * sl_p * random_size[t] 
            yt0 += dny[t] * distance * sl_p
        else:
            break
    return distance

def c_rw(sl_p=0.1, init_pos=10/np.sqrt(2),tar_pos=0, tmax=10**4):
    target = [tar_pos, tar_pos]
    random_direction = (1 - np.random.rand(tmax) * 2) * np.pi
    xt0 = init_pos#x-coordinate of the position
    yt0 = init_pos#y-coordinate of the position
    distance_pre = ((xt0 - target[0])**2 + (yt0 - target[1])**2) ** (1/2)
    angle = np.random.rand() * 2 * np.pi
    for t in range(tmax -1):
        distance = ((xt0 - target[0])**2 + (yt0 - target[1])**2) ** (1/2)
        if distance < 10**20:
            if distance > distance_pre:#wrong direction
                angle += random_direction[t] * np.random.normal(loc=0, scale=0.5)
            else:#approaching the target
                angle += random_direction[t] * np.random.normal(loc=0, scale=0.25)
            xt0 += np.cos(angle) * distance * sl_p
            yt0 += np.sin(angle) * distance * sl_p
        else:
            break
        distance_pre = distance
    return distance

test_n = 1000#10**6 for Fig3B-D #100 for Fig 3I, SupFig 3B #1000 for SupFig 2, 3C
result = np.zeros(test_n , dtype=float)
for test in range(test_n):
    #result[test] = random_walk_1D(init_pos=10, tmax=100)#Fig 3B. init_pos=1,10,or 100, tmax=100 or 1000
    result[test] = random_walk_2D(sl_p=0.1,init_pos=10/np.sqrt(2),tar_pos=0,pow_n=1,tmax=100,border=0)#Fig 3C. init_pos=1/np.sqrt(2),10/np.sqrt(2),or 100/np.sqrt(2), tmax=100 or 1000
    #result[test] = random_walk_2D(sl_p=0.1,init_pos=10/np.sqrt(2),tar_pos=0,pow_n=1,tmax=10**3,border=500)#Fig 3D. tmax=10**3, 10**4 (testn=10**6) or 10**5 (testn=10**4)
    #result[test] = random_walk_2D(sl_p=0.1,init_pos=400,tar_pos=0,pow_n=1,tmax=10**5,border=500)#Fig 3I 400->0. sl_p=0.1, 0.02 or 0.5. Activate obstacles.
    #result[test] = random_walk_2D(sl_p=0.1,init_pos=0,tar_pos=400,pow_n=1,tmax=10**5,border=500)#Fig 3I 0->400. sl_p=0.1, 0.02 or 0.5. Activate obstacles.
    #result[test] = xr_rw(sl_p=0.1, init_pos=10/np.sqrt(2),tar_pos=0,tmax=10**3)#SupFig 2. tmax=1000 or 10**4
    #result[test] = c_rw(sl_p=0.1, init_pos=10/np.sqrt(2),tar_pos=0,tmax=10**3)#SupFig 2. tmax=1000 or 10**4
    #result[test] = random_walk_2D(sl_p=0.01,init_pos=10/np.sqrt(2),tar_pos=0,pow_n=1,tmax=10**5,border=500)#SupFig 3B-C. sl_p=0.5, 0.1, or 0.02 in SupFig 3C.
    #result[test] = random_walk_2D(sl_p=0.001,init_pos=10/np.sqrt(2),tar_pos=0,pow_n=2,tmax=10**5,border=500)#SupFig 3C.sl_p=0.01, 0.001, or 0.0001
    #result[test] = random_walk_2D(sl_p=1,init_pos=10/np.sqrt(2),pow_n=0.5,tar_pos=0,tmax=10**5,border=500)#SupFig 3C.(sl_p=1, pow_n=0.5) or (sl_p=0.417, pow_n=-1)

print([np.mean(result),np.median(result)])
# for Fig 3E-H Movie 1-2, SummaryFig
#traj = random_walk_2D_traj(sl_p=0.1,init_pos=10,tar_pos=0,pow_n=1,tmax=10**5,border=0)#Fig 3E
#traj = random_walk_2D_traj(sl_p=0.1,init_pos=10,tar_pos=0,pow_n=1,tmax=10**7,border=0)#Fig 3F. tmax was breaked at 1000743. For the blue dots, tmax=10**6, border=500
traj = random_walk_2D_traj(sl_p=0.1,init_pos=400,tar_pos=0,pow_n=1,tmax=10**5,border=500)#Fig 3G Movie 1, Activate obstacles.
#traj = random_walk_2D_traj(sl_p=0.1,init_pos=0,tar_pos=400,pow_n=1,tmax=10**5,border=500)#Fig 3H.Movie 2 Activate obstacles.
#traj012 = random_walk_2D_traj(sl_p=0.1,init_pos=400,tar_pos=0,pow_n=1,tmax=10**3,border=500)#SummaryFig, save 3 trajectories. Without obstacles.
#np.savez_compressed('data0', traj_s=traj)
#loaded = np.load('data0.npz')
#traj = loaded['traj_s']#[:,:1000]

#trajectory
import matplotlib.pyplot as plt
#To make Fig 3E,G,H # for Fig 3G,H blocks should be activated
#block1=np.array([[-200, -300], [300,-300],[300,300],[-300,300],[-300,-300],[-250,-300]]).T
#block2=np.array([[0, 100], [-100,100],[-100,-100],[100,-100],[100,100],[50,100]]).T
#block3=np.array([[-10, -50], [50,-50],[50,50],[-50,50],[-50,-50],[-30,-50]]).T
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
plt.rcParams["font.size"] = 18
t0=0#30000#change to the indicated starting time point
t1 =t0 + 1000#3000#tmax#change to the indicated end time point#1000 for SummaryFig
plt.plot(traj[0,t0:t1], traj[1,t0:t1],linewidth=0.5)
plt.plot(traj[0,t0], traj[1,t0], 'gs')#'kx')#
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