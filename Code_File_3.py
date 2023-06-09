#Code for random walk controlling step-length.
#Simulations in Figure 3. Use also Code_File_4 for visualization. 
import numpy as np

#For 1-dimensional s-rw in Figure 3b. 
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

# for 2-dimensional s-rw in Figure 3c-f, SupFig 1,3
# for Figure 2e,f,  activate and return "[traj_x, traj_y]"
# To set obstacles for Figure 3g-h, activate """--""" parts
def random_walk_2D(sl_p=0.1, init_pos=10/np.sqrt(2), pow_n=1, tmax=10**5, border=0):
    target = [0, 0]
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
def random_walk_2D_traj(sl_p=0.1, init_pos=10/np.sqrt(2), pow_n=1, tmax=10**5, border=0):
    target = [0, 0]
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
            if max(np.abs(xt0),np.abs(yt0)) > 300:
                [xt1, yt1] = outside_in(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=300, gap=-50)
                [xt1, yt1] = inside_out(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=500, gap=0)
            elif max(np.abs(xt0),np.abs(yt0)) > 100:
                [xt1, yt1] = outside_in(x0=traj_x[t], x1=xt1, y0=traj_y[t], y1=yt1, block=100, gap=50)
                [xt1, yt1] = inside_out(x0=traj_y[t], x1=xt1, y0=traj_y[t], y1=yt1, block=300, gap=-50)
            elif max(np.abs(xt0),np.abs(yt0)) > 50:
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
    return np.array([traj_x, traj_y])

# for SupFig 2
def xr_rw(sl_p=0.1, init_pos=10/np.sqrt(2), tmax=10**4):
    target = [0, 0]
    random_size = 1 - 2 * np.random.rand(tmax)
    random_angle = (1 - np.random.rand(tmax) * 2) * np.pi
    dny = np.sin(random_angle)
    xt = init_pos#x-coordinate of the position
    yt = init_pos#y-coordinate of the position
    for t in range(tmax -1):
        distance = ((xt - target[0])**2 + (yt - target[1])**2) ** (1/2)
        if distance < 10**20:
            xt += xt * sl_p * random_size[t] 
            yt += dny[t] * distance * sl_p
        else:
            break
    return distance

def c_rw(sl_p=0.1, init_pos=10/np.sqrt(2), tmax=10**4):
    target = [0, 0]
    random_direction = (1 - np.random.rand(tmax) * 2) * np.pi
    xt = init_pos#x-coordinate of the position
    yt = init_pos#y-coordinate of the position
    distance_pre = ((xt - target[0])**2 + (yt - target[1])**2) ** (1/2)
    angle = np.random.rand() * 2 * np.pi
    for t in range(tmax -1):
        distance = ((xt - target[0])**2 + (yt - target[1])**2) ** (1/2)
        if distance < 10**20:
            if distance > distance_pre:#wrong direction
                angle += random_direction[t] * np.random.normal(loc=0, scale=0.5)
            else:#approaching the target
                angle += random_direction[t] * np.random.normal(loc=0, scale=0.25)
            xt += np.cos(angle) * distance * sl_p
            yt += np.sin(angle) * distance * sl_p
        else:
            break
        distance_pre = distance
    return distance

test_n = 1000#10**6 for Fig3b-d #100 for Fig 3h, SupFig 3b #1000 for SupFig 2, 3c
result = np.zeros(test_n , dtype=float)
for test in range(test_n):
    #result[test] = random_walk_1D(init_pos=10, tmax=100)#Fig 3b. init_pos=1,10,or 100, tmax=100 or 1000
    result[test] = random_walk_2D(sl_p=0.1,init_pos=10/np.sqrt(2),pow_n=1,tmax=100,border=0)#Fig 3c. init_pos=1/np.sqrt(2),10/np.sqrt(2),or 100/np.sqrt(2), tmax=100 or 1000
    #result[test] = random_walk_2D(sl_p=0.1,init_pos=10/np.sqrt(2),pow_n=1,tmax=10**3,border=500)#Fig 3d. tmax=10**3, 10**4 (testn=10**6) or 10**5 (testn=10**4)
    #result[test] = random_walk_2D(sl_p=0.1,init_pos=400,pow_n=1,tmax=10**5,border=500)#Fig 3h. sl_p=0.1, 0.02 or 0.5. Activate obstacles.
    #result[test] = xr_rw(sl_p=0.1, init_pos=10/np.sqrt(2), tmax=10**3)#SupFig 2. tmax=1000 or 10**4
    #result[test] = c_rw(sl_p=0.1, init_pos=10/np.sqrt(2), tmax=10**3)#SupFig 2. tmax=1000 or 10**4
    #result[test] = random_walk_2D(sl_p=0.01,init_pos=10/np.sqrt(2),pow_n=1,tmax=10**5,border=500)#SupFig 3b-c. sl_p=0.5, 0.1, or 0.02 in SupFig 3b.
    #result[test] = random_walk_2D(sl_p=0.001,init_pos=10/np.sqrt(2),pow_n=2,tmax=10**5,border=500)#SupFig 3c.sl_p=0.01, 0.001, or 0.0001
    #result[test] = random_walk_2D(sl_p=1,init_pos=10/np.sqrt(2),pow_n=0.5,tmax=10**5,border=500)#SupFig 3c.(sl_p=1, pow_n=0.5) or (sl_p=0.417, pow_n=-1)

print([np.mean(result),np.median(result)])
# for Figure 3e,f,g, SupFig 1,4, Movie 1
#traj = random_walk_2D_traj(sl_p=0.1,init_pos=10,pow_n=1,tmax=10**5,border=0)#Fig 3e
#traj = random_walk_2D_traj(sl_p=0.1,init_pos=10,pow_n=1,tmax=10**7,border=0)#Fig 3f. tmax was breaked at 1000743. For the blue dots, tmax=10**6, border=500
#traj012 = random_walk_2D_traj(sl_p=0.1,init_pos=400,pow_n=1,tmax=10**3,border=500)#SupFig 1, save 3 trajectories. Without obstacles.
traj = random_walk_2D_traj(sl_p=0.1,init_pos=400,pow_n=1,tmax=10**5,border=500)#SupFig 4. Activate obstacles.

#np.savez_compressed('data0', traj_s=traj)
#loaded = np.load('data0.npz')
#traj = loaded['traj_s']#[:,:1000]

#trajectory
import matplotlib.pyplot as plt
#To make Figure 3e, g and SupFig4# for Fig 3g blocks should be activated
#block1=np.array([[-200, -300], [300,-300],[300,300],[-300,300],[-300,-300],[-250,-300]]).T
#block2=np.array([[0, 100], [-100,100],[-100,-100],[100,-100],[100,100],[50,100]]).T
#block3=np.array([[-10, -50], [50,-50],[50,50],[-50,50],[-50,-50],[-30,-50]]).T
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
plt.rcParams["font.size"] = 18
t0=0#13000#change to the indicated starting time point
t1 =t0 + 1000#3000#tmaxchange to the indicated end time point#1000 for supFig1
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
