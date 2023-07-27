# for Figure 5, maze
#This simulation takes several months to complete
#You can change the value testn to 10 or 100. 
#If testn = 10, 1 week. If testn = 100, 1 month is required to complete. 
import numpy as np
import matplotlib.pyplot as plt

block1=np.array([[-200,-300],[300,-300],[300,300],[-300,300],[-300,-300],[-250,-300]]).T
block2=np.array([[0,100],[-100,100],[-100,-100],[100,-100],[100,100],[50,100]]).T
block3=np.array([[-10,-50],[50,-50],[50,50],[-50,50],[-50,-50],[-30,-50]]).T
border = 500
tmax = 10**5#Maximum number of steps in an Acutual walk
mtmax = 100#Maximum number of steps in MCTS
sl_p = 0.1#step-length is 0.1r
target = np.array([0,0])
n_direction0 = np.ones((border*2,border*2,4),dtype=float)#N
policy = np.zeros((border*2,border*2,4),dtype=float)#P
value = np.ones((border*2,border*2), dtype=float) * border * 2#V
for i in range(border*2):#initial value is the distance
    for j in range(border*2):
        value[i,j] = np.sqrt((i - border - target[0])**2 + (j - border - target[1])**2)
testn = 1000#repetition times of Acutual Walk#100
init_position = np.array([int(border*0.8),int(border*0.8)])#(400,400)#(0,0)
init_distance = np.sqrt(np.sum((init_position - target)**2))#565.685
repeat_MCTS = 1000#number of repetitions of MCTS
data_t = np.zeros(testn, dtype=int)#for recording
trajectory_data = np.zeros((2, tmax, testn),dtype=float)

def inside_out(x0, x1, y0, y1, block=50, gap=10):#same to Code_File_3
    xc = x1#next position
    yc = y1
    if np.abs(x0) < block:#current position is inside the block
        if np.abs(y0) < block:#current position is inside the block
            if np.abs(x1) >= block:
                    xc = np.sign(x1) * (block - 1)#stay near the wall
                    yc = y0 + (y1 - y0) * (xc - x0) / (x1 - x0)#intesection point
                    if np.abs(yc) >= block:#correct the intersction point
                        yc = np.sign(y1) * (block - 1)
                        xc = x0 + (x1 - x0) * (yc - y0) / (y1 - y0)
                        if np.abs(xc) >= block:
                            xc = np.sign(x1) * (block - 1)
                        if np.sign(yc) == np.sign(gap):#Check whether the point is in the gap
                            if np.sign(xc) == np.sign(gap):
                                if np.abs(xc) > block - 2*np.abs(gap):
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

def outside_in(x0, x1, y0, y1, block=50, gap=10):
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

def next_location(xy=init_position, distance=1.0, p_in_SS=policy, n_in_SS=n_direction0, pnr=0.8):#Single Step
    dir_U = p_in_SS[int(border + xy[0]), int(border + xy[1]), :]#P at loc(i)
    dir_Q = n_in_SS[int(border + xy[0]), int(border + xy[1]), :]#N at loc(i)
    direction = np.random.choice(range(4), p=pnr*(dir_U+0.1)/np.sum(dir_U+0.1) + (1-pnr)*(dir_Q)/np.sum(dir_Q))
    #randomly choose either of [0,1,2,3] based on Policy and Number of visiting
    random_direction = (-2 + direction + np.random.rand()) * np.pi / 2#if direction=0, select in the range of -pai~-pai/2
    dnx = np.cos(random_direction)
    dny = np.sin(random_direction)
    xt1 = xy[0] + sl_p * distance * dnx#Candidate of the next position
    yt1 = xy[1] + sl_p * distance * dny
    if max(np.abs(xy)) > 300:
         [xt1, yt1] = outside_in(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=300, gap=-50)
         [xt1, yt1] = inside_out(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=500, gap=0)
    elif max(np.abs(xy)) > 100:
         [xt1, yt1] = outside_in(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=100, gap=50)
         [xt1, yt1] = inside_out(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=300, gap=-50)
    elif max(np.abs(xy)) > 50:
         [xt1, yt1] = outside_in(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=50, gap=-20)
         [xt1, yt1] = inside_out(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=100, gap=50)
    else:
         [xt1, yt1] = inside_out(x0=xy[0], x1=xt1, y0=xy[1], y1=yt1, block=50, gap=-20)
    if np.abs(xt1) >= border - 0.01:
       xt1 = np.sign(xt1) * (border - 0.01)
    if np.abs(yt1) >= border - 0.01:
       yt1 = np.sign(yt1) * (border - 0.01)      
    return [xt1, yt1, direction]#next position and the selected direction

def MCTS(init_M=init_position, policy_M=policy, n_in_M=n_direction0, pn_r=0.8):#Monte Carlo tree search
    bwalk = np.zeros((2, mtmax), dtype=float)#walking trajectory in MCTS
    bwalk[:, 0] = init_M#initial position of a MCTS
    bwalk_d = np.zeros(mtmax, dtype=int)#data of selected direction
    distance_M = np.sqrt(np.sum((init_M - target)**2))
    for mt in range(mtmax - 1):
        [xt1, yt1, direction] = next_location(xy=bwalk[:, mt], distance=distance_M, p_in_SS=policy_M, n_in_SS=n_in_M, pnr=pn_r)
        bwalk[:, mt + 1] = [xt1, yt1]
        bwalk_d[mt] = direction
        distance_M = np.sqrt(np.sum((bwalk[:, mt + 1] - target)**2))
        if distance_M < 20:#successfuly reaching the destination
            bwalk_d[mt + 1] = 0#direction = 0
            break
    return [bwalk[:,:(mt+2)], bwalk_d[:(mt+2)]]#output the trajectory and the selected directions

def n_evaluation(bwalk, bwalk_d, n_in_M=n_direction0, value_M=value):#N_Evaluation after MCTS
    tlen = len(bwalk_d)#required step-number in a MCTS
    i_loc = bwalk.astype(int) + border#express the location in integers
    if tlen < mtmax:#if MCTS successfuly reached the destination, r < 20
        q_ts = 0#evaluation of the MCTS
        value_M[i_loc[0,-1], i_loc[1,-1]] = 0#value of the final location is assigned to 0
    else:
        q_ts = value_M[i_loc[0,-1], i_loc[1,-1]]#The value of MCTS is V at the last location
        
    for t in range(tlen - 1):#for the pathway in MCTS
        val = value_M[i_loc[0, t], i_loc[1, t]]#value of the location
        if val > q_ts:#MCTS makes the position better to some extent
            n_in_M[i_loc[0,t], i_loc[1,t], bwalk_d[t]] += (val - q_ts) / val / (tlen - t)#Selected direction is added by the improved value per step
            value_M[i_loc[0,t], i_loc[1,t]] = val - (val - q_ts)/(tlen - t - 1) * 0.01#Value of the location will be same to expected value after the step
        else:#if MCTS fails
            value_M[i_loc[0,t],i_loc[1,t]] = val * 0.99 + q_ts * 0.01#Value of the location becomes worse
    """#Followings can replace the above 7 lines
    for t in range(tlen):#for the pathway in MCTS
        val = value_M[i_loc[0, t], i_loc[1, t]]#value of the location
        if val > q_ts:#MCTS makes the position better to some extent
            n_in_M[i_loc[0,t], i_loc[1,t], bwalk_d[t]] += (val - q_ts) / val / (tlen - t)#Selected direction is added by the improved value per step
            value_M[i_loc[0,t], i_loc[1,t]] = val - (val - q_ts)/(tlen - t) * 0.02#Value of the location will be same to expected value after the step
        else:#if MCTS fails
            value_M[i_loc[0,t],i_loc[1,t]] = val * 0.98 + q_ts * 0.02#Value of the location becomes worse
    """
    return [n_in_M, value_M]#update N and V

def random_walk(init_a=init_position, policy_a=policy, value_a=value, n_MCTS=repeat_MCTS):#Actual Walk
    twalk = np.ones((2, tmax), dtype=float) * -10000
    twalk[:, 0] = init_a
    n_direction = n_direction0#Initialize N to [1,1,1,1] at any location
    distance_a = np.sqrt(np.sum((init_a - target)**2))
    pn = 0.5 + 0.5 * init_distance/(init_distance + n_MCTS)#increase as n_MCTS is small
    for tt in range(tmax - 1):
        for btest in range(n_MCTS):#MCTS and N_Evaluation
            [bwalk, bwalk_d] = MCTS(init_M=twalk[:,tt], policy_M=policy_a, n_in_M=n_direction, pn_r=pn)
            [n_direction, value_a] = n_evaluation(bwalk=bwalk,bwalk_d=bwalk_d,n_in_M=n_direction,value_M=value_a)
        [xt1, yt1, dire] = next_location(xy=twalk[:,tt], distance=distance_a, p_in_SS=policy_a, n_in_SS=n_direction,pnr=pn)
        twalk[:,tt + 1] = [xt1, yt1]#move to the next position
        distance_a = np.sqrt(np.sum((twalk[:,tt + 1] - target)**2))
        if distance_a < 20:#successfuly reaching the goal
            break
    return [twalk[:,:(tt+1)], n_direction, value_a]#output the trajectory, N, and V

def p_evaluation(policy_n=policy, n_final=n_direction0, n_MCTS=repeat_MCTS):#Evaluation of Actual Walk
    n_sum = np.sum(n_final, axis=2) - 4
    pn_ratio = 0.5 + 0.5 * init_distance/(init_distance + n_MCTS)
    for i in range(4):
        policy_n[:,:,i] = policy_n[:,:,i] * pn_ratio + (n_final[:,:,i] - 1)/(n_sum + 10-7) * (1 - pn_ratio)
    return policy_n#update P after an Actural Walk
    
for i in range(testn):
    [twalk, n_direction, value] = random_walk(init_a=init_position, policy_a=policy, value_a=value,n_MCTS=repeat_MCTS)
    data_t[i] = np.shape(twalk)[1]#Time to reach the goal
    if  data_t[i] < tmax:
        v_rw = 0#Evaluation result of the Actual Walk
    else:
        v_rw = value[int(twalk[0,-1]+border), int(twalk[1,-1]+border)]
    repeat_MCTS = int(0.9 * repeat_MCTS + 0.1 * data_t[i] * (v_rw + 1))#typically one tenth of step number of random walk
    policy = p_evaluation(policy_n=policy, n_final=n_direction, n_MCTS=repeat_MCTS)
    trajectory_data[:,:data_t[i],i] = twalk#for recording
#    if  i == 9:#for recording
#        policy10 = np.array(policy)
#        value10 = np.array(value)

#np.savez_compressed('maze', policy=policy, policy10=policy10, value=value, value10=value10, data=trajectory_data)
#loaded = np.load('maze.npz')
#trajectory_data = loaded['data']
#policy = loaded['policy']
#value = loaded['value']

#followings are codes for visualization
policy_angle = np.degrees(np.arctan2(policy[:,:,0] - policy[:,:,2], policy[:,:,1] - policy[:,:,3]) + np.pi/4)
policy_angle[policy_angle > 180] -= 360
border2=int(border*2)
fig = plt.figure(figsize=(8,6))
plt.rcParams["font.size"] = 18
ax=fig.add_subplot(111)
idx = np.arange((border2)**2) 
x, y, z = idx//(border2), idx%(border2), policy_angle[idx//border2, idx%border2]
im = ax.scatter(x-border, y-border, c=z, s=50, cmap="jet", vmin=-180, vmax=180)
fig.colorbar(im)
ax.set_ylim(-border, border)
ax.set_xlim(-border, border)
plt.show()

border2=int(border*2)
fig = plt.figure(figsize=(8,6))
plt.rcParams["font.size"] = 18
ax=fig.add_subplot(111)
idx = np.arange((border2)**2) 
x, y, z = idx//(border2), idx%(border2), value[idx//border2, idx%border2]
im = ax.scatter(x-border, y-border, c=z, s=50, cmap="jet")
fig.colorbar(im)
ax.set_ylim(-border, border)
ax.set_xlim(-border, border)
plt.show()

tt = 97#0#7#In SupFig1 trajectory of tt=999 is plotted
walk0 = trajectory_data[:,:data_t[tt],tt]
walk1 = trajectory_data[:,:data_t[tt+1],tt+1]
walk2 = trajectory_data[:,:data_t[tt+2],tt+2]
f = plt.figure()
f.set_figwidth(6)
f.set_figheight(6)
plt.plot(walk0[0, :], walk0[1, :],linewidth=1)
plt.plot(walk0[0,-1], walk0[1,-1], 'ro')
plt.plot(walk1[0, :], walk1[1, :],linewidth=1)
plt.plot(walk1[0,-1], walk0[1,-1], 'ro')
plt.plot(walk2[0, :], walk2[1, :],linewidth=1)
plt.plot(walk2[0,-1], walk0[1,-1], 'ro')
plt.plot(block1[0,:],block1[1,:], '-k',linewidth=3)
plt.plot(block2[0,:],block2[1,:], '-k',linewidth=3)
plt.plot(block3[0,:],block3[1,:], '-k',linewidth=3)
plt.xlim(-border, border)
plt.ylim(-border, border)
plt.show()