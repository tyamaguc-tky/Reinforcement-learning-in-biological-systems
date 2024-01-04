#Pattern generation in Figure 4, SupFig S5, Movie 3-5
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=0.1):
    return 1 / (1 + np.exp(-a * x))

border_x = 500#1000 in Figure 4F,I,J, SupFig S5D and movie 5
border_y = 500#200 in Figure 4F,I,J, SupFig S5D and movie 5
frame = 50#Matrix is produced when the location of the cell is not near the boundary.
cell_n = 1000#the number of each type of cells 
rand_index = np.random.permutation(np.arange(0,(border_x-2*frame)*(border_y-2*frame)))[:cell_n*2]
#Initial position is random, but not overlapped.
loc_a = np.zeros((2, cell_n),dtype=int)#locations of cells a
loc_b = np.zeros((2, cell_n),dtype=int)#locations of cells b
for i in range(cell_n):#Initial position
    loc_a[:, i] = np.divmod(rand_index[i], border_y-2*frame)
    loc_b[:, i] = np.divmod(rand_index[i+cell_n], border_y-2*frame) 
loc_a += frame
loc_b += frame
ex_a = np.zeros((border_x, border_y),dtype=float)#extracellular matrix a
ex_b = np.zeros((border_x, border_y),dtype=float)#extracellular matrix b
curr_pos = np.zeros((border_x, border_y), dtype=int)
for i in range(cell_n):#1 and -1 indicates the presence of a and b
    curr_pos[loc_a[0, i],loc_a[1, i]] = 1
    curr_pos[loc_b[0, i],loc_b[1, i]] = -1 

tmax = 10000#100000 in Fig 4C and SupFig S5
sl_max = 10#2 in SupFig S5B and SummaryFig
wid = 20#width of the painting area of matrix
step = np.ones((2, cell_n)) * sl_max
sig_a = 0.1#gain of sigmoid curve
sig_h = 0.3#-gain of sigmoid curve for the upper limit
thre_a = 50#threshold. This is changed in each pattern
thre_h = 200#can be changed in each pattern
h_p = 0.3#weight of the other matrix
data_n = 10000#10000#tmax#for recording
pos_data = np.zeros((border_x, border_y, data_n+1), dtype=np.int8)
pos_data[:,:,0] = curr_pos
pos_data_e100 = np.zeros((border_x, border_y, data_n+1), dtype=np.int8)
pos_data_e100[:,:,0] = curr_pos
step = np.zeros((2, cell_n), dtype=float)
random_direction = np.zeros((2, cell_n), dtype=float)
step_xy = np.zeros((2, cell_n, 2), dtype=int)

cell_c = 10#Number of cell death
cell_new = np.zeros((2, cell_c),dtype=int)

for t in range(tmax - 1):
    if t < 0:# change to != 0 in case with cell death:#when < 0, this part is skipped without cell death
        cell_new[0,:] = np.random.randint(frame, border_x-frame, size=cell_c).astype(int)
        cell_new[1,:] = np.random.randint(frame, border_y-frame, size=cell_c).astype(int)
        cell_i = np.random.randint(cell_n*2, size=cell_c)#index of killed cells
        #cell_ch = int(cell_c/2)#Activate these 2 lines to kill improper cells in Fig 4G,I,K
        #cell_i = np.append(np.argsort(step[0,:])[-cell_ch:],np.argsort(step[1,:])[-(cell_c-cell_ch):]+cell_n)
        for i in range(cell_c):
            if curr_pos[cell_new[0,i], cell_new[1,i]] == 0:            
                if cell_i[i] < cell_n:#cell a is replaced
                    curr_pos[loc_a[0,cell_i[i]], loc_a[1,cell_i[i]]] = 0
                    curr_pos[cell_new[0,i], cell_new[1,i]] = 1
                    loc_a[:,cell_i[i]] = cell_new[:,i]
                else:#cell b is replaced
                    curr_pos[loc_b[0,cell_i[i]-cell_n], loc_b[1,cell_i[i]-cell_n]] = 0
                    curr_pos[cell_new[0,i],cell_new[1,i]] = -1
                    loc_b[:,cell_i[i]-cell_n] = cell_new[:,i]

    ex_a = ex_a * 0.9#decay of matrix
    ex_b = ex_b * 0.9

    random_direction = np.random.rand(2, cell_n) * np.pi * 2
    #activate either sets from No1~No5
#No1 independet, Fig 4A
    #step[0,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_a[0,:],loc_a[1,:]] - thre_a), a=sig_a)) * sl_max)
    #step[1,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_b[loc_b[0,:],loc_b[1,:]] - 2*thre_a), a=sig_a)) * sl_max)
#No2 separation, Fig 4B, SummaryFig
#    step[0,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_a[0,:],loc_a[1,:]] - ex_b[loc_a[0,:],loc_a[1,:]] - thre_a), a=sig_a)) * sl_max)
#    step[1,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_b[loc_b[0,:],loc_b[1,:]] - ex_a[loc_b[0,:],loc_b[1,:]] - 2*thre_a), a=sig_a)) * sl_max)
#No3 spotted, Fig 4C,K,L, movie 3-4
    #step[0,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_a[0,:],loc_a[1,:]] - 2*thre_a), a=sig_a)) * sl_max)
    #step[1,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_b[0,:],loc_b[1,:]] - thre_a), a=sig_a)+ sigmoid(x=1*(ex_a[loc_b[0,:],loc_b[1,:]] - thre_h), a=sig_h)) * sl_max)
#No 4 faced lines, Fig 4D
    #step[0,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_a[0,:],loc_a[1,:]]-2*thre_a), a=sig_a) + h_p*sigmoid(x=-1*(ex_b[loc_a[0,:],loc_a[1,:]]-thre_a), a=sig_a) + h_p*sigmoid(x=1*(ex_b[loc_a[0,:],loc_a[1,:]]-thre_h), a=sig_h)) * sl_max)
    #step[1,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_b[loc_b[0,:],loc_b[1,:]]-2*thre_a), a=sig_a) + h_p*sigmoid(x=-1*(ex_a[loc_b[0,:],loc_b[1,:]]-thre_a), a=sig_a) + h_p*sigmoid(x=1*(ex_a[loc_b[0,:],loc_b[1,:]]-thre_h), a=sig_h)) * sl_max)
#No 5 striped, Fig 4 E-J, SupFig S5, movie 5
    step[0,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_a[loc_a[0,:],loc_a[1,:]]-2*thre_a), a=sig_a)+sigmoid(x=1*(ex_a[loc_a[0,:],loc_a[1,:]]-2*thre_h), a=sig_h) + h_p*sigmoid(x=-1*(ex_b[loc_a[0,:],loc_a[1,:]]-thre_a/2), a=sig_a) + h_p*sigmoid(x=1*(ex_b[loc_a[0,:],loc_a[1,:]]-thre_h/2), a=sig_h)) * sl_max)
    step[1,:] = np.random.exponential((0.1+sigmoid(x=-1*(ex_b[loc_b[0,:],loc_b[1,:]]-2*thre_a), a=sig_a)+sigmoid(x=1*(ex_b[loc_b[0,:],loc_b[1,:]]-2*thre_h), a=sig_h) + h_p*sigmoid(x=-1*(ex_a[loc_b[0,:],loc_b[1,:]]-thre_a/2), a=sig_a) + h_p*sigmoid(x=1*(ex_a[loc_b[0,:],loc_b[1,:]]-thre_h/2), a=sig_h)) * sl_max)

    step_xy[:,:,0] = step[:,:] * np.cos(random_direction[:,:])
    step_xy[:,:,1] = step[:,:] * np.sin(random_direction[:,:]) 
    step_xy = step_xy.astype(int)#the position is discrete in lattice
    
    for i in range(cell_n):#each cell walks and paint matrix
        [xt1, yt1] = loc_a[:,i] + np.squeeze(step_xy[0,i,:])#next position
        if np.min([xt1, yt1]) >= 0 and xt1 < border_x and yt1 < border_y:#if the next position is inside the boundary
            if curr_pos[xt1, yt1] == 0:#if the position is not occupied by other cells
                curr_pos[loc_a[0,i], loc_a[1,i]] = 0#move out
                loc_a[:,i] = [xt1, yt1]#new position
            if np.min([xt1, yt1]) >= frame and xt1 < border_x - frame and yt1 < border_y - frame:#if the position is inside the frame
                ex_a[xt1-wid:xt1+wid, yt1-wid:yt1+wid] += 1#matrix is painted in the square area                
        curr_pos[loc_a[0,i], loc_a[1,i]] = 1#move in
           
        [xt1, yt1] = loc_b[:,i] + np.squeeze(step_xy[1,i,:])#next position
        if np.min([xt1, yt1]) >= 0 and xt1 < border_x and yt1 < border_y:
            if curr_pos[xt1, yt1] == 0:
                curr_pos[loc_b[0,i], loc_b[1,i]] = 0
                loc_b[:,i] = [xt1, yt1]
            if np.min([xt1, yt1]) >= frame and xt1 < border_x - frame and yt1 < border_y - frame:
                ex_b[xt1-wid:xt1+wid, yt1-wid:yt1+wid] += 1
        curr_pos[loc_b[0,i], loc_b[1,i]] = -1

    if (t+1)%(tmax/data_n)==0:#Recording for movie and progress
        pos_data[:,:,int((t+1)//(tmax/data_n))] = curr_pos
pos_data[:,:,-1] = curr_pos
"""#for movie 3-4, activate the followings instead of upper 3 lines.
    if t < 1000:
        pos_data[:,:,int(t+1)] = curr_pos
    if (t+1)%(100)==0:
        pos_data_e100[:,:,int((t+1)//(tmax/data_n))] = curr_pos
pos_data_e100[:,:,-1] = curr_pos
"""
np.savez_compressed('pattern',pos_data=pos_data)
#np.savez_compressed('pattern_movie',pos_data=pos_data, e100=pos_data_e100)

fig = plt.figure(figsize=(border_x/100,border_y/100))
plt.rcParams["font.size"] = 18
plt.plot(loc_a[0],loc_a[1],'ro', markersize=2)
plt.plot(loc_b[0],loc_b[1],'bs', markersize=2)
plt.xlim(0, border_x)
plt.ylim(0, border_y)
plt.show()

"""
t = 100#should be changed as needed
pos_a = np.where(pos_data[:,:,t]==1)
pos_b = np.where(pos_data[:,:,t]==-1)
fig = plt.figure(figsize=(border_x/100,border_y/100))
plt.plot(pos_a[0],pos_a[1],'ro', markersize=2)
plt.plot(pos_b[0],pos_b[1],'bs', markersize=2)
plt.xlim(0,border_x)
plt.ylim(0,border_y)
plt.show()
"""
"""#for movie
graphs = []
fig = plt.figure()
fig.set_figwidth(border_x/100)
fig.set_figheight(border_y/100)
plt.xlim(0,border_x)
plt.ylim(0,border_y)
for t in range(100):
    pos_a = np.where(pos_data[:,:,t]==1)
    pos_b = np.where(pos_data[:,:,t]==-1)
    g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
    g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
    g_t = plt.text(10, -10, f't={t}', fontsize=25)
    graphs.append(g_a + g_b + [g_t])
"""
"""
#for movie3-4
loaded = np.load('pattern_movie.npz')
pos_data =loaded['pos_data']
pos_data_e100 =loaded['e_100']
graphs = []
fig = plt.figure()
fig.set_figwidth(border_x/100)
fig.set_figheight(border_y/100)
plt.xlim(0,border_x)
plt.ylim(0,border_y)
for t in range(100):
    pos_a = np.where(pos_data[:,:,t]==1)
    pos_b = np.where(pos_data[:,:,t]==-1)
    g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
    g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
    g_t = plt.text(10, 505, f't={t}', fontsize=25)
    graphs.append(g_a + g_b + [g_t])
for t in range(100,1001):
    if t%(10)==0:
        pos_a = np.where(pos_data[:,:,t]==1)
        pos_b = np.where(pos_data[:,:,t]==-1)
        g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
        g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
        g_t = plt.text(10, 505, f't={t}', fontsize=25)
        graphs.append(g_a + g_b + [g_t])
for t in range(10,101):    
        pos_a = np.where(pos_data_e100[:,:,t]==1)
        pos_b = np.where(pos_data_e100[:,:,t]==-1)
        g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
        g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
        g_t = plt.text(10, 505, f't={"{:,}".format(t*100)}', fontsize=25)
        graphs.append(g_a + g_b + [g_t])        
for t in range(101,1002):    
    if t%(10)==0:
        pos_a = np.where(pos_data_e100[:,:,t]==1)
        pos_b = np.where(pos_data_e100[:,:,t]==-1)
        g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
        g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
        g_t = plt.text(10, 505, f't={"{:,}".format(t*1000)}', fontsize=25)
        graphs.append(g_a + g_b + [g_t])    
anim = animation.ArtistAnimation(fig, graphs, interval=1, repeat=False)
#plt.show()
anim.save('movie_pattern_spot.gif', writer='pillow')
"""
"""#for moive 5
loaded = np.load('pattern.npz')
pos_data =loaded['pos_data']
border_x = 1000 #in movie 5
border_y = 200 #in movie 5
import matplotlib.animation as animation
graphs = []
fig = plt.figure()
fig.set_figwidth(border_x/100)
fig.set_figheight(border_y/100+0.5)
plt.rcParams["font.size"] = 18
plt.xlim(0,border_x)
plt.ylim(0,border_y)
for t in range(2001):#len(pos_data[0,0,:])):
    pos_a = np.where(pos_data[:,:,t]==1)
    pos_b = np.where(pos_data[:,:,t]==-1)
    g_a = plt.plot(pos_a[0], pos_a[1], 'ro',markersize=2)
    g_b = plt.plot(pos_b[0], pos_b[1], 'bs',markersize=2)
    g_t = plt.text(10, 205, f't={"{:,}".format(t)}', fontsize=25)
    graphs.append(g_a + g_b + [g_t])
anim = animation.ArtistAnimation(fig, graphs, interval=1, repeat=False)
anim.save('movie_pattern_stripe_w_rcd.gif', writer='pillow')
#GIF file is converted to MP4 in Adobe web site.
"""