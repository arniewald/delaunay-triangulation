import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from time import time
from qpsolvers import solve_qp

from functions.reading_functions import extract_run_params, read_circles_data
from functions.initialization_functions import refine
from functions.measuring_functions import adjacency
from functions.class_functions import premeasurement, fully_train
from functions.plotting_functions import plot_classifier, plot_classifier_scrollable, plot_metrics
from classes.Classifier import *
from classes.Trainer import Trainer
from classes.Measurer import Measurer
from classes.Writer import Writer
from classes.Reader import Reader

data_name = 'circles'
size = 100
al = 0.3
errw = 0.5

data, labels, dim = read_circles_data(n_samples = 10000, noise = 0.1)

colors = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in range(len(data))]
colors = [[max(c[0],0),0,max(c[2],0)] for c in colors]
""" fig0, ax0 = plt.subplots()
ax0.scatter(data[:,0],data[:,1],color=colors,s=1)
ax0.set_xlabel('Feature 1')
ax0.set_ylabel('Feature 2')
ax0.set_title('Initial data') """

hull = ConvexHull(data)
hull = list(hull.vertices)

length = len(hull)
out_hull_size = max(0,size-length)
out_hull = random.sample([i for i in range(len(data)) if i not in hull],out_hull_size)
sample = np.concatenate([hull,out_hull])
sample = np.array([int(i) for i in sample])
sample.sort()
out_hull = [i for i in range(len(sample)) if sample[i] in out_hull]
rem = np.array([int(i) for i in range(len(data)) if int(i) not in sample])
tri, bc = subtesselate(data,sample,dim)

""" fig1, ax1 = plt.subplots()
colors_sample = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in sample]
colors_sample = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
ax1.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax1.scatter(data[sample][:,0],data[sample][:,1],s=10,color=colors_sample,alpha=1)

ax1.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('First triangulation')
#ax1.set_title('Without refining')
plt.show() """

""" data, labels, hull = refine(data,labels,dim,rep=1)
length = len(hull)
out_hull_size = max(0,size-length)
out_hull = random.sample([i for i in range(len(data)) if i not in hull],out_hull_size)
sample = np.concatenate([hull,out_hull])
sample = np.array([int(i) for i in sample])
sample.sort()
out_hull = [i for i in range(len(sample)) if sample[i] in out_hull]
rem = np.array([int(i) for i in range(len(data)) if int(i) not in sample])
tri, bc = subtesselate(data,sample,dim)

fig1, ax1 = plt.subplots()
colors_sample = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in sample]
colors_sample = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
ax1.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax1.scatter(data[sample][:,0],data[sample][:,1],s=10,color=colors_sample,alpha=1)

ax1.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Refining')
plt.show()
 """
B = labels[rem].copy()
A = np.zeros((len(rem),len(sample)), dtype=np.float32)
for i in range(len(rem)):
    x = bc[rem[i]][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
    y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
    A[i,y] = x[1:(dim+2)]

At = np.transpose(A)
print(B.shape,At.shape)
P = csc_matrix(2*np.matmul(At,A))
q = -2*np.matmul(At,B)
lb = np.zeros(len(sample))
ub = np.ones(len(sample))
print('Starting lsq fit...')
labels[sample] = solve_qp(P,q,lb=lb,ub=ub,solver='piqp')


e = abs(np.matmul(A[:len(rem),:],labels[sample])-B[:len(rem)])
P2 = P
q2 = -2*np.matmul(At,e)
err = solve_qp(P2,q2,lb=lb,solver='piqp')

fig2, ax2 = plt.subplots()
colors_sample = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in sample]
colors_sample = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
sizes = np.ones(len(err))+100*err
ax2.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax2.scatter(data[sample][:,0],data[sample][:,1],s=sizes,color=colors_sample,alpha=1)

ax2.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Labels estimation')

L = len(out_hull)
adj = adjacency(tri,out_hull)
disin = np.zeros(L)
errin = np.zeros(L) 
for i in range(L):
    try: 
        errin[i] = sample[adj[out_hull[i]][np.argmax([(err[j]-err[out_hull[i]]) for j in adj[out_hull[i]]])]]
        disin[i] = sample[adj[out_hull[i]][np.argmax([sum((data[sample[j]]-data[sample[out_hull[i]]])**2) for j in adj[out_hull[i]]])]]
    except Exception as e:
        print("Exception at tr node ",sample[i],": ",adj[i]) 
        print("Exception: ",e)
        print(sample,err)
errin = [int(i) for i in errin]
disin = [int(i) for i in disin]
new_data = data.copy()
new_data[sample[out_hull]] += al*(errw*(new_data[errin]-new_data[sample[out_hull]])+(1-errw)*(new_data[disin]-new_data[sample[out_hull]]))
new_tri, new_bc = subtesselate(new_data,sample,dim)
err_data = data.copy()
dist_data = data.copy()
err_data[sample[out_hull]] += al*errw*(err_data[errin]-err_data[sample[out_hull]])
dist_data[sample[out_hull]] += al*(1-errw)*(dist_data[disin]-dist_data[sample[out_hull]])
err_dif = err_data-data
dist_dif = dist_data-data
dif = new_data-data
              
fig3, ax3 = plt.subplots()
hull = [i for i in range(len(sample)) if i not in out_hull]
colors_hull = [colors[sample[i]] for i in hull]
new_colors = [colors_sample[i] for i in out_hull]
ax3.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
#ax3.scatter(data[sample][:,0],data[sample][:,1],s=sizes,color=colors_sample,alpha=1)
ax3.scatter(data[sample[hull]][:,0],data[sample[hull]][:,1],s=sizes[hull],color=colors_hull,alpha=1)
ax3.scatter(data[sample[out_hull]][:,0],data[sample[out_hull]][:,1],s=sizes[out_hull],color=new_colors,marker='x',alpha=1)
ax3.scatter(new_data[sample[out_hull]][:,0],new_data[sample[out_hull]][:,1],color=new_colors,alpha=1,s=sizes[out_hull])
ax3.quiver(data[sample[out_hull]][:,0],data[sample[out_hull]][:,1],err_dif[sample[out_hull]][:,0],err_dif[sample[out_hull]][:,1],angles='xy',scale=1,scale_units='xy',units='xy',headwidth=2,color='purple',label='Error gradient')
ax3.quiver(data[sample[out_hull]][:,0],data[sample[out_hull]][:,1],dist_dif[sample[out_hull]][:,0],dist_dif[sample[out_hull]][:,1],angles='xy',scale=1,scale_units='xy',units='xy',headwidth=2,color='orange',label='Distance gradient')
ax3.quiver(data[sample[out_hull]][:,0],data[sample[out_hull]][:,1],dif[sample[out_hull]][:,0],dif[sample[out_hull]][:,1],angles='xy',scale=1,scale_units='xy',units='xy',headwidth=2,color='black',label='Total gradient')
ax3.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=0.5,linewidth=0.5)
ax3.triplot(new_data[sample][:,0],new_data[sample][:,1],new_tri.simplices,color="black",alpha=1,linewidth=0.5)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_title('Moving data')
ax3.legend(loc='upper right')

fig4, ax4 = plt.subplots()
ax4.scatter(new_data[rem][:,0],new_data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax4.scatter(data[sample[hull]][:,0],data[sample[hull]][:,1],s=sizes[hull],color=colors_hull,alpha=1)
ax4.scatter(new_data[sample[out_hull]][:,0],new_data[sample[out_hull]][:,1],color=new_colors,alpha=1,s=sizes[out_hull])
ax4.triplot(new_data[sample][:,0],new_data[sample][:,1],new_tri.simplices,color="black",alpha=1,linewidth=0.5)
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')
ax4.set_title('New data')

plt.show()