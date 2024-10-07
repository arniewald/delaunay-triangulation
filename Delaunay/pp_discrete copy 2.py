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
ax1.set_title('First triangulation')
#ax1.set_title('Without refining')
plt.show()

W_shape = ((len(rem)+len(sample)),(len(rem)+len(sample)))
#dim_labels = nº of possible labels - 1
#B = labels[:,0]
rem = [i for i in range(len(data)) if i not in sample]
B = labels[rem]

M = np.zeros((len(rem),len(sample)), dtype=np.float32)
for i in range(len(rem)):
    x = bc[rem[i]][1:(dim+3)]            #We extract index and barycentric coordinates of the i-th remaining point
    y = tri.simplices[int(x[0])]    #We extract the points of the triangulation containing the i-th remaining point
    M[i,y] = x[1:dim+2]
A = block_diag(*([M]))

adj = adjacency(tri,range(len(sample)))

S = np.zeros((len(sample),len(sample)), dtype=np.float32)
for i in range(len(sample)):
    nodes = [node for node in adj[i] if node != i]
    len_nodes = len(nodes)
    S[i,i] = -len_nodes
    for node in nodes:
        S[i,node] = 1
A = np.concatenate([A,block_diag(*([S]))])
B = np.concatenate([B,np.zeros(len(sample))])

W1 = np.zeros(W_shape)
for i in range(len(rem)):
    W1[i,i] = 1
for i in range(len(rem), (len(rem)+len(sample))):
    W1[i,i] = 0

W2 = np.zeros(W_shape)
for i in range(len(rem)):
    W2[i,i] = 1
for i in range(len(rem), (len(rem)+len(sample))):
    W2[i,i] = 10

At = np.transpose(A)
print(B.shape,At.shape)
P1 = csc_matrix(2*np.matmul(np.matmul(At,W1),A))
q1 = -2*np.matmul(np.matmul(At,W1),B)
P2 = csc_matrix(2*np.matmul(np.matmul(At,W2),A))
q2 = -2*np.matmul(np.matmul(At,W2),B)
lb = np.zeros(len(sample))
ub = np.ones(len(sample))
print('Starting lsq fit...')
labels1 = labels.copy()
labels2 = labels.copy()

labels1[sample] = solve_qp(P1,q1,lb=lb,ub=ub,solver='piqp')
labels2[sample] = solve_qp(P2,q2,lb=lb,ub=ub,solver='piqp')


e1 = abs(np.matmul(A[:len(rem),:],labels1[sample])-B[:len(rem)])
e2 = abs(np.matmul(A[:len(rem),:],labels2[sample])-B[:len(rem)])
Mt = np.transpose(M)
PP = csc_matrix(2*np.matmul(Mt,M))
qq1 = -2*np.matmul(e1,M)
qq2 = -2*np.matmul(e2,M)
lb = np.zeros(len(sample))
ub = np.ones(len(sample))
err1 = solve_qp(PP,qq1,G=None,h=None,A=None,b=None,lb=lb,ub=ub,solver='piqp')
err2 = solve_qp(PP,qq2,G=None,h=None,A=None,b=None,lb=lb,ub=ub,solver='piqp')

fig2, ax2 = plt.subplots()
colors_sample1 = [[min(1,labels1[i]),0,1-min(labels1[i],1)] for i in sample]
colors_sample1 = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample1]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
sizes1 = np.ones(len(err1))+100*err1
ax2.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax2.scatter(data[sample][:,0],data[sample][:,1],s=sizes1,color=colors_sample1,alpha=1)

ax2.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_title('Average weight = 0')

fig3, ax3 = plt.subplots()
colors_sample2 = [[min(1,labels2[i]),0,1-min(labels2[i],1)] for i in sample]
colors_sample2 = [[max(c[0],0),0,max(c[2],0)] for c in colors_sample2]
colors_rem = [[min(1,labels[i]),0,1-min(labels[i],1)] for i in rem]
colors_rem = [[max(c[0],0),0,max(c[2],0)] for c in colors_rem]
sizes2 = np.ones(len(err2))+100*err2
ax3.scatter(data[rem][:,0],data[rem][:,1],s=2,color=colors_rem,alpha=0.2)
ax3.scatter(data[sample][:,0],data[sample][:,1],s=sizes2,color=colors_sample2,alpha=1)
ax3.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=1,linewidth=0.5)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_title('Average weight = 10')

plt.show()