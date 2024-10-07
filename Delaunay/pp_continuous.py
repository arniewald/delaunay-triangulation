import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay, ConvexHull
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from time import time
from qpsolvers import solve_qp
from functions.initialization_functions import subtesselate

n_samples = 500
eps_sample = 50
size = 50
dim = 2

eps = 0.3
xs = np.array([np.random.normal() for _ in range(n_samples)])
ys = np.array([np.random.normal() for _ in range(n_samples)])
data = np.array([(xs[i],ys[i]) for i in range(n_samples)])
labels = np.sin(xs*ys) + eps*np.array([np.random.normal() for _ in range(n_samples)])

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


start = time()
print('Starting lsq fit...')
labels_delaunay = solve_qp(P,q,solver='piqp')

fig0 = plt.figure()
ax00 = fig0.add_subplot(121)
ax01 = fig0.add_subplot(122,projection='3d')
ax00.scatter(data[:,0],data[:,1],color='black',s=1)
ax00.set_xlabel('Feature 1')
ax00.set_ylabel('Feature 2')
ax01.scatter3D(data[:,0],data[:,1],labels,color='black',s=1)
ax01.set_xlabel('Feature 1')
ax01.set_ylabel('Feature 2')
ax01.set_zlabel('Label')

plt.show()

fig1 = plt.figure()
ax10 = fig1.add_subplot(121)
ax11 = fig1.add_subplot(122,projection='3d')
ax10.scatter(data[:,0],data[:,1],color='black',s=1)
ax10.scatter(data[sample][:,0],data[sample][:,1],color='black',s=10)
ax10.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=0.2,linewidth=0.5)
ax10.set_xlabel('Feature 1')
ax10.set_ylabel('Feature 2')
ax11.scatter3D(data[:,0],data[:,1],labels,color='black',s=1)
ax11.scatter3D(data[sample][:,0],data[sample][:,1],labels[sample],color='black',s=10)
ax11.set_xlabel('Feature 1')
ax11.set_ylabel('Feature 2')
ax11.set_zlabel('Label')

plt.show()

fig2 = plt.figure()
ax20 = fig2.add_subplot(121)
ax21 = fig2.add_subplot(122,projection='3d')
ax20.scatter(data[:,0],data[:,1],color='black',s=1)
ax20.scatter(data[sample][:,0],data[sample][:,1],color='black',s=10)
ax20.triplot(data[sample][:,0],data[sample][:,1],tri.simplices,color="black",alpha=0.2,linewidth=0.5)
ax20.set_xlabel('Feature 1')
ax20.set_ylabel('Feature 2')
ax21.scatter3D(data[:,0],data[:,1],labels,color='black',s=1)
ax21.scatter3D(data[sample][:,0],data[sample][:,1],labels[sample],color='black',s=10)
ax21.scatter3D(data[sample][:,0],data[sample][:,1],labels_delaunay,color='red',s=10)
ax21.set_xlabel('Feature 1')
ax21.set_ylabel('Feature 2')
ax21.set_zlabel('Label')

plt.show()