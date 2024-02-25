import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.datasets import make_circles, make_classification, make_moons

from functions import *

dim = 2
n_samples = 10000
size = 100
al = 0.1
errw = 0.5
time = 10
rep = 1

fig, axs = plt.subplots(2,2,figsize=(6, 6))
fig.tight_layout()
files = ["esterrs.txt","lsqferrs.txt","sigmas.txt","rerrs.txt"]
titles = ['Estimated error',"Linear fit error", "Variance", "Real error"]
for i in range(len(files)):
    file = open(files[i],"r")
    ys = file.readlines()[0]
    ys = ys.split("\t")
    ys.pop()
    ysaux = []
    for y in ys:
        ysaux.append(float(y))
    ys = ysaux
    axs[i//2,i%2].plot(range(len(ys)),ys)
    axs[i//2,i%2].set_title(titles[i])

plt.show()