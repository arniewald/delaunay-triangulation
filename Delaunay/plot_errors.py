"""
Plots the error measurement. Includes: average lsq error, variance of lsq error, maximum error, variance of
size of edges of triangulation and real error.
"""

import matplotlib.pyplot as plt

folder = "errors/"
files = ["avs","sigmas","maxs","evars","rerrs"]
sufix = '_Iris3D_mc_bctime10_th0.2.txt'
files = [folder+f+sufix for f in files]
titles = ['Mean error' + sufix,"Error variance", "Maximum error","Edge variance","Real error"]
fig, axs = plt.subplots(len(files),1,figsize=(6, 6))
fig.tight_layout()

for i in range(len(files)):
    file = open(files[i],"r")
    ys = file.readlines()[0]
    ys = ys.split("\t")
    ys.pop()
    ysaux = []
    for j in range(0):
        ysaux.append(float(ys[j]))
    ys = ysaux
    axs[i].plot(range(len(ys)),ys)
    axs[i].set_title(titles[i])

plt.show()