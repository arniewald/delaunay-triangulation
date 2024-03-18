import matplotlib.pyplot as plt

folder = "Delaunay/errors/"
files = ["avs","sigmas","maxs","evars","rerrs"]
sufix = 'Iris4D_binarystart_errw0.txt'
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
    for y in ys:
        ysaux.append(float(y))
    ys = ysaux
    axs[i].plot(range(len(ys)),ys)
    axs[i].set_title(titles[i])

plt.show()