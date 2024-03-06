import matplotlib.pyplot as plt


fig, axs = plt.subplots(3,1,figsize=(6, 6))
fig.tight_layout()
folder = "errors/"
files = ["avs.txt","sigmas.txt","maxs.txt"]
files = [folder+f for f in files]
titles = ['Mean error',"Error variance", "Maximum error"]


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