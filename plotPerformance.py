import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.font_manager
plt.rcParams['font.family'] = 'Britannic Bold'
# plt.rcParams['font.serif'] = ['Times New Roman']

labelFontsize = 18
tickFontsize = 15

def plot():
    
    matsizes = np.array([10,100,1000,10000,100000,1000000],dtype=int)
    kernels = ["CPU", "Red0", "Red1", "Red2"]

    Ts = np.zeros((len(kernels),len(matsizes)))
    CPUt = [0.21, 0.35, 1.36, 11.8, 90, 1000]
    Ts[0,:] = CPUt

    NT = 256
    for i in range(len(kernels[1:])):
        for m in range(len(matsizes)):
            data = np.loadtxt(f"KernelTimings/reduce{i}_N{matsizes[m]}_repeat11_NT{NT}.csv",delimiter=",")
            kt = np.mean(data[1:])
            Ts[i+1,m] = kt
    
    # print(np.round(Ts[1,:]/Ts[2,:],2))
    # print(np.round(Ts[1,:]/Ts[3,:],2))
    # exit()

    x = np.arange(len(kernels))  # the label locations
    width = 0.35  # the width of the bars

    #colors = ["r","g","b","cyan","orange","magenta"]
    colors = plt.cm.cool(np.linspace(0,1,len(matsizes)))
    f,ax = plt.subplots(1,1,figsize=(10,5))
    
    gap = 0
    for i in range(len(kernels)):
        x = np.arange(len(matsizes)) + gap
        y = Ts[i,:]
        ax.bar(x,y, width=1, ec="w", lw=1, color=colors,alpha=0.99)
        gap += len(matsizes)+1

        if i > 1:
            scaleup = np.round(Ts[1,:]/Ts[i,:],1)
            ax.text(x[-1],y[-1]*1.2,f"{scaleup[-1]}x", fontsize=tickFontsize*0.8, ha="center")
            ax.text(x[-2],y[-2]*1.2,f"{scaleup[-2]}x", fontsize=tickFontsize*0.8, ha="center")
            ax.text(x[-3],y[-3]*1.2,f"{scaleup[-3]}x", fontsize=tickFontsize*0.8, ha="center")
    
    ax.set_xticks([2.5,9,16.5,23],kernels)
    ax.tick_params(labelsize=tickFontsize)
    
    ax.set_ylabel("t (microsecs)",fontsize=labelFontsize)
    ax.set_yscale("log")

    legend_boxes = [mpatches.Patch(color=colors[i], label=f"N = {matsizes[i]}") for i in range(len(matsizes))]
    ax.legend(frameon=False, handles=legend_boxes, ncol=2, fontsize=tickFontsize*0.9)

    plt.tight_layout()
    plt.savefig("RedPerform.png",dpi=300)
    plt.close()


if __name__ == "__main__":

    plot()