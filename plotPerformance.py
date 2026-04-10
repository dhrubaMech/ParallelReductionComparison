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

def plot1MIL():
    NT = 256
    input = 1000000
    kernels = ["Interleaved\nV1", "Interleaved\nV2", "Sequential\nAddressing", 
               "First Add\nDuring Load", "Unroll Last\nWarp", "Complete\nUnroll"]

    Ts = np.zeros(len(kernels))
    for i in range(len(kernels)):
        data = np.loadtxt(f"KernelTimings/reduce{i}_N{input}_repeat11_NT{NT}.csv",delimiter=",")
        Ts[i] = np.mean(data[1:])
    
    # colors = plt.cm.cool(np.linspace(0,1,len(kernels)))
    colors = plt.cm.cool(np.linspace(0,1,len(kernels)))
    plt.figure(figsize=(7,5))

    plt.bar(np.arange(len(kernels)),Ts, width=1, ec="w", lw=3, color=colors,alpha=0.99)

    for i in range(len(kernels)):
        if i == 0:
            plt.text(i,Ts[i]+1,f"Naive",fontsize=tickFontsize, ha="center")
        else:
            plt.text(i,Ts[i]+1,f"{np.round(Ts[0]/Ts[i],2)}x",fontsize=tickFontsize, ha="center")

    plt.xlabel("KERNELS",fontsize=labelFontsize,labelpad=10)
    plt.ylabel("t (microsecs)",fontsize=labelFontsize)

    plt.ylim([0,60])
    plt.xticks(np.arange(len(kernels)),kernels)
    plt.tick_params(axis="y",labelsize=tickFontsize)
    plt.tick_params(axis="x",labelsize=tickFontsize*0.75)

    plt.tight_layout()
    plt.savefig("compareKernelsN1Mil.png",dpi=300)
    plt.close()


def plotThreadLaunched():
    NT = 256
    input = 1000000
    kernels = ["Interleaved\nV1", "Interleaved\nV2", "Sequential\nAddressing", 
               "First Add\nDuring Load", "Unroll Last\nWarp", "Complete\nUnroll"]
    Threads = np.array([64,128,256,512])

    Ts = np.zeros((len(kernels),len(Threads)))
    for i in range(len(kernels)):
        for t in range(len(Threads)):
            data = np.loadtxt(f"KernelTimings/reduce{i}_N{input}_repeat25_NT{Threads[t]}.csv",delimiter=",")
            Ts[i,t] = np.mean(data[1:])
    
    # colors = plt.cm.cool(np.linspace(0,1,len(kernels)))
    colors = plt.cm.cool(np.linspace(0,1,len(kernels)))
    plt.figure(figsize=(5,5))

    xx,yy = np.meshgrid(Threads,np.arange(len(kernels)))
    plt.imshow(Ts,cmap="cool_r",origin="lower")

    for i in range(len(kernels)):
        for j in range(len(Threads)):
            plt.text(j,i,np.round(Ts[i,j],2),ha="center",c="w")

    plt.xlabel("THREADS",fontsize=labelFontsize,labelpad=10)
    plt.ylabel("KERNELS",fontsize=labelFontsize)

    plt.xticks(np.arange(len(Threads)),Threads)
    plt.yticks(np.arange(len(kernels)),kernels[:])
    #plt.tick_params(axis="y",labelsize=tickFontsize)
    #plt.tick_params(axis="x",labelsize=tickFontsize*0.65)

    plt.tight_layout()
    plt.savefig("EffectThreads.png",dpi=300)
    plt.close()



if __name__ == "__main__":

    # plot()

    plot1MIL()

    plotThreadLaunched()