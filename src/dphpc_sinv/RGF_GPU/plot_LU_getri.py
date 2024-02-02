# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 36})


    images_path = "/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/RGF_GPU/images2/"
    results_path = "/usr/scratch/mont-fort23/almaeder/rgf_times_batched/"


    # int n_blocks_input[nb_test] = {3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512};
    # int blocksize_input[bs_test] = {64, 128, 256, 512};


    warmup = 2
    measurements = 20

    # int batch_sizes_input[nbatch_test] = {1, 2, 4, 8, 16, 32, 64, 128};
    # int n_blocks_input[nb_test] = {2, 4, 6, 8, 10, 12, 14};
    # int blocksize_input[bs_test] = {32, 64, 128, 256, 512, 1024};


    block_sizes = [64, 128, 256, 512, 768, 1024]
    batchsizes =[1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    batchsizes2 =[1, 2, 4, 8, 16, 32, 64, 128]

    
        

    times_batched = [np.zeros([len(batchsizes), measurements], dtype=np.float64) for _ in range(len(block_sizes))]
    times_for = [np.zeros([len(batchsizes), measurements], dtype=np.float64) for _ in range(len(block_sizes))]

    for j, blocksize in enumerate(block_sizes):
        for i, batchsize in enumerate(batchsizes):
        
            times_for[j][i,:] = np.loadtxt(results_path + "times_inv_for_"+ str(blocksize) +  "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]

        median_for = np.median(times_for[j], axis=1)
        for i, batchsize in enumerate(batchsizes):
            times_batched[j][i,:] = np.loadtxt(results_path + "times_inv_batched_"+ str(blocksize) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
            times_batched[j][i,:] = median_for[i]/times_batched[j][i,:]
    times = np.array([times_batched]).reshape((len(block_sizes), len(batchsizes), measurements))
            
    tis = len(block_sizes)
    stds = []
    medians = []
    interval = []
    confidence = 0.95
    for i in range(tis):
        stds.append(np.std(times[i], axis=1))
        medians.append(np.median(times[i], axis=1))
        interval.append(np.empty((2, len(batchsizes))))
        for j in range(len(batchsizes)):
            interval[i][:,j] = st.t.interval(confidence=confidence, df=len(times[i][j])-1,
                loc=np.median(times[i][j]),
                scale=st.sem(times[i][j]))

    yer_fft = []
    for i in range(tis):
        yer_fft.append(np.copy(interval[i]))
        for j in range(times[i].shape[0]):
            yer_fft[i][0,j] = -yer_fft[i][0,j] + medians[i][j]
            yer_fft[i][1,j] = yer_fft[i][1,j] - medians[i][j]
    


    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    labels = []
    for i in range(len(block_sizes)):
        labels.append("Blocksize: " + str(block_sizes[i]))

    labels = labels[:tis]
    colors = [
        "tab:green",
        "tab:blue",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan"
    ]
    colors = colors[:tis]
    for i in range(tis):
        x = np.array(batchsizes)
        cond = np.where(medians[i] > 1e-5)
        # cond = np.logical_and(np.where(medians[i] > 1e-5), np.isnan(interval[i][0]) == False)
        x = x[cond]
        med = np.array(medians[i])[cond]
        inter_low = np.array(interval[i][0])[cond]
        inter_high = np.array(interval[i][1])[cond]
        ax.plot(x, med, label=labels[i], color=colors[i], linestyle='dashed', linewidth=3)
        plt.errorbar(x, med, yerr=np.squeeze(yer_fft[i][:,cond]), color=colors[i], capsize=10, barsabove=True, marker='x', linestyle='None', linewidth=3)
    
    # plot dashed horizontal line at y=1
    ax.axhline(y=1, color='black', linestyle='dashed', linewidth=3)


    ax.set_ylabel("Speed-up")
    ax.set_xlabel("Batchsize")
    # ax.set_ylim(bottom=0)
    ax.set_yscale("log", base=10)
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2], minor=False)
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().get_major_formatter().labelOnlyBase = False


    # ax.set_xticks(number_of_blockss)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batchsizes2, minor=False)
    ax.set_xticklabels(batchsizes2, minor=False)
    #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.legend(fontsize="20")
    plt.savefig(images_path + "speedup_lu_getri.png", bbox_inches='tight', dpi=300, format="png")
    plt.close(fig)
