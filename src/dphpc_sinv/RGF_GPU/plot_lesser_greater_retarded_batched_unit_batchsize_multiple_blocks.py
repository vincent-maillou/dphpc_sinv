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


    block_sizes = [64, 128, 256, 512, 1024]
    number_of_blockss = [2, 4, 6, 8]
    batchsizes =[1]


    for batchsize in batchsizes:
        

        times_batched = np.zeros([len(block_sizes), measurements], dtype=np.float64)
        times_for = np.zeros([len(block_sizes), measurements], dtype=np.float64)
        times_memcpy = np.zeros([len(block_sizes), measurements], dtype=np.float64)

        times_for = [np.zeros([len(block_sizes), measurements], dtype=np.float64) for _ in range(len(number_of_blockss))]
        times_batched = [np.zeros([len(block_sizes), measurements], dtype=np.float64) for _ in range(len(number_of_blockss))]
        times_memcpy = [np.zeros([len(block_sizes), measurements], dtype=np.float64) for _ in range(len(number_of_blockss))]
        for j,number_of_blocks in enumerate(number_of_blockss):
            for i, blocksize in enumerate(block_sizes):
                matrix_size = number_of_blocks * blocksize
                times_for[j][i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_for_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]

            median_for = np.median(times_for[j], axis=1)
            for i, blocksize in enumerate(block_sizes):
                matrix_size = number_of_blocks * blocksize
                times_batched[j][i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_batched_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
                times_batched[j][i,:] = median_for[i]/times_batched[j][i,:]
                times_memcpy[j][i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_memcpy_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
                times_memcpy[j][i,:] = median_for[i]/times_memcpy[j][i,:]
        times = np.array([times_memcpy, times_batched]).reshape((2*len(number_of_blockss), len(block_sizes), measurements))
            
        tis = len(number_of_blockss)*2
        stds = []
        medians = []
        interval = []
        confidence = 0.95
        for i in range(tis):
            stds.append(np.std(times[i], axis=1))
            medians.append(np.median(times[i], axis=1))
            interval.append(np.empty((2, len(block_sizes))))
            for j in range(len(block_sizes)):
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
        fig.set_size_inches(20, 14)
        labels = [
            "Method 2",
            "Method 3"
        ]
        labels = labels[:tis]
        labels = []
        for i in range(len(number_of_blockss)-1):
            labels.append(None)
        labels.append("Method 2")
        for number_of_blocks in number_of_blockss:
            labels.append("Method 3: " + str(number_of_blocks) + " Blocks")
        colors2 = [
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
        colors = []
        for i in range(len(number_of_blockss)):
            colors.append("tab:orange")
        for i in range(len(number_of_blockss)):
            colors.append(colors2[i])

        colors = colors[:tis]
        for i in range(tis):
            x = np.array(block_sizes)
            cond = np.where(medians[i] > 1e-5)
            # cond = np.logical_and(np.where(medians[i] > 1e-5), np.isnan(interval[i][0]) == False)
            x = x[cond]
            med = np.array(medians[i])[cond]
            inter_low = np.array(interval[i][0])[cond]
            inter_high = np.array(interval[i][1])[cond]
            ax.plot(x, med, label=labels[i], color=colors[i], linestyle='dashed', linewidth=3)
            plt.errorbar(x, med, yerr=np.squeeze(yer_fft[i][:,cond]), color=colors[i], capsize=10, barsabove=True, marker='x', linestyle='None', linewidth=3)
        
        ax.set_ylabel("Speed-up")
        ax.set_xlabel("Blocksize")
        ax.set_ylim(bottom=0)
        # ax.set_xticks(number_of_blockss)
        ax.set_xscale("log", base=2)
        ax.set_xticks(block_sizes, minor=False)
        ax.set_xticklabels(block_sizes, minor=False)
        #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.legend()
        plt.savefig(images_path + "speedup_single_batch_mb_lgr_"+ str(batchsize) + ".png", bbox_inches='tight', dpi=300, format="png")
        plt.close(fig)
