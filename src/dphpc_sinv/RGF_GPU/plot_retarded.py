# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})


    images_path = "/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/RGF_GPU/images/"
    results_path = "/usr/scratch/mont-fort23/almaeder/rgf_times/"


    # int n_blocks_input[nb_test] = {3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512};
    # int blocksize_input[bs_test] = {64, 128, 256, 512};

    block_sizes = [64, 128, 256, 512]
    warmup = 2
    measurements = 20
    number_of_blockss = [3*16, 3*32, 3*64, 3*128, 3*256, 3*512]
    number_of_blockss = [3*16, 3*32, 3*64, 3*128, 3*256, 3*512]
    for blocksize in block_sizes:
        times_retarded_fits_gpu_memory = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_fits_gpu_memory_with_copy_compute_overlap = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_does_not_fit_gpu_memory = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times = [times_retarded_fits_gpu_memory, times_retarded_fits_gpu_memory_with_copy_compute_overlap, times_retarded_does_not_fit_gpu_memory, times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap]
        for i, number_of_blocks in enumerate(number_of_blockss):
            matrix_size = number_of_blocks * blocksize
            times[0][i,:] = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[1][i,:] = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_with_copy_compute_overlap_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[2][i,:] = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[3][i,:] = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]

        times = np.array([times_retarded_fits_gpu_memory, times_retarded_fits_gpu_memory_with_copy_compute_overlap, times_retarded_does_not_fit_gpu_memory, times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap])
        

        stds = []
        medians = []
        interval = []
        confidence = 0.95
        for i in range(4):
            stds.append(np.std(times[i], axis=1))
            medians.append(np.median(times[i], axis=1))
            interval.append(np.empty((2, len(number_of_blockss))))
            for j in range(len(number_of_blockss)):
                interval[i][:,j] = st.t.interval(confidence=confidence, df=len(times[i][j])-1,
                    loc=np.median(times[i][j]),
                    scale=st.sem(times[i][j]))
                # print("interval: ", interval[i][:,j])
                # print("median: ", medians[i][j])

        yer_fft = []
        for i in range(4):
            yer_fft.append(np.copy(interval[i]))
            for j in range(times[i].shape[0]):
                yer_fft[i][0,j] = -yer_fft[i][0,j] + medians[i][j]
                yer_fft[i][1,j] = yer_fft[i][1,j] - medians[i][j]
        # if(blocksize == 512):
        #     print("times_retarded_fits_gpu_memory: ", times_retarded_fits_gpu_memory)
        


        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        labels = [
            "Whole Matrix fits",
            "Whole Matrix fits with overlap",
            "Few Block fit",
            "Few Block fit with overlap"
        ]
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red"
        ]
        for i in range(4):
            x = np.array(number_of_blockss)
            cond = np.where(medians[i] > 1e-5)
            # cond = np.logical_and(np.where(medians[i] > 1e-5), np.isnan(interval[i][0]) == False)
            x = x[cond]
            med = np.array(medians[i])[cond]
            inter_low = np.array(interval[i][0])[cond]
            inter_high = np.array(interval[i][1])[cond]
            ax.plot(x, med, label=labels[i], color=colors[i], linestyle='dashed')
            # ax.fill_between(x, inter_low, inter_high, alpha=0.2, color=colors[i])
            plt.errorbar(x, med, yerr=np.squeeze(yer_fft[i][:,cond]), color=colors[i], capsize=5, barsabove=True, marker='x', linestyle='None')
        # ax.boxplot(times.T, labels=labels, showfliers=False)
        # ax.set_yscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(
            "RGF with "+ str(blocksize) +" blocksize on GPU")
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("Numbers of Blocks")
        # ax.set_xticks(number_of_blockss)
        ax.set_xscale("log", base=2)
        ax.set_xticks(number_of_blockss, minor=False)
        ax.set_xticklabels(number_of_blockss, minor=False)
        #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.legend()
        plt.savefig(images_path + "lineplot_retarded_"+ str(blocksize) +".png", bbox_inches='tight', dpi=300)
