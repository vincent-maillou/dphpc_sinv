# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib  

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 36})


    images_path = "/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/RGF_GPU/images/"
    results_path = "/usr/scratch/mont-fort23/almaeder/rgf_times/"


    # int n_blocks_input[nb_test] = {3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512};
    # int blocksize_input[bs_test] = {64, 128, 256, 512};

    block_sizes = [256]
    warmup = 2
    measurements = 20
    number_of_blockss = [3*16, 3*32, 3*64, 3*128, 3*256, 3*512]

    flops_gemm = 1.075
    flops_LU = 8/3 * 1.075
    flops_rgf = np.array([(number_of_blocks*flops_LU + 7*(number_of_blocks-1)*flops_gemm) for number_of_blocks in number_of_blockss])/1000*block_sizes[0]**3/512**3

    #number_of_blockss = [3*4, 3*8, 3*16, 3*32, 3*64, 3*128, 3*256, 3*512]
    for blocksize in block_sizes:
        times_retarded_fits_gpu_memory = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_fits_gpu_memory_with_copy_compute_overlap = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_does_not_fit_gpu_memory = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times_cpu = np.zeros([len(number_of_blockss), measurements], dtype=np.float64)
        times = [times_retarded_fits_gpu_memory, times_retarded_fits_gpu_memory_with_copy_compute_overlap, times_retarded_does_not_fit_gpu_memory, times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap]
        for i, number_of_blocks in enumerate(number_of_blockss):
            matrix_size = number_of_blocks * blocksize
            times[0][i,:] = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[1][i,:] = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_with_copy_compute_overlap_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[2][i,:] = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            times[3][i,:] = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
            
            times[0][i,:] = flops_rgf[i]/times[0][i,:]
            times[1][i,:] = flops_rgf[i]/times[1][i,:]
            times[2][i,:] = flops_rgf[i]/times[2][i,:]
            times[3][i,:] = flops_rgf[i]/times[3][i,:]
            
            times_cpu[i,:] = np.loadtxt(results_path + "times_retarded_cpu_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(blocksize) + ".txt", dtype=np.float64)[warmup:]
        times = np.array([times_retarded_fits_gpu_memory, times_retarded_fits_gpu_memory_with_copy_compute_overlap, times_retarded_does_not_fit_gpu_memory, times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap, times_cpu])
        
        tis = 4
        stds = []
        medians = []
        interval = []
        confidence = 0.95
        for i in range(tis):
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
        for i in range(tis):
            yer_fft.append(np.copy(interval[i]))
            for j in range(times[i].shape[0]):
                yer_fft[i][0,j] = -yer_fft[i][0,j] + medians[i][j]
                yer_fft[i][1,j] = yer_fft[i][1,j] - medians[i][j]
        # if(blocksize == 512):
        #     print("times_retarded_fits_gpu_memory: ", times_retarded_fits_gpu_memory)
        


        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        labels = [
            "Method 1",
            "Method 2",
            "Method 3",
            "Method 4",
            "Reference CPU"
        ]
        labels = labels[:tis]
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple"
        ]
        colors = colors[:tis]
        for i in range(tis):
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
        # if blocksize == 64:
        #     ax.set_yticks([1e-2, 1e-1, 1e0], minor=False)
        # elif blocksize == 128:
        #     ax.set_yticks([1e-1, 1e0], minor=False)
        # elif blocksize == 256:
        #     ax.set_yticks([1e-1, 1e0, 1e1], minor=False)
        # elif blocksize == 512:
        #     ax.set_yticks([1e0, 1e1], minor=False)
        ax.get_yaxis().get_major_formatter().labelOnlyBase = False
        #ax.set_ylim(bottom=0)
        # ax.set_title(
        #     "RGF with "+ str(blocksize) +" blocksize")
        ax.set_ylabel("TFLOP/s")
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=0.5)
        ax.set_xlabel("Numbers of Blocks")
        # ax.set_xticks(number_of_blockss)
        ax.set_xscale("log", base=2)
        ax.set_xticks(number_of_blockss, minor=False)
        ax.set_xticklabels(number_of_blockss, minor=False)
        #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.legend(loc='center right')
        plt.savefig(images_path + "lineplot_"+ str(blocksize) +".eps", bbox_inches='tight', dpi=300, format="eps")
        plt.savefig(images_path + "lineplot_"+ str(blocksize) +".png", bbox_inches='tight', dpi=300, format="png")
