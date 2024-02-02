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


    block_sizes = [64, 128, 256]
    number_of_blockss = [2, 4, 6, 8, 10, 12, 14]
    batchsizes =[1, 2, 4, 8, 16, 32, 64, 128]

    block_sizes = [64, 128, 256, 512]
    number_of_blockss = [2, 4, 6, 8, 10, 12, 14]
    batchsizes =[1, 2, 4, 8, 16, 32, 64]

    block_sizes = [64, 128, 256, 512, 1024]
    number_of_blockss = [2, 4, 6, 8, 10, 12, 14]
    batchsizes =[1, 2, 4, 8, 16]


    flops_gemm = 1.075
    flops_LU = 8/3 * 1.075
    # flops_rgf = batchsize*np.array([(number_of_blocks*flops_LU + 7*(number_of_blocks-1)*flops_gemm)*bs**3/512**3 for bs in block_sizes])
    for batchsize in batchsizes:
        for number_of_blocks in number_of_blockss:
            flops_rgf = batchsize/1000*np.array([(number_of_blocks*flops_LU + (4+33*(number_of_blocks-1) )*flops_gemm)*bs**3/512**3 for bs in block_sizes])


            times_batched = np.zeros([len(block_sizes), measurements], dtype=np.float64)
            times_for = np.zeros([len(block_sizes), measurements], dtype=np.float64)
            times_memcpy = np.zeros([len(block_sizes), measurements], dtype=np.float64)
            for i, blocksize in enumerate(block_sizes):
                matrix_size = number_of_blocks * blocksize
                times_for[i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_for_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
                tmp = np.copy(times_for[i,:])
                # times_for[i,:] /= tmp  
                times_for[i,:] = flops_rgf[i]/times_for[i,:]         
                times_batched[i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_batched_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
                # times_batched[i,:] /= tmp
                # times_batched[i,:] = 1/times_batched[i,:]
                times_batched[i,:] = flops_rgf[i]/times_batched[i,:]

                times_memcpy[i,:] = np.loadtxt(results_path + "times_lesser_greater_retarded_memcpy_"+ str(matrix_size) + "_" + str(number_of_blocks) + "_" + str(batchsize) + ".txt", dtype=np.float64)[warmup:]
                times_memcpy[i,:] = flops_rgf[i]/times_memcpy[i,:]
            times = np.array([times_for, times_memcpy, times_batched])
            
            tis = 3
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
                "Method 3"
            ]
            labels = labels[:tis]
            colors = [
                "tab:blue",
                "tab:orange",
                "tab:green",
            ]
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
                # ax.fill_between(x, inter_low, inter_high, alpha=0.2, color=colors[i])
                plt.errorbar(x, med, yerr=np.squeeze(yer_fft[i][:,cond]), color=colors[i], capsize=10, barsabove=True, marker='x', linestyle='None', linewidth=3)
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
            # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
            #ax.set_ylim(bottom=0)
            # ax.set_title(
            #     "RGF with "+ str(number_of_blocks) +" Blocks \n and Batchsize " + str(batchsize))
            ax.set_ylabel("TFLOP/s")
            ax.set_xlabel("Blocksize")
            # ax.set_xticks(number_of_blockss)
            ax.set_xscale("log", base=2)
            ax.set_xticks(block_sizes, minor=False)
            ax.set_xticklabels(block_sizes, minor=False)
            #ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax.legend()
            # plt.savefig(images_path + "scaling_batched_lgr_"+ str(number_of_blocks) + "_"+ str(batchsize) + "_"+ str(batchsize)    +".eps", bbox_inches='tight', dpi=300, format="eps")
            plt.savefig(images_path + "scaling_batched_lgr_"+ str(number_of_blocks) + "_"+ str(batchsize) + ".png", bbox_inches='tight', dpi=300, format="png")
            plt.close(fig)
