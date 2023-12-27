# Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})


    base_path = ""
    images_path = base_path + ""
    results_path = base_path + "/usr/scratch/mont-fort17/almaeder/rgf_bench/times/"

    block_sizes = [512]
    number_of_blocks = [45]
    matrix_sizes = [block_sizes[i]*number_of_blocks[i] for i in range(len(block_sizes))]
    batch_sizes = [1]

    for i in range(len(matrix_sizes)):
        warmup = 10
        times_retarded_fits_gpu_memory = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]
        times_retarded_fits_gpu_memory_with_copy_compute_overlap = np.loadtxt(results_path + "times_retarded_fits_gpu_memory_with_copy_compute_overlap_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]
        times_retarded_does_not_fit_gpu_memory = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]
        times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap = np.loadtxt(results_path + "times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]

        times = np.array([times_retarded_fits_gpu_memory, times_retarded_fits_gpu_memory_with_copy_compute_overlap, times_retarded_does_not_fit_gpu_memory, times_retarded_does_not_fit_gpu_memory_with_copy_compute_overlap])

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        labels = [
            "Fits",
            "Fits with overlap",
            "Does not fit",
            "Does not fit with overlap"
        ]
        ax.boxplot(times.T, labels=labels, showfliers=False)
        # ax.set_yscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(
            "RGF Retarded for "+ str(matrix_sizes[i]) +"x"+ str(matrix_sizes[i]) +" system matrix  \n with "+ str(block_sizes[i]) +"x"+ str(block_sizes[i]) +" blocks and "+str(batch_sizes[i])+" batchsize")
        ax.set_ylabel("Time [s]")
        plt.savefig(images_path + "boxplot_retarded_"+ str(matrix_sizes[i]) +".png", bbox_inches='tight', dpi=300)
