import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})


    base_path = ""
    images_path = base_path + ""
    results_path = base_path + ""

    matrix_sizes = [5408]
    block_sizes = [416]
    batch_sizes = [112]
    number_of_blocks = [13]



    for i in range(len(matrix_sizes)):
        warmup = 10
        times_for = np.loadtxt(results_path + "times_for_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]
        times_batched = np.loadtxt(results_path + "times_batched_"+ str(matrix_sizes[i]) + "_" + str(number_of_blocks[i]) + "_" + str(batch_sizes[i]) + "_.txt", dtype=np.float64)[warmup:]

        times = np.array([times_for, times_batched])

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)
        labels = [
            "For loop",
            "Batched"
        ]
        ax.boxplot(times.T, labels=labels, showfliers=False)
        # ax.set_yscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(
            "RGF for "+ str(matrix_sizes[i]) +"x"+ str(matrix_sizes[i]) +" system matrix  \n with "+ str(block_sizes[i]) +"x"+ str(block_sizes[i]) +" blocks and "+str(batch_sizes[i])+" batchsize")
        ax.set_ylabel("Time [s]")
        plt.savefig(images_path + "boxplot_"+ str(matrix_sizes[i]) +".png", bbox_inches='tight', dpi=300)
