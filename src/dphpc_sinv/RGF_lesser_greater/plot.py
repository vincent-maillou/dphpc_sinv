import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})


    base_path = ""
    images_path = base_path + ""
    results_path = base_path + ""

    warmup = 10
    times_for = np.loadtxt(results_path + "times_for.txt", dtype=np.float64)[warmup:]
    times_batched = np.loadtxt(results_path + "times_batched.txt", dtype=np.float64)[warmup:]
    times_batched2 = np.loadtxt(results_path + "times_batched2.txt", dtype=np.float64)[warmup:]
    times_batched_optimized = np.loadtxt(results_path + "times_batched_optimized.txt", dtype=np.float64)[warmup:]

    times = np.array([times_for, times_batched, times_batched2, times_batched_optimized])

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    labels = [
        "For loop",
        "Batched",
        "Batched \n w.o. cudaMalloHost",
        "Batched Fused <> \n w.o. cudaMalloHost",
    ]
    ax.boxplot(times.T, labels=labels, showfliers=False)
    # ax.set_yscale("log")
    ax.set_title(
        "RGF for 1000x1000 system matrix  \n with 100x100 blocksize and 100 batchsize")
    ax.set_ylabel("Time [s]")
    plt.savefig(images_path + "boxplot_1000.png", bbox_inches='tight', dpi=300)
