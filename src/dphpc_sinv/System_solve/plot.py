import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter, FixedFormatter, FixedLocator

if __name__ == "__main__":
    times = np.loadtxt("/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/System_solve/times.txt", dtype=np.float64)[:, 10:]


    stds = np.std(times, axis=1)
    means = np.mean(times, axis=1)
    confidence = 0.95
    interval = np.empty((2, times.shape[0]))
    for i in range(times.shape[0]):
        interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                       loc=np.mean(times[i]),
                                       scale=st.sem(times[i]))

    num_bars = 7

    yer = [interval[:, i].reshape((2,-1)) for i in range(num_bars)]
    for i in range(times.shape[0]):
        yer[i][0] = -yer[i][0] + means[i]
        yer[i][1] = yer[i][1] - means[i]



    # width of the bars
    barWidth = 0.8
    position0 = np.arange(int(times.shape[0]))
    position1 = position0 + barWidth

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 12)

    # Create blue bars
    ax.bar(position1[0], means[0], width=barWidth, color="blue",
            edgecolor="black", yerr=yer[0], capsize=7, label="MKL gesv (14 Threads)")

    # Create cyan bars
    ax.bar(position1[1], means[1], width=barWidth, color="cyan",
            edgecolor="black", yerr=yer[1], capsize=7, label="MKL gbsv (14 Threads)")

    # Create olive bars
    ax.bar(position1[2], means[2], width=barWidth, color="olive",
            edgecolor="black", yerr=yer[2], capsize=7, label="MKL pbsv (14 Threads)")

    # Create blue bars
    ax.bar(position1[3], means[3], width=barWidth, color="red",
            edgecolor="black", yerr=yer[3], capsize=7, label="CuSparse CG (1843 Steps)")

    # Create cyan bars
    ax.bar(position1[4], means[4], width=barWidth, color="green",
            edgecolor="black", yerr=yer[4], capsize=7, label="ILU + CuSparse CG (43 Steps)")

    # Create cyan bars
    ax.bar(position1[5], means[5], width=barWidth, color="orange",
            edgecolor="black", yerr=yer[5], capsize=7, label="CuSolver dense")

    # Create cyan bars
    ax.bar(position1[6], means[6], width=barWidth, color="purple",
            edgecolor="black", yerr=yer[6], capsize=7, label="CuSolver sparse Cholesky")



    # general layout
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off


    # ax.get_yaxis().set_minor_locator(FixedLocator([0.1, 0.2, 0.3, 0.5]))
    # ax.get_yaxis().set_minor_formatter(FixedFormatter(["a", "b", "c", "d"]))
    # ax.tick_params(
    #     axis='y',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False)

    # ax.yaxis.set_ticks([0.05, 0.1, 0.2, 0.5], minor=False)
    # ax.yaxis.set_ticklabels(["a", "b", "c", "d"], minor=False)

    ax.set_title("Small KMC System Solve Benchmark on Attelas \n (7165x7165, 1e-14 tolerance)")
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Methods")
    ax.legend()
    
    plt.savefig("plot.png",bbox_inches='tight', dpi=300)
