import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter, FixedFormatter, FixedLocator

if __name__ == "__main__":
    warmup = 10
    times = np.loadtxt("/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/System_solve/times.txt", dtype=np.float64)[:, warmup:]

    stds = np.std(times, axis=1)
    means = np.median(times, axis=1)
    confidence = 0.95
    interval = np.empty((2, times.shape[0]))
    for i in range(times.shape[0]):
        interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                       loc=np.median(times[i]),
                                       scale=st.sem(times[i]))


    yer = [interval[:, i].reshape((2,-1)) for i in range(times.shape[0])]
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

    colors =["blue", "cyan", "olive", "red", "green", "orange", "purple", "gray", "brown"]

    labels =[
        "MKL gesv (14 Threads)",
        "MKL posv (14 Threads)",
        "MKL gbsv (14 Threads)",
        "MKL pbsv (14 Threads)",
        "CuSparse CG (1843 Steps)",
        "ILU + CuSparse CG (43 Steps)",
        "CuSolver dense LU",
        "CuSolver dense Cholesky",
        "CuSolver sparse Cholesky"
    ]

    for i in range(times.shape[0]):
        ax.bar(position0[i], means[i], width=barWidth, color=colors[i],
                edgecolor="black", yerr=yer[i], capsize=7, label=labels[i])

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
