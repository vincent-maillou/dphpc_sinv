import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

if __name__ == "__main__":
    times = np.loadtxt("/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/System_solve/times.txt", dtype=np.float64)[:, 1:]

    stds = np.std(times, axis=1)
    means = np.mean(times, axis=1)
    confidence = 0.95
    interval = np.empty((2, times.shape[0]))
    for i in range(times.shape[0]):
        interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                       loc=np.mean(times[i]),
                                       scale=st.sem(times[i]))

    bars0 = means[0]
    bars1 = means[1]
    bars2 = means[2]
    bars3 = means[3]
    bars4 = means[4]
    bars5 = means[5]

    yer0 = interval[:, 0].reshape((2,-1))
    yer1 = interval[:, 1].reshape((2,-1))
    yer2 = interval[:, 2].reshape((2,-1))
    yer3 = interval[:, 3].reshape((2,-1))
    yer4 = interval[:, 4].reshape((2,-1))
    yer5 = interval[:, 5].reshape((2,-1))
    yer0[yer0 < 0] = 0
    yer1[yer1 < 0] = 0
    yer2[yer2 < 0] = 0
    yer3[yer3 < 0] = 0
    yer4[yer4 < 0] = 0
    yer5[yer5 < 0] = 0
    # width of the bars
    barWidth = 0.5
    position0 = np.arange(int(times.shape[0]/1))
    position1 = position0 + barWidth


    # Create blue bars
    plt.bar(position1[0], bars0, width=barWidth, color="blue",
            edgecolor="black", yerr=yer0, capsize=7, label="MKL gesv")

    # Create cyan bars
    plt.bar(position1[1], bars1, width=barWidth, color="cyan",
            edgecolor="black", yerr=yer1, capsize=7, label="MKL gbsv")

    # Create blue bars
    plt.bar(position1[2], bars2, width=barWidth, color="red",
            edgecolor="black", yerr=yer2, capsize=7, label="CuSparse CG (1600 Steps)")

    # Create cyan bars
    plt.bar(position1[3], bars3, width=barWidth, color="green",
            edgecolor="black", yerr=yer3, capsize=7, label="ILU + CuSparse CG (39 Steps)")

    # Create cyan bars
    plt.bar(position1[4], bars4, width=barWidth, color="orange",
            edgecolor="black", yerr=yer4, capsize=7, label="CuSolver dense")

    # Create cyan bars
    plt.bar(position1[5], bars5, width=barWidth, color="purple",
            edgecolor="black", yerr=yer5, capsize=7, label="CuSolver sparse Cholesky")

    # general layout
    plt.title("Small KMC system solve benchmark")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.ylabel("Time [s]")
    plt.xlabel("Methods")
    plt.legend()

    plt.yscale("log")

    # save plot
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.savefig("plot.png")
