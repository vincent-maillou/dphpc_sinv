import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter, FixedFormatter, FixedLocator

if __name__ == "__main__":

    measurements = 8


    times_tot = []
    steps_tot = []

    for step in range(measurements):
        warmup = 10
        times = np.loadtxt(
            "/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/System_solve/times"+ str(step) +".txt", dtype=np.float64)[:,warmup:]

    



        steps = np.loadtxt(
            "/home/sem23f28/Documents/dphpc_sinv/src/dphpc_sinv/System_solve/steps"+ str(step) +".txt", dtype=np.int32)

        if step != 0 and step != 2:
            times_tot.append(times)
            steps_tot.append(steps)

        stds = np.std(times, axis=1)
        means = np.median(times, axis=1)
        confidence = 0.95
        interval = np.empty((2, times.shape[0]))
        for i in range(times.shape[0]):
            interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                        loc=np.median(times[i]),
                                        scale=st.sem(times[i]))

        yer = [interval[:, i].reshape((2, -1)) for i in range(times.shape[0])]
        for i in range(times.shape[0]):
            yer[i][0] = -yer[i][0] + means[i]
            yer[i][1] = yer[i][1] - means[i]

        # width of the bars
        barWidth = 0.8
        position0 = np.arange(int(times.shape[0]))
        position1 = position0 + barWidth

        plt.rcParams.update({'font.size': 24})
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 9)

        colors = ["blue", "cyan", "olive", "red",
                "green", "orange", "purple", "gray", "brown", "black", "pink", "yellow"]

        labels = [
            "MKL gesv (14 Threads)",
            "MKL posv (14 Threads)",
            "MKL gbsv (14 Threads)",
            "MKL pbsv (14 Threads)",
            "CuSparse CG ("+str(steps[0]) +" Steps)",
            "CG with Guess ("+str(steps[1]) +" Steps)",
            "Precon. CG ("+str(steps[2]) +" Steps)",
            "Precon. CG with Guess ("+str(steps[3]) +" Steps)",
            "ILU + CuSparse CG ("+str(steps[4]) +" Steps)",
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

        ax.set_yscale("log")


        ax.set_title(
            "Step " + str(step) + " on Attelas \n (7165x7165, 1e-10 tolerance)")
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("Methods")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig("plot"+ str(step) +".png", bbox_inches='tight', dpi=300)
        plt.close('all')

    times = times_tot[0]
    steps = steps_tot[0]
    for i in range(len(times_tot)-1):
        times = np.concatenate((times, times_tot[i+1]), axis=1)
        steps += steps_tot[i+1]
    steps = steps // len(times_tot)

    stds = np.std(times, axis=1)
    means = np.median(times, axis=1)
    confidence = 0.95
    interval = np.empty((2, times.shape[0]))
    for i in range(times.shape[0]):
        interval[:, i] = st.t.interval(confidence=confidence, df=len(times[i])-1,
                                    loc=np.median(times[i]),
                                    scale=st.sem(times[i]))

    yer = [interval[:, i].reshape((2, -1)) for i in range(times.shape[0])]
    for i in range(times.shape[0]):
        yer[i][0] = -yer[i][0] + means[i]
        yer[i][1] = yer[i][1] - means[i]

    # width of the bars
    barWidth = 0.8
    position0 = np.arange(int(times.shape[0]))
    position1 = position0 + barWidth

    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    colors = ["blue", "cyan", "olive", "red",
            "green", "orange", "purple", "gray", "brown", "black", "pink", "yellow"]

    labels = [
        "MKL gesv (14 Threads)",
        "MKL posv (14 Threads)",
        "MKL gbsv (14 Threads)",
        "MKL pbsv (14 Threads)",
        "CuSparse CG ("+str(steps[0]) +" Steps)",
        "CG with Guess ("+str(steps[1]) +" Steps)",
        "Precon. CG ("+str(steps[2]) +" Steps)",
        "Precon. CG with Guess ("+str(steps[3]) +" Steps)",
        "ILU + CuSparse CG ("+str(steps[4]) +" Steps)",
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

    ax.set_yscale("log")


    ax.set_title(
        "Average on Attelas \n (7165x7165, 1e-10 tolerance)")
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Methods")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig("plot_average.png", bbox_inches='tight', dpi=300)
    plt.close('all')