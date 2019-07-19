import matplotlib.pyplot as plt


def hist_plot(generator):
    # plot data to look like hist but actuall line plot
    x = []
    y = []
    for i in range(len(generator.values)):
        x.append(generator.binedges[i])
        y.append(generator.values[i])
        x.append(generator.binedges[i+1])
        y.append(generator.values[i])

    f = plt.figure()
    plt.plot(x, y, 'k', alpha=0.6)
    plt.plot(*generator.analytical, 'r', alpha=0.4)
    plt.xlim(generator.limits)
    plt.ylabel("count / {:.3e}".format(generator.binwidth))
    plt.title("nbins={}, sigma={:.2f}".format(generator.nbins, generator.sigma))

    return plt