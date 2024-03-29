import matplotlib.pyplot as plt


def getplotxy(generator):
    # plot data to look like hist but actuall line plot
    x = []
    y = []
    for i in range(len(generator.values)):
        x.append(generator.binedges[i])
        y.append(generator.values[i])
        x.append(generator.binedges[i+1])
        y.append(generator.values[i])

    return x, y

def histplot(generator, *args, fig=None, **kwargs):
    # plot data to look like hist but actuall line plot
    x, y = getplotxy(generator)

    if not fig:
        _ = plt.figure()

    plt.plot(x, y, *args, **kwargs)
    # plt.plot(*generator.analytical, 'r', alpha=0.4)
    # plt.xlim(generator.limits)
    # plt.ylim([0,1.05])
    plt.ylabel("count / {:.3e}".format(generator.binwidth))
    plt.title("nbins={}".format(generator.nbins))

    return plt