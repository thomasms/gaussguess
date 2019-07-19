import math
import numpy as np
import matplotlib.pyplot as plt


class Generator(object):
    def __init__(self, nbins, sigma=0.1):
        """
           nbins must be odd!
        """
        assert nbins%2 == 1
        # centre around 0
        self.__mu = 0
        self._raw = None
        self.nbins = nbins
        self.sigma = sigma
        self.values = None

        # always keep xrange between [-0.5, 0.5]
        self.binedges = np.linspace(-0.5, 0.5, num=self.nbins+1, endpoint=True)
        assert len(self.binedges) == (self.nbins + 1)
        self.binwidth = (self.binedges[-1] - self.binedges[-2])

    def sample(self, nentries=10000):
        self._raw = np.random.normal(self.__mu, self.sigma, nentries)
        self.values, _ = np.histogram(self._raw, self.binedges)

        # normalise values between [0,1]
        self.values = self.values/max(self.values)

        return self

    @property
    def limits(self):
        return min(self.binedges), max(self.binedges)

    @property
    def analytical(self):
        actual = 1.0/(self.sigma * np.sqrt(2 * np.pi)) * \
            np.exp( - (self.binedges - self.__mu)**2 / (2 * self.sigma**2) )
        actual = actual/max(actual)
        return self.binedges, actual


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

