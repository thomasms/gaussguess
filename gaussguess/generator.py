import math
import numpy as np


class Generator(object):
    def __init__(self, nbins, *args, **kwargs):
        """
           nbins must be odd!
        """
        assert nbins%2 == 1
        self._raw = None
        self.values = None
        self.nbins = nbins

        # always keep xrange between [-0.5, 0.5]
        self.binedges = np.linspace(-0.5, 0.5, num=self.nbins+1, endpoint=True)
        assert len(self.binedges) == (self.nbins + 1)
        self.binwidth = (self.binedges[-1] - self.binedges[-2])

    @property
    def limits(self):
        return min(self.binedges), max(self.binedges)

    def sample(self, nentries=10000):
        self._raw = np.ones(nentries)*0.1
        self.values, _ = np.histogram(self._raw, self.binedges)
        # normalise values between [0,1]
        self.values = self.values/max(self.values)

        return self

class GaussGenerator(Generator):
    def __init__(self, *args, sigma=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        # centre around 0
        self.__mu = 0
        self.sigma = sigma

    def sample(self, nentries=10000):
        self._raw = np.random.normal(self.__mu, self.sigma, nentries)
        self.values, _ = np.histogram(self._raw, self.binedges)

        # normalise values between [0,1]
        self.values = self.values/max(self.values)

        return self

    @property
    def analytical(self):
        actual = 1.0/(self.sigma * np.sqrt(2 * np.pi)) * \
            np.exp( - (self.binedges - self.__mu)**2 / (2 * self.sigma**2) )
        actual = actual/max(actual)
        return self.binedges, actual


