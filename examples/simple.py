import gaussguess as gg
import gaussguess.plotter as plotter

nbins = 11
stats = 1000

dists = [
    gg.GaussDistribution(nbins, sigma=0.1),
    gg.GaussDistribution(nbins, sigma=1.0),
    gg.GaussDistribution(nbins, sigma=2.0, xlim=[0, 10]),
    gg.LaplaceDistribution(nbins, lam=0.1),
    gg.LaplaceDistribution(nbins, lam=1.2),
    gg.UniformDistribution(nbins),
    gg.UniformDistribution(3),
    gg.TriangularDistribution(nbins),
    gg.TriangularDistribution(nbins, center=0.4),
    gg.TriangularDistribution(47),
    gg.PoissonDistribution(nbins),
    gg.PoissonDistribution(nbins, lam=6, xlim=[0, 10]),
]

for d in dists:
    result = d.sample(nentries=stats)
    plotter.histplot(result).show()
