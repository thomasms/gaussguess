import gaussguess as gg

nbins = 51
stats = 100000

dists = [
    gg.GaussDistribution(nbins, sigma=0.1),
    gg.GaussDistribution(nbins, sigma=1.0),
    gg.GaussDistribution(nbins, sigma=0.01),
    gg.UniformDistribution(nbins),
    gg.UniformDistribution(3),
    gg.TriangularDistribution(nbins),
    gg.TriangularDistribution(nbins, center=0.4),
    gg.TriangularDistribution(47),
]

for d in dists:
    result = d.sample(nentries=stats)
    gg.histplot(result).show()
