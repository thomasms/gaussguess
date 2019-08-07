import gaussguess as gg

nbins = 9
stats = 100
sn_ratio = 100
signal = gg.GaussDistribution(nbins, sigma=0.1).sample(nentries=stats)
noise = gg.UniformDistribution(nbins).sample(nentries=100)
measured = signal + (stats/sn_ratio)*noise
# print(gen.values)
gg.histplot(signal).show()
gg.histplot(noise).show()
gg.histplot(measured).show()
