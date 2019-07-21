import gaussguess as gg

stats = 10000
sn_ratio = 100
signal = gg.GaussDistribution(9, sigma=0.1).sample(nentries=stats)
noise = gg.FlatDistribution(9).sample(nentries=10)
measured = signal + (stats/sn_ratio)*noise
# print(gen.values)
gg.histplot(signal).show()
gg.histplot(noise).show()
gg.histplot(measured).show()
