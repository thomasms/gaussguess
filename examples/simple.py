import gaussguess as gg

signal = gg.GaussDistribution(9, sigma=0.1).sample(nentries=10000)
noise = gg.FlatDistribution(9).sample(nentries=200000)
measured = signal + 0.3*noise
# print(gen.values)
gg.histplot(signal).show()
gg.histplot(noise).show()
gg.histplot(measured).show()
