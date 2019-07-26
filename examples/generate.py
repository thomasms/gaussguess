import gaussguess as gg

NTRUE  = 100000
NFALSE = 100000
NBINS = 5
SIGMA = 1.0
STATS=100

counter = 0
with open('samples_size{}.csv'.format(NBINS), 'wt') as csv:
    csv.write("index,sigma,stats,isgauss,{}\n".format(",".join(["bin{}".format(i) for i in range(NBINS)])))
    for _ in range(NTRUE):
        signal = gg.GaussDistribution(NBINS, sigma=SIGMA).sample(nentries=STATS)
        csv.write("{},{},{},{}".format(counter,SIGMA,STATS,1))
        for v in signal.values:
            csv.write(",{:.15e}".format(v))
        csv.write("\n")
        counter += 1
    for _ in range(NTRUE):
        background = gg.FlatDistribution(NBINS).sample(nentries=STATS)
        csv.write("{},{},{},{}".format(counter,SIGMA,STATS,0))
        for v in background.values:
            csv.write(",{:.15e}".format(v))
        csv.write("\n")
        counter += 1
