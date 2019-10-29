import gaussguess as gg

NBINS = 11

signaldists = [
    gg.GaussDistribution(NBINS, sigma=0.1),
    gg.GaussDistribution(NBINS, sigma=1.0),
    gg.GaussDistribution(NBINS, sigma=2.0, xlim=[0, 10]),
    gg.GaussDistribution(NBINS, sigma=0.2)
]

backgrounddists = [
    gg.LaplaceDistribution(NBINS, lam=0.1),
    gg.LaplaceDistribution(NBINS, lam=1.2),
    gg.UniformDistribution(NBINS),
    gg.TriangularDistribution(NBINS),
    gg.TriangularDistribution(NBINS, center=0.4),
    gg.PoissonDistribution(NBINS),
    gg.PoissonDistribution(NBINS, lam=6, xlim=[0, 10]),
]

classifier = gg.Classifier(NBINS)
classifier.generatedata(signaldists, backgrounddists, nloops=50000, statsrange=[10, 1000], trainingratio=0.9)
test_loss, test_acc = classifier.train(epochs=100)
print(test_loss, test_acc)
classifier.savemodel("{}bins_model.h5".format(NBINS))