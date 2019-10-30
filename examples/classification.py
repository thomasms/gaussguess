import gaussguess as gg

NBINS = 21

signaldists = [
    gg.GaussDistribution(NBINS, sigma=0.05),
    gg.GaussDistribution(NBINS, sigma=0.1),
    gg.GaussDistribution(NBINS, sigma=0.15),
    gg.GaussDistribution(NBINS, sigma=0.2),
    gg.GaussDistribution(NBINS, sigma=0.25),
    gg.GaussDistribution(NBINS, sigma=0.3),
    gg.GaussDistribution(NBINS, sigma=0.4),
    gg.GaussDistribution(NBINS, sigma=2.0, xlim=[0, 10]),
]

backgrounddists = [
    gg.LaplaceDistribution(NBINS, lam=0.1),
    gg.LaplaceDistribution(NBINS, lam=1.2),
    gg.UniformDistribution(NBINS),
    gg.TriangularDistribution(NBINS),
    gg.TriangularDistribution(NBINS, center=0.4),
    gg.TriangularDistribution(NBINS, center=0.3),
    gg.TriangularDistribution(NBINS, center=0.8),
    gg.PoissonDistribution(NBINS),
    gg.PoissonDistribution(NBINS, lam=6, xlim=[0, 10]),
]

classifier = gg.DistributionBinaryClassifier(NBINS)
classifier.generatedata(signaldists, backgrounddists, nloops=50000, statsrange=[10, 1000], trainingratio=0.8)
cb = gg.LossAndAccuracyCallback()
test_loss, test_acc = classifier.train(epochs=20, callbacks=[cb])
# print(cb.epochs, cb.accuracy)
classifier.savemodel("{}bins_model.h5".format(NBINS))