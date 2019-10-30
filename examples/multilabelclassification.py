import gaussguess as gg

NBINS = 111

classes = ["gauss", "uniform", "triangular", "laplace", "poisson"]
iclasses = range(len(classes))

getlabel = lambda name: iclasses[classes.index(name)]
labeled_data = [
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.05)),
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.10)),
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.15)),
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.75)),
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.20)),
    (getlabel("gauss"), gg.GaussDistribution(NBINS, sigma=0.20)),
    (getlabel("laplace"), gg.LaplaceDistribution(NBINS, lam=0.1)),
    (getlabel("laplace"), gg.LaplaceDistribution(NBINS, lam=0.12)),
    (getlabel("laplace"), gg.LaplaceDistribution(NBINS, lam=0.052)),
    (getlabel("uniform"), gg.UniformDistribution(NBINS)),
    (getlabel("uniform"), gg.UniformDistribution(NBINS)),
    (getlabel("uniform"), gg.UniformDistribution(NBINS)),
    (getlabel("uniform"), gg.UniformDistribution(NBINS)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.1)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.2)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.5)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.6)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.99)),
    (getlabel("triangular"), gg.TriangularDistribution(NBINS, center=0.4)),
    (getlabel("poisson"), gg.PoissonDistribution(NBINS, lam=6, xlim=[0, 10])),
    (getlabel("poisson"), gg.PoissonDistribution(NBINS, lam=2, xlim=[0, 10])),
    (getlabel("poisson"), gg.PoissonDistribution(NBINS, lam=3, xlim=[0, 10])),
]

labels, dists = map(list, zip(*labeled_data))

classifier = gg.DistributionMultiLabelClassifier(NBINS, nlabels=len(classes))
classifier.generatedata(dists, labels, nloops=50000, statsrange=[100, 10000], trainingratio=0.8)
cb = gg.LossAndAccuracyCallback()
test_loss, test_acc = classifier.train(epochs=20, callbacks=[cb])
# print(cb.epochs, cb.accuracy)
classifier.savemodel("{}bins_model_multilabel.h5".format(NBINS))