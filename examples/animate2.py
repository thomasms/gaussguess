import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

import gaussguess as gg
import gaussguess.plotter as plotter


NBINS = 21
sigma = 0.15
fig, ax = plt.subplots()

classes = ["gauss", "uniform", "triangular", "laplace", "poisson"]
iclasses = range(len(classes))

classifier = gg.DistributionMultiLabelClassifier(NBINS, nlabels=len(classes))
classifier.loadmodel("{}bins_model_multilabel.h5".format(NBINS))
normop = lambda x: x/max(x) if max(x) > 0 else 0

dist = gg.GaussDistribution(NBINS, sigma=sigma, xlim=[0,1])
gen = dist.sample(nentries=1).normalise()
prediction = classifier.predict(gen)
x, y = plotter.getplotxy(gen.normalise(op=normop))
lims = gen.limits

line, = ax.plot(x, y, 'k', alpha=0.6)
stat_label = ax.text(0.01, 0.99, "{} entries".format(1))
texts = []
startX = 0.7
startY = 0.95
for i, c in enumerate(classes):
    texts.append(ax.text(startX, startY, "{} = {:.2f}%".format(c, prediction[i]*100)))
    startY -= 0.05

def animate(i):
    entries = math.floor(math.pow(2,i))
    gen = dist.sample(nentries=entries).normalise()
    stat_label.set_text("{} entries".format(entries))
    prediction = classifier.predict(gen)
    x, y = plotter.getplotxy(gen.normalise(op=normop))
    for j, c in enumerate(classes):
        if prediction[j] > 0.5:
            texts[j].set_color('r')
        else:
            texts[j].set_color('k')
        texts[j].set_text("{} = {:.2f}%".format(c, prediction[j]*100))
    line.set_xdata(x)  # update the data
    line.set_ydata(y)  # update the data
    return [line,]

ani = animation.FuncAnimation(fig, animate, np.arange(0, 20),
                                interval=500, repeat_delay=200, blit=False)

plt.xlim(lims)
plt.ylabel("count / {:.3e}".format(gen.binwidth))
plt.title("Gaussian")
# ani.save('gauss_21.gif', writer='imagemagick', fps=2)

plt.show()