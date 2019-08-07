import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import gaussguess as gg


nbins = 111
sigma = 0.1
fig, ax = plt.subplots()

dist = gg.GaussDistribution(nbins, sigma=sigma, xlim=[0,1])
gen = dist.sample(nentries=10)
x, y = gg.getplotxy(gen)
lims = gen.limits

line, = ax.plot(x, y, 'k', alpha=0.6)

lastsize = 1
def animate(i):
    global lastsize
    gen = dist.sample(nentries=lastsize)
    x, y = gg.getplotxy(gen)
    line.set_xdata(x)  # update the data
    line.set_ydata(y)  # update the data
    if lastsize > 2000000:
        lastsize = 1
    else:
        lastsize *= 2
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 20),
                                interval=100, repeat_delay=200, blit=False)

plt.xlim(lims)
plt.ylabel("count / {:.3e}".format(gen.binwidth))
plt.title("Increasing number of entries")
# ani.save('animate2.gif', writer='imagemagick', fps=10)

plt.show()