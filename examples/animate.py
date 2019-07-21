import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import gaussguess as gg


stats = 100000
sigma = 0.05
fig, ax = plt.subplots()

gen = gg.GaussDistribution(3, sigma=sigma).sample(nentries=stats)
x, y = gg.getplotxy(gen)
lims = gen.limits

line, = ax.plot(x, y, 'k', alpha=0.6)

def animate(i):
    gen = gg.GaussDistribution((2*i) + 1, sigma=sigma).sample(nentries=stats)
    x, y = gg.getplotxy(gen)
    line.set_xdata(x)  # update the data
    line.set_ydata(y)  # update the data
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 100),
                                interval=50, blit=False)

plt.xlim(lims)
plt.ylabel("count / {:.3e}".format(gen.binwidth))
plt.title("Increasing number of bins")
# ani.save('animate.gif', writer='imagemagick', fps=10)

plt.show()