import gaussguess as gg

gen = gg.GaussGenerator(111, sigma=0.1).sample(nentries=100000)
print(gen.values)
gg.hist_plot(gen).show()
