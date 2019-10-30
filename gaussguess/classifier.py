import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .distribution import Distribution

class LossAndAccuracyCallback(keras.callbacks.Callback):

    def __init__(self):
        super(LossAndAccuracyCallback, self).__init__()

        # best_weights to store the weights at which the minimum loss occurs.
        self.epochs = []
        self.loss = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.loss.append(logs['loss'])
        self.accuracy.append(logs['accuracy'])

class DistributionClassifier(object):
    """
        Use a simple NN to identify a distribution
        Either it is a signal distribution (Gaussian) 
        or background (not Gaussian).

        The final layer has size 2 [prob_of_back_dist, prob_of_signal_dist]
    """
    def __init__(self, nbins):
        self._nbins = nbins

        self._model = None

        self.test_labels = None
        self.test_data = None

        self.training_labels = None
        self.training_data = None

    def loadmodel(self, filename):
        """
            Load an existing trained model here
        """
        self._model = keras.models.load_model(filename)
        return self

    def savemodel(self, filename):
        """
            Save as h5 file
        """
        self._model.save(filename)
        return self

    def generatedata(self, signaldistributions, backdistributions, nloops=1000, 
        statsrange=[10, 1000], trainingratio=0.9):
        pass

    def train(self, epochs=10, callbacks=None):
        cb = callbacks
        if not callbacks:
            cb = []
        
        _ = self._model.fit(self.training_data, self.training_labels, epochs=epochs,
          callbacks=cb)
        test_loss, test_acc = self._model.evaluate(self.test_data, self.test_labels)
        return test_loss, test_acc

    def predict(self, distribution):
        """
            Expects a distribution of type Distribution
        """
        values = distribution.values
        assert len(values) == self._nbins
        return self._model.predict(values.reshape((1, self._nbins)))[0]
        

class DistributionBinaryClassifier(DistributionClassifier):
    """
        Use a simple NN to identify a distribution
        Either it is a signal distribution (Gaussian) 
        or background (not Gaussian).

        The final layer has size 2 [prob_of_back_dist, prob_of_signal_dist]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(self._nbins,)),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])

        self._model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def generatedata(self, signaldistributions, backdistributions, nloops=1000, 
        statsrange=[10, 1000], trainingratio=0.9, normfunc=lambda x: x/sum(x)):
        """
            Set the training and test data here.

            Does equal number of signal and background distributions, based on nloops.

            Randomly samples from list of distributions.

            Distributions must be a list of type Distribution.
        """
        try:
            from tqdm import tqdm
            iter = tqdm(range(nloops))
        except:
            iter = range(nloops)
            
        # list of data + labels
        data = []
        for _ in iter:
            nentries = random.randrange(statsrange[0], statsrange[1])

            distindx = random.randrange(0, len(signaldistributions))
            signaldistributions[distindx].sample(nentries=nentries).normalise(op=normfunc)
            data.append((signaldistributions[distindx].values, 1))

            distindx = random.randrange(0, len(backdistributions))
            backdistributions[distindx].sample(nentries=nentries).normalise(op=normfunc)
            data.append((backdistributions[distindx].values, 0))

        # split based on trainingratio
        ntraining = math.floor(len(data)*trainingratio)
        ntest = len(data) - ntraining

        self.test_labels = np.array([l for _, l in data[:ntest]])
        self.test_data = np.array([d for d, _ in data[:ntest]])

        self.training_labels = np.array([l for _, l in data[ntest:]])
        self.training_data = np.array([d for d, _ in data[ntest:]])
        return self

class DistributionMultiLabelClassifier(DistributionClassifier):
    """
        Use a simple NN to identify a distribution label
    """
    def __init__(self, *args, nlabels=2, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(self._nbins,)),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(nlabels, activation=tf.nn.softmax)
        ])

        self._model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def generatedata(self, distributions, labels, nloops=1000, 
        statsrange=[10, 1000], trainingratio=0.9, normfunc=lambda x: x/sum(x)):
        """
            Set the training and test data here.

            Does equal number of signal and background distributions, based on nloops.

            Randomly samples from list of distributions.

            Distributions must be a list of type Distribution.
        """
        assert len(distributions) == len(labels)

        try:
            from tqdm import tqdm
            iter = tqdm(range(nloops))
        except:
            iter = range(nloops)
            
        # list of data + labels
        data = []
        for _ in iter:
            nentries = random.randrange(statsrange[0], statsrange[1])

            distindx = random.randrange(0, len(distributions))
            distributions[distindx].sample(nentries=nentries).normalise(op=normfunc)
            data.append((distributions[distindx].values, labels[distindx]))

        # split based on trainingratio
        ntraining = math.floor(len(data)*trainingratio)
        ntest = len(data) - ntraining

        self.test_labels = np.array([l for _, l in data[:ntest]])
        self.test_data = np.array([d for d, _ in data[:ntest]])

        self.training_labels = np.array([l for _, l in data[ntest:]])
        self.training_data = np.array([d for d, _ in data[ntest:]])
        return self
