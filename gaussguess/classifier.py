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
        self.accuracy.append(logs['acc'])

class Classifier(object):
    """
        Use a simple NN to identify gaussians
    """
    def __init__(self, nbins):
        self._nbins = nbins

        self._model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(nbins,)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])

        self._model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

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
        """
            Set the training and test data here.

            Does equal number of signal and background distributions.

            Randomly samples from list of each distribution.



            distributions must be a list of type Distribution.
        """

        # list of data + labels
        data = []
        # signal
        for _ in range(nloops):
            distindx = random.randrange(0, len(signaldistributions))
            nentries = random.randrange(statsrange[0], statsrange[1])
            signaldistributions[distindx].sample(nentries=nentries)
            data.append((signaldistributions[distindx].values, 1))

        # background
        for _ in range(nloops):
            distindx = random.randrange(0, len(backdistributions))
            nentries = random.randrange(statsrange[0], statsrange[1])
            backdistributions[distindx].sample(nentries=nentries)
            data.append((backdistributions[distindx].values, 0))
        
        # shuffle data
        random.shuffle(data)

        # split based on trainingratio
        ntraining = math.floor(len(data)*trainingratio)
        ntest = len(data) - ntraining

        self.test_labels = np.array([l for _, l in data[:ntest]])
        self.test_data = np.array([d for d, _ in data[:ntest]])

        self.training_labels = np.array([l for _, l in data[ntest:]])
        self.training_data = np.array([d for d, _ in data[ntest:]])
        return self

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
            Expects an array of values of size equal to nbins
        """
        values = distribution.values
        assert len(values) == self._nbins
        return self._model.predict(values.reshape((1, self._nbins)))[0]
        
