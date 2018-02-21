from os.path import join
import numpy as np
from constants import *


class DatasetLoader(object):
    def __init__(self):
        pass

    def load_from_save(self):
        self._images = np.load(TRAINING_SET)
        self._labels = np.load(TRAINING_LABELS)
        self._images_test = np.load(TEST_SET)
        self._labels_test = np.load(TEST_LABELS)
        self._labels_test = self._labels.reshape([-1, len(EMOTIONS)])
        self._labels = self._labels.reshape([-1, len(EMOTIONS)])
        self._images_test = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
        self._images = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def images_test(self):
        return self._images_test

    @property
    def labels_test(self):
        return self._labels_test
