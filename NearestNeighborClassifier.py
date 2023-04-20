# Nearest Neighbor Classifier
import numpy as np


class NearestNeighbor(object):

    def __init__(self):
        self.ytr = None
        self.Xtr = None

    def train(self, x, y):
        self.Xtr = x
        self.ytr = y

    def predict(self, Xte):
        #  Use of np.zeros to initialize the output variable
        ype = np.zeros(Xte.size[0])

        for i in range(Xte.size[0]):
            # Dimension of Xte is NxD, we want to compare each image
            # by computing L1 distance between all the pixels of each image
            # Notation to access 1xD would be Xte[1, :] or Xte[1]
            dist = np.sum(np.abs(self.Xtr - Xte[i, :]), axis=1)
            index = np.argmin(dist)
            ype[i] = self.ytr[index]

        return ype


def main():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')  # a magic function we provide
    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

    nn = NearestNeighbor()
    nn.train(Xtr, Ytr)
    Ype = nn.predict(Xte)
    print(np.mean(Ype == Yte) * 100)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
