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
            # L1 distance is also known as Manhattan distance
            L1dist = np.sum(np.abs(self.Xtr - Xte[i, :]), axis=1)
            # L2 distance is also known as Euclidean Distance
            L2dist = np.sqrt(np.sum(np.square(np.abs(self.Xtr - Xte[i, :])), axis=1))
            index = np.argmin(L1dist)
            ype[i] = self.ytr[index]

        return ype


def main():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')  # a magic function we provide
    # flatten out all images to be one-dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072

    # Use validation set for hyperparameter tuning, so use a part of the training dataset as validation dataset
    Xval_rows = Xtr_rows[:1000, :]
    Xtr_rows = Xtr_rows[1000:, :]
    Yval = Ytr[:1000]
    Ytr = Ytr[1000:]

    # Hyperparameter Tuning for k in K-NN
    K = [1, 3, 5, 10]
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    accuracy = np.array()
    # Use validation set for hyperparameter tuning
    for k in K:
        Ype = nn.predict(Xval_rows, k)
        accuracy.append(np.mean(Ype == Yval))

    # Do prediction now using the k
    k = np.argmax(accuracy)
    Ype = nn.predict(Xte_rows, k)
    print("Accuracy: ", np.mean(Ype == Yval) * 100)

# Cross Validation
# In practice, people prefer to avoid cross-validation in favor of having a single validation split, since
# cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training
# data for training and rest for validation. However, this depends on multiple factors: For example if the no. of
# hyperparameter is large you may prefer to use bigger validation splits. If the number of examples in the validation
# set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can
# see in practice would be 3-fold, 5-fold or 10-fold cross- validation

# 1. Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit
# variance.
# 2. If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA
# 3. Split your training data randomly into 70-90% train/val. More hyperparameter, means larger validation data.
# If data set is less use cross-validation
# 4. Train and evaluate on validation data use many values of k
# 5. if kNN takes too long, use ANN i.e approximate nearest neighbor
# 6.
