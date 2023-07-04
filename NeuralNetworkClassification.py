from typing import Tuple

import torch
from torchview import draw_graph
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

sns.set(rc={'figure.figsize': (10, 7)})
sns.set(rc={'figure.dpi': 100})
sns.set(style='white', palette='muted', font_scale=1.2)

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv('penguins.csv').dropna()
print(df.head())
print(df.shape)

# plot for the visualization of species
'''
df['species'].value_counts().plot(kind='bar', rot=0)
plt.show()
'''

# scatter plot for the visualization of flipper_length_mm and body_mass_g of the species
'''
sns.scatterplot(data=df, x='flipper_length_mm', y='body_mass_g', hue='species')
plt.show()
'''

# Pairplot is a matrix of scatter plots, with each variable plotted against each other variable. The diagonal shows
# the distribution of the variable. The off-diagonal shows the scatter plots of the variables against each other. The
# hue parameter is used to color the points in the scatter plot. what is the diagonal? The distribution of the
# variable. How do I understand the distribution of the variable? The distribution of the variable is the frequency
# of the variableo i.e number of times the variable occurs in the dataset.
'''
sns.pairplot(df, hue='species')
plt.show()
'''

# Neural Network Classification
# weight initialization
# torch.nn.init.kaiming_normal(model.weights) or torch.nn.init.xavier_normal(model.weights)

# Activation functions
# ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
'''
x = torch.linspace(-5, 5, 100)
y = torch.relu(x)
plt.plot(x.numpy(), x.numpy(), label="Linear")
plt.plot(x.numpy(), y.numpy(), label="ReLU")
plt.xlabel('Input')
plt.ylabel('Output')
plt.legent()
plt.show()
'''
'''
x = torch.linspace(-5, 5, 100)
linear = nn.Linear(100, 100)
linear.requires_grad_(False)
y_linear = linear(x)
y_relu = torch.relu(y_linear)
plt.scatter(x.numpy(), y_linear.numpy(), label='Linear')
plt.scatter(x.numpy(), y_relu.numpy(), label='ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
'''


# Create a Neural Network for Classification
class PenguinClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_features, 8)
        self.linear_2 = nn.Linear(8, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(features)
        x = torch.relu(x)
        x = self.linear_2(x)
        return x


model = PenguinClassifier(n_features=4, n_classes=3)

model_graph = draw_graph(model, input_size=(1, 4))
model_graph.visual_graph


# Dataset Splitting
train, test = train_test_split(df, test_size=0.2, random_state=42)
# reset index: It resets the index of the dataframe, and use the drop parameter to avoid the old index being added as
# a column
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
print(train.shape, test.shape)

# Data Preprocessing
# 1. Convert the data to tensors
# 2. Normalize the data
Species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def create_dataset(data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values,
        dtype=torch.float32
    )
    labels = torch.tensor(data['species'].map(Species_map), dtype=torch.int64)
    return features, labels


train_features, train_labels = create_dataset(train)
test_features, test_labels = create_dataset(test)
print(train_features.shape, train_labels.shape)
print(test_features.shape, test_labels.shape)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training
def calculate_accuracy(model_outputs, labels):
    value, predicted_index = torch.max(model_outputs, dim=1)
    # how does torch.max work? torch.max returns the maximum value of the tensor and the index of the maximum value
    # of the tensor. torch.max(model_outputs, dim=1) returns the maximum value of the tensor and the index of the
    # maximum value of the tensor along the dimension 1.
    correct = (predicted_index == labels).sum().item()  # In this case predicted_index is the label itself
    return 100 * (correct / labels.shape[0])


epochs = 3000
losses = {
    'train': [],
    'test': []
}
accuracies = {
    'train': [],
    'test': []
}
for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()

    train_outputs = model(train_features)
    train_loss = loss_fn(train_outputs, train_labels)
    train_acc = calculate_accuracy(train_outputs, train_labels)
    losses['train'].append(train_loss.item())
    accuracies['train'].append(train_acc)
    train_loss.backward()
    optimizer.step()

    model.eval()

    with torch.inference_mode():
        outputs = model(test_features)
        test_loss = loss_fn(outputs, test_labels)
        test_acc = calculate_accuracy(outputs, test_labels)
        losses['test'].append(test_loss.item())
        accuracies['test'].append(test_acc)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), mode='valid') / window_size

'''
plt.plot(moving_average(losses['train'], 100), label='train')
plt.plot(moving_average(losses['test'], 100), label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(moving_average(accuracies['train'], 100), label='train')
plt.plot(moving_average(accuracies['test'], 100), label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
print(f"Train Accuracy: {accuracies['train'][-1]:.2f}")
print(f"Test Accuracy: {accuracies['test'][-1]:.2f}")

# confusion matrix
outputs = model(test_features)
_, predictions = torch.max(outputs, dim=1)
'''cm = confusion_matrix(test_labels, predictions, labels=list(Species_map.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(Species_map.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.tight_layout()
plt.show()'''
# sns.boxplot(data=df, x='species', y='body_mass_g')
# plt.show()
# what is a boxplot? A box plot is a method for graphically depicting groups of numerical data through their
# quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability
# outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram. Outliers
# may be plotted as individual points. Box plots are non-parametric: they display variation in samples of a
# statistical population without making any assumptions of the underlying statistical distribution. The spacings
# between the different parts of the box indicate the degree of dispersion (spread) and skewness in the data,
# and show outliers. In addition to the points themselves, they allow one to visually estimate various L-estimators,
# notably the interquartile range, midhinge, range, mid-range, and trimean. Box plots can be drawn either
# horizontally or vertically. Box plots received their name from the box in the middle.

# what is a quartile? A quartile is a type of quantile. The first quartile (Q1) is defined as the middle number
# between the smallest number (minimum) and the median of the data set. The second quartile (Q2) is the median of the
# data. The third quartile (Q3) is the middle value between the median and the highest value (maximum) of the data set.
