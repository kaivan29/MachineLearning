import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

# Predicting penguin body mass from flipper length
# https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data
# https://www.kaggle.com/parulpandey/penguins-simple-linear-regression

df = pd.read_csv('penguins.csv')
print(df.head())
# print(df.shape)
df = df.dropna()
print(df.shape)

df.plot.scatter("flipper_length_mm", "body_mass_g", figsize=(12, 8), s=36)


# what do the above parameters mean?
# figsize: tuple of integers, width, height in inches
# s: scalar or array_like, shape (n, ), optional
# what does s do on the plot? It is the size of the dots on the plot
# plt.tight_layout()
# what does tight_layout do? Automatically adjust subplot parameters to give specified padding.
# plt.show()


# y = wx + b
# weight = w x flipper_length + b

class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # what is nn.Parameter? A kind of Tensor, that is automatically registered as a parameter when assigned as an
        # attribute to a Module. Parameters and buffers need to be registered, or they will not appear in .parameters()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        y = self.weights * data + self.bias
        return y


model = SimpleLinearRegression()
# print(model)

# Parameters
# PyTorch nn.Parameter is a special type of tensor that is meant to be used as a parameter in a neural
# network. It acts like a normal tensor, but it has a different set of methods and attributes to make it easier to
# use in deep learning models.
# The main difference between nn.Parameter and a regular tensor is that nn.Parameter is registered in the model's
# parameters and will be optimized during the training process. Additionally, the requires_grad attribute of
# nn.Parameter is set to True by default, allowing PyTorch to keep track of the gradients of the parameters during
# the training process and update them as needed. This makes it easier and more convenient to train deep learning
# models with PyTorch.

# name_parameters() returns an iterator over module parameters, yielding both the name of the parameter as well as
# the parameter itself.
for name, param in model.named_parameters():
    print(name, param.data)

flipper_length = torch.tensor(df.flipper_length_mm.values)
body_mass = torch.tensor(df.body_mass_g.values)
# print(flipper_length.shape)
# print(body_mass.shape)

# Untrained Model prediction
predicted_body_mass = model(flipper_length)

# what does .detach() do to a tensor? Returns a new Tensor, detached from the current graph.
# The result will never require gradient. What does this mean? It means that the tensor will not be used to compute
# gradients. In other words, the tensor is not a part of the computational graph. This is useful when you want to
# use a tensor as a constant in further computations / calculations.
# plt automatically converts tensor to numpy array and then plots it
'''
plt.plot(flipper_length, predicted_body_mass.detach(), label="predicted")
plt.plot(flipper_length, body_mass, ".", label="original data")
plt.legend()
plt.show()
'''

# Training
# Loss function: Mean Squared Error
loss_fn = nn.MSELoss()
# Initialize the optimizer: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Train the model
losses = []

# The loss is then calculated using the loss_fn function and the predicted_mass and body_mass. The current loss value
# is added to the losses list.
# After calculating the loss, the gradients are set to zero using optimizer.zero_grad(), and the backpropagation is
# performed using loss.backward(). Finally, the optimizer updates the model parameters using optimizer.step().
for i in tqdm(range(10000)):
    predicted_body_mass = model(flipper_length)
    loss = loss_fn(predicted_body_mass, body_mass)
    # what does torch.item() do? Returns the value of this tensor as a standard Python number. This only works for
    # tensors with one element. For other cases, see tolist().
    losses.append(loss.item())
    # what does zero_grad() do? Sets gradients of all model parameters to zero.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss function
'''
plt.figure(figsize=(12, 10))
plt.plot(losses)
plt.tight_layout()
plt.show()
'''
# It appears that the loss has decreased significantly. The lower the loss value, the more accurate the predictions
# of the model will be.

# Evaluate the model
predictions = model(flipper_length)
# print(predictions.detach()[:5])
# print(body_mass[:5])
'''
plt.plot(flipper_length, predictions.detach(), label="predicted")
plt.plot(flipper_length, body_mass, ".", label="original data")
plt.legend()
plt.show()
'''

# evaluate the model using the mean absolute error
# MAE = 1/n * sum(|y - y_hat|)
MAE = (torch.abs(predictions - body_mass)).mean()
print(MAE)
# what does 470 MAE mean here? It means that on average, the model's predictions are off by 470 grams. This is not
# very good, considering that the average body mass is around 4200 grams. This means that the model's predictions are
# off by around 10% on average. This is not very good, but it is not terrible either. It is possible to improve the
# model's performance by using a more complex model, or by using a more complex loss function.
