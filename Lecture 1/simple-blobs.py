#! C:/Program Files/Python3/python.exe

# Define a network that can find and separate data from 2 blobs
# from different classes

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Helper functions
def plot_data(pl, X, y):
    pl.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    pl.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl

# Funcition to draw decision boundaries
def plot_decision_boundary(model, X, y):
    Amin, Bmin = X.min(axis=0) - 0.1
    Amax, Bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(Amin, Amax, 101)
    vticks = np.linspace(Bmin, Bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    plot_data(plt, X, y)

    return plt


# Generate data blobs. Data will be either 0 or 1 when 2 is number of centers.
# X is a [number of samples, 2] sized array. X[sample] contains x,y position of sample in space
# ex: X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
# y is a [number of samples] sized array. y[sample] contains class index (ie. 0 or 1 when theer are 2 centers)
# ex: y[1] = 0, y[1] = 1
X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

# To show the figure in tkinter GUI
# plot_data(plt, X, y).show()

# Instead save the figure
plot_data(plt, X, y).savefig("mygraph.png")


# Split data into training data and test Data
from sklearn.model_selection import train_test_split as tts

# Split 30% data into test and rest to train
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

# Create keras model
from keras.models import Sequential
from keras.layers import Dense
# Adam is used for back propagation to adjust weights & bias
from keras.optimizers import Adam

# Steps to create a models
# 1. Create a model
# 2. Add Layers
# 3. Compile model
# 4. Train Model (via fit)
# 5. Evalute Performance w/ testing data

# 1. Create model from sequestial Layer, each layer is inserted at end of network and gets input from previous layer
# OR input from data passed in case of first Layer
model = Sequential()

# 2. Add Layer, which is first and only dense layer we train to divide 2 classes data
# There's only 1 neuron(param 1), ie data belongs to class 0 or 1
# Using shape with (y,x) as (2,), 1D array of 2 elements with unspecified batches of data
# sigmod retrun 0 or 1, 
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))

# 3. Compile
# Adam optimizer to reduce loss, model how incorrectly predicts the class
# Binary_entropy calculates the loss
# Maximize the accuracy
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

# 4. Fit (Make adjust & bias) model with data from make_blobs to minimize loss
# Running trainin set through fit 100 times, each run called epoch to optimize
model.fit(X_train, y_train, epochs=100, verbose=0)

# 5. Evalute model for loss & accuracy  on test data
eval_result = model.evaluate(X_test, y_test)

print("\nLoss: ", eval_result[0], "\nAccuracy: ", eval_result[1])

# Plot decision boundary of model we learned
plot_decision_boundary(model, X, y).savefig("modelgraph.png")