#! C:/Program Files/Python3/python.exe

# Define a network that can find and separate data from 2 blobs
# from different classes

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split as tts
from keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def plot_data(pl, X, y):
    pl.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)
    pl.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl

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


X, y = make_circles(n_samples=1000, factor=0.6, noise=.1, random_state=42)
plot_data(plt, X, y).savefig("mygraph1.png")



X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)
model = Sequential()
model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)
eval_result = model.evaluate(X_test, y_test)

print("\nLoss: ", eval_result[0], "\nAccuracy: ", eval_result[1])
plot_decision_boundary(model, X, y).savefig("modelgraph1.png")