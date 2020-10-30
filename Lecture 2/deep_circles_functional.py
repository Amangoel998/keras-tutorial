

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split as tts

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


X, y = make_circles(n_samples=1000, factor=.6, noise=.1, random_state=42)
plot_data(plt, X, y)
#.savefig("mygraph_functional.png")
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

# Implementing Functional API
from keras.models import Model
from keras.layers import Input, Dense


from keras.optimizers import Adam

# Create an input layer
i = Input(shape=(2,))
# Add a refernce for a hidden layers with given inputs layer
x = Dense(4, activation="tanh", name="Hidden-layer-1")(i)
# Now the next layer will take previos hidden layer as input and give ref back
y = Dense(4, activation="tanh", name="Hidden-layer-2")(x)
# Output Layer
o = Dense(1, activation="sigmoid", name="Output-layer")(y)

# Create the model by providing input and output layers

model = Model(inputs=i, outputs=o)

# Summarize Model
model.summary()

model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)
eval_result = model.evaluate(X_test, y_test)

print("\nLoss: ", eval_result[0], "\nAccuracy: ", eval_result[1])

plot_decision_boundary(model, X, y).savefig("modelgraph_functional.png")