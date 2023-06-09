import numpy as np
import perdata
import random as r

class Perceptron:
    def __init__(self, learning_rate, epochs) :
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self,z):
        return np.heaviside(z,0)

    def fit(self, X, y):
        n_features = X.shape[1]

        self.weights = np.zeros((n_features))
        self.bias = 0

        for epoch in range(self.epochs):

            for i in range(len(X)):
                z = np.dot(X, self.weights) + self.bias
                y_pred = self.activation(z)
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * X[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)


# import data and put into two lists, train and test

allData = perdata.allData

r.shuffle(allData)

trainData = allData[:800]
testData = allData[800:]


X_train, y_train = zip(*trainData)
X_test, y_test = zip(*testData)


p = Perceptron(0.001, 100)

p.fit(X_train, y_train)

pred = p.predict(X_test)






# Approach: randomly get 800 nonrepeating indices

