import numpy as np
import perdata
import random as r
import stringtoint as ss

from stringtoint import strToInt

class Perceptron:
    def __init__(self, learning_rate, epochs) :
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self,z):
        return np.heaviside(z,0)

    def dot(self, K, L):
        if len(K) != len(L):
            return 0
        return sum(i[0] * i[1] for i in zip(K, L))

    def fit(self, X, y):

        return #self.weights, self.bias

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)


# import data and put into two lists, train and test

allData = perdata.allData

r.shuffle(allData)

trainData = allData[:800]
testData = allData[800:]

X_train = [X[0] for X in trainData]
y_train = [X[1] for X in trainData]

X_test = [X[0] for X in testData]
y_test = [X[1] for X in testData]

X_train_int = [ss.strToInt(i) for i in X_train]
X_test_int = [ss.strToInt(i) for i in X_test]


### IMPORTANT - You need to decide what activation function to use. 
# Heaviside doesn't seem to work with your data type. 


p = Perceptron(0.001, 100)
#test_weights = [0] * len(X_train_int)
test_weights = [r.uniform(-0.99,0.99) for i in range(100)]
test_dot = p.dot(X_train_int[3], test_weights)
act = p.activation(test_dot)
model = p.fit(X_train_int, y_train)

print("Test weighst: {}".format(test_weights))

print("Test dot is {}".format(test_dot))

print("Activation is: {}".format(act))



#pred = p.predict(X_test_int)

