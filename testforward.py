import numpy as np
import perdata
import random as r
import stringtoint as ss

from stringtoint import strToInt

class Perceptron:
    def __init__(self, learning_rate, epochs) :
        self.weights = [0] * 100
        self.bias = 1
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
    def fit(self, inputs, labels):
        for _ in range(self.epochs):
            numCorrect = 0
            for input_vector, label in zip(inputs, labels):
                weighted_sum = p.dot(input_vector, self.weights)
                activation = p.activation(weighted_sum)
                guess = activation
                correctQ = guess - label
                if correctQ == 0:
                    numCorrect += 1
                newWeights = [self.learning_rate * (label - activation) * n for n in input_vector]
                self.weights = [sum(x) for x in zip(self.weights, newWeights)]
                self.bias += self.learning_rate * (label - activation)
                #print(self.weights)
                #print("Self error: {}".format(error))
            print("Current epoch: {}".format(_))
            #print(self.weights)
            avgError = numCorrect / len(labels)
            print("Average error: {}".format(avgError))
        return self.weights




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




p = Perceptron(0.01, 1000)

new_weights = p.fit(X_train_int,y_train)

print(new_weights)
print(p.bias)


usrP = input("Enter a phrase here: ")

usrD = ss.strToInt(usrP)

res = p.predict(usrD)

print(res)


#model = p.fit(X_train_int, y_train)







#pred = p.predict(X_test_int)

