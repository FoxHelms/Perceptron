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

    def fit(self, x_list, y):
        #n_features = len(x_list)

        self.weights = [0] * 100
        self.bias = 0

        #print("Xlist is " + str(x_list))

        for epoch in range(self.epochs):

            for i in range(len(x_list)):
                z = self.dot(x_list, self.weights) #+ self.bias
                y_pred = self.activation(z)
                print("Y pred{}".format(y_pred))
                print(y_pred[i])
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i]) * x_list[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self, x_list):
        z = np.dot(x_list, self.weights) + self.bias
        return self.activation(z)


# import data and put into two lists, train and test

allData = perdata.allData

r.shuffle(allData)

trainData = allData[:800]
testData = allData[800:]


#These return tuples, I don't trust it...
#x_list_train, y_train = zip(*trainData)
#x_list_test, y_test = zip(*testData)

#Try this method instead

x_list_train = [x_list[0] for x_list in trainData]
y_train = [x_list[1] for x_list in trainData]

x_list_test = [x_list[0] for x_list in testData]
y_test = [x_list[1] for x_list in testData]

x_list_train_int = [ss.strToInt(i) for i in x_list_train]
x_list_test_int = [ss.strToInt(i) for i in x_list_test]


### IMPORTANT - You need to decide what activation function to use. 
# Heaviside doesn't seem to work with your data type. 






p = Perceptron(0.001, 100)
test_weights = [0] * len(x_list_train_int)

test_dot = p.dot(x_list_train_int[3], test_weights)

print("Test dot is {}".format(test_dot))

model = p.fit(x_list_train_int, y_train)

#pred = p.predict(x_list_test_int)

