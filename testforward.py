
'''
numpy: dot product, "e to the power of" method
perdata: my list of tuples
random: shuffle method
stringtoint: convert string to integer
re: regular expression substitution method ** idu this well enough
'''

import numpy as np
import perdata
import random as r
import stringtoint as ss
import re
from stringtoint import strToInt

'''
I'm making  a class so that it's easier to modify my model / best practices. 
'''

class Perceptron:
    # Default weights, biases
    # User defined learning rate and epoch
    def __init__(self, learning_rate, epochs) :
        self.weights = [0] * 100
        self.bias = 1
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Using sigmoid activation function because this is binary clasification problem
    def activation(self,z):
        return 1/(1 + np.exp(-z))

    # I use this dot product to calculate weighted sum but idk if it's necessary. 
    # Sum of piecewise products
    def dot(self, K, L):
        if len(K) != len(L):
            return 0
        return sum(i[0] * i[1] for i in zip(K, L))

    # Calculates weighted sum and passes through sigmoid activation function
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    # Make small changes to weights vector 
    def fit(self, inputs, labels):
        # for every epoch
        for _ in range(self.epochs):
            # create an empty list of labels and guesses, and zero correct guesses
            lblz = []
            gsz = []
            numCorrect = 0
            # combine inputs list and labels list, then iterate through the pairs
            for input_vector, label in zip(inputs, labels):
                # calculate sum of pieceweise products and then add bias
                weighted_sum = self.dot(input_vector, self.weights) + self.bias
                # perform sigmoid activation on weighted sum
                activation = self.activation(weighted_sum)
                # store this FLOAT in guess variable
                guess = activation
                # add the INTEGER answer to the label list
                lblz.append(label)
                # add the guess to the guess list
                gsz.append(guess)
                # If there's less than 0.30 diffeerence between the guess and answer, increment correct answers. 
                correctQ = guess - label
                if abs(correctQ) < 0.30:
                    numCorrect += 1
                # If there is a large discrepency in between the guess and answer, 
                # !!!TRY adding mislabeled inputs to a list and then calculating delta once per EPOCH rather than per INPUT !!!
                elif abs(correctQ) > 0.30:
                    # calculate delta for weights
                    newWeights = [self.learning_rate * ((label - activation) / abs(label - activation)) * n for n in input_vector]
                    # We are adding the new weights to the old weights
                    self.weights = [x+y for x,y in zip(self.weights, newWeights)]
                    # add the learn rate times the difference in answer and guess to the bias
                    self.bias += self.learning_rate * (label - activation)
                
            # difference between answer and guess
            # pos number means false negative
            # neg number means false positive
            nets = [lbl - gs for lbl,gs in zip(lblz,gsz)]
            # Show what epoch we're on
            print("Current epoch: {}".format(_))
            # Show how many correct guesses
            print("number correct: {}".format(numCorrect))
            # Calculate and print the num correct guesses divided by the num of labels
            avgError = numCorrect / len(labels)
            print("Average score: {}".format(avgError))
        # This project returns weights. 
        return self.weights




# import data and put into two lists, train and test

allData = perdata.allData


# shuffle data

r.shuffle(allData)


# Segmenting data
trainData = allData[:1000]
testData = allData[1000:]

# training data is list of first item in tuple
X_train = [X[0] for X in trainData]

# convert all numbers to lowercase
X_train_lower = [d.lower() for d in X_train]
# remove special characters from string
X_train_clean = [re.sub('[^A-Za-z0-9]+', ' ', dirty) for dirty in X_train_lower]
# labels are second item in tuple
y_train = [X[1] for X in trainData]

# convert cleaned, lowered string to int
X_train_int = [ss.strToInt(i) for i in X_train_clean]



# instantiate a perceptron
p = Perceptron(0.1, 100)

# run fit method on training ints and labels
new_weights = p.fit(X_train_int,y_train)

# display trained weights and bias
print(new_weights)
print(p.bias)



# test weights and bias on a user generated phrase 
usrP = input("Enter a phrase here: ")

usrD = ss.strToInt(usrP)

# perform np dot product and sigmoid activation on new string and WHAT WEIGHTS?!?!?!
res = p.predict(usrD)

print(res)

