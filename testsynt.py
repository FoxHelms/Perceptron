weights = [0,0,0,0,0]

ugh1 = [1,2,3]



ugh2 = [6,7,8,9,10]
ugh3 = [1,1,1,1,1]

seven = 7

ugh4 = [sum(x) for x in zip(ugh2, ugh3)]

print(ugh4)

weights[1:] = [seven * ugh for ugh in ugh2]

#print(weights)

training_inputs = [[1,2,3],[4,5,6],[7,8,9]]
labels = [1,0,1]

'''
for inputs, label in zip(training_inputs,labels):
    print("inputs {}".format(inputs))
    print("label {}".format(label))
'''





