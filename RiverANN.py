from itertools import count
from math import exp
from random import random
from random import seed

data = [[10.4, 4.393, 9.291, 26.1, 0, 0, 0,	4, 24.86],
[9.95, 4.239, 8.622, 24.86, 0, 0, 0.8, 0, 23.6],
[9.46, 4.124, 8.057, 23.6, 0, 0, 0.8, 0, 23.47],
[9.41, 4.363, 7.925, 23.47, 2.4, 24.8, 0.8, 61.6, 60.7],
[26.3, 11.962, 58.704, 60.7, 11.2, 5.6, 33.6, 111.2, 98.01]]

seed(1)

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

def deNormalizeOutput(value, min, max):
    return((value-0.1)*(max-min)/0.8)+min
 
# Rescale dataset columns to the range 0.1-0.9
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = 0.8*(row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])+0.1



def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row 
    for layer in network:
        new_inputs = []
        count = 0
        for neuron in layer:

            activation = activate(neuron['weights'], inputs)
            print("activation: ",activation)

            if count == 0:
                neuron['output'] = transfer(activation)
            else:
                neuron['output'] = activation
            new_inputs.append(neuron['output'])
            #count+=1
        inputs = new_inputs
        #print("layer",layer)
    return inputs

#network is a list of arrays (layers) and neurons are dictionaries
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    #create layers 
    #weights initialized to random values 
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def showNetwork(NN):
    for layer in NN:
        print("layer: ", layer )



#-1 because last input = expected value
nInputs = len(data[0])-1
nOutputs = 1
network = initialize_network(nInputs, 2, nOutputs)

expectedArr=[]
for i in range (0,len(data)):
    expectedArr.append(data[i][-1])

expMin= min(expectedArr)
expMax = max(expectedArr)

#showNetwork(network)
minmax = dataset_minmax(data)
normalize_dataset(data, minmax)
#print(data)
for row in data:
    #print(row[:-1])
    output = forward_propagate(network, row[:-1])
    print(deNormalizeOutput(output[0], expMin,expMax))

