from itertools import count
from math import exp
import random
import pandas as pd

#extract of data from excel file| last value = expected value
data = [[10.4, 4.393, 9.291, 26.1, 0, 0, 0,	4, 24.86],
[9.95, 4.239, 8.622, 24.86, 0, 0, 0.8, 0, 23.6],
[9.46, 4.124, 8.057, 23.6, 0, 0, 0.8, 0, 23.47],
[9.41, 4.363, 7.925, 23.47, 2.4, 24.8, 0.8, 61.6, 60.7],
[26.3, 11.962, 58.704, 60.7, 11.2, 5.6, 33.6, 111.2, 98.01],
[32.1, 10.237, 34.416, 98.01, 0, 0, 1.6, 0.8, 56.99],
[19.3, 7.254, 22.263, 56.99, 5.6, 4, 17.6, 36, 56.66],
[22, 7.266, 29.587, 56.66, 1.6, 0, 1.6, 2.4, 78.1],
[35.5, 8.153, 60.253, 78.1, 14.4, 0.8, 55.2, 104.8, 125.7],
[51, 13.276, 93.951, 125.7, 20.8, 2.4, 76, 136.8, 195.9],
[65.5, 25.561, 69.503, 195.9, 10.4, 16, 12, 28, 125.4],
[32, 20.715, 40.514, 125.4, 7.2, 4, 0.8, 24, 161.5],
[64.3, 36.54, 106.753, 161.5, 39.2, 8.8, 82.8, 118, 204],
[69.1, 28.457, 60.325, 204, 5.6, 1.6, 8, 12.8, 200.6],
[94.6, 33.761, 142.159, 200.6, 36, 12, 72, 72, 234.4],
[99, 38.484, 77.439, 234.4, 8.8, 2.4, 7.2, 40, 160.1],
[58.2, 21.022, 57.103, 160.1, 6.4, 0.8, 0, 21.6, 104.1],
[34.1, 15.191, 42.697, 104.1, 16.8, 2.4, 44, 103.2, 136.4],
[59.8, 16.456, 79.028, 136.4, 7.2, 0, 34.4, 86.4, 137.1],
[48, 18.587, 70.19, 137.1, 6.4, 0, 19.2, 26.4, 124.3],
[42.6, 18.687, 88.489, 124.3, 27.2, 4, 51.2, 80.4, 175.9],
[66.6, 26.501, 74.365, 175.9, 4.8, 2.4, 6.4, 18.4, 128.1],
[41.4, 23.124, 101.231, 128.1, 40.8, 18.4, 62.4, 120.4, 270.3],
[124, 57.874, 138.304, 270.3, 28, 16.8, 7.2, 29.6, 206.5],
[70, 34.921, 62.332, 206.5, 2.4, 0, 0, 2.4, 116.4],
[35.4, 18.518, 43.343, 116.4, 0.8, 0, 0, 9.6, 92.52],
[33.2, 16.112, 39.664, 92.52, 3.2, 7.2, 0.8, 12, 104.3],
[36.3, 18.059, 61.693, 104.3, 0.8, 6.4, 7.2, 84.8, 106],
[33.9, 18.287, 43.674, 106, 0.8, 0, 4.8, 4, 80.07],
[25.8, 15.197, 31.682, 80.07, 0, 0, 0, 2.4, 64.14],
[22.1, 11.821, 26.37, 64.14, 0, 7.2, 7.2, 4, 56.13],
[20.2, 10.624, 21.884, 56.13, 0, 0, 0.8, 4.8, 50.29],
[18.4, 9.607, 19.046, 50.29, 0, 0, 0.8, 0, 43.38],
[16.9, 6.394, 17.181, 43.38, 0, 0, 0, 0, 40.47],
[15.8, 7.112, 15.511, 40.47, 0, 0, 0.8, 0, 37.36],
[14.9, 6.213, 14.625, 37.36, 4.8, 9.6, 0, 2.4, 37.74],
[15.5, 6.641, 14.918, 37.74, 0.8, 7.2, 4.8, 1.6, 38.31],
[15.8, 6.336, 14.1, 38.31, 0, 0, 1.6, 0, 36.22]]

def getData():
    #load all excel data to list
    df = pd.read_excel (r'D:\AI CW data with changes.xlsx')
    dataList = list() 
    dataList = df.values.tolist()

    #turn list of row data into a 2d array 
    for i in range (0, len(dataList)):
        dataList[i] = (str(dataList[i]).split(","))

    #remove unnecesary data excel file had from the list
    dataOnly = [dataList[i][1:10] for i in range (1,len(dataList))]

    #new array to hold riverdata as float values
    floatDataOnly = [[0 for j in range (0,9)] for i in range(len(dataOnly))] 
    for i in range (0,len(dataOnly)):
        for j in range (0,9):
            floatDataOnly[i][j] = float(dataOnly[i][j])

    return floatDataOnly

def splitData(dataSet):
    NormalGroup = []
    LessRainGroup = []
    moreRainGroup = []
    trainingData = []
    validationData = []
    testData= []

    #segment data into expected rain groups
    for i in range(0,len(dataSet)):
        if (((i%365) <= 31) or ((i%365 >= 150) and (i%365 < 304)) or (i%365 > 334)): #jan /jun-oct / dec
            NormalGroup.append(dataSet[i])
        elif ((i%365 >= 32) and (i%365 < 150)) : #feb to may
            LessRainGroup.append(dataSet[i])
        elif ((i%365 >= 304) and (i%365 <= 334)) : #Nov
            moreRainGroup.append(dataSet[i])

    random.seed(1)
    #populate training data set
    for i in range (0,88):
        trainingData.append(moreRainGroup.pop(random.randrange(len(moreRainGroup))))
    for i in range (0,504):
        trainingData.append(NormalGroup.pop(random.randrange(len(NormalGroup))))
    for i in range (0, 272):
        trainingData.append(LessRainGroup.pop(random.randrange(len(LessRainGroup))))

    #populate validation data set
    for i in range (0,18):
        validationData.append(moreRainGroup.pop(random.randrange(len(moreRainGroup))))
    for i in range (0,181):
        validationData.append(NormalGroup.pop(random.randrange(len(NormalGroup))))
    for i in range (0, 100):
        validationData.append(LessRainGroup.pop(random.randrange(len(LessRainGroup))))

    #populate training data
    for i in range (0,18):
        testData.append(moreRainGroup.pop(random.randrange(len(moreRainGroup))))
    for i in range (0,180):
        testData.append(NormalGroup.pop(random.randrange(len(NormalGroup))))
    for i in range (0, 100):
        testData.append(LessRainGroup.pop(random.randrange(len(LessRainGroup))))

    return (trainingData, validationData, testData)

dataTuple = splitData(getData())
#print(dataTuple[2])
trainingData = dataTuple[0]
validationData = dataTuple[1]
testData = dataTuple[2]


#initialize random number
random.seed(1)

# Find the min and max values for each column used for standardising
def dataset_minmax(dataset):
	stats = [[min(column), max(column)] for column in dataset] 
	return stats

#apply inverse of normalizastion for the network output
def deNormalizeOutput(value, min, max):
    return((value-0.1)*(max-min)/0.8)+min
 
# Rescale dataset columns to the range 0.1-0.9
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1): #leave last value (expected value)
			row[i] = 0.8*(row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])+0.1

#normalise exp-output to allow it to be compared to the FP output
def normalizeExp_value(value):
    return 0.8*(value - expMin) / (expMax - expMin)+0.1

#caluclate weighted sum of neuron for activation
def activate(weights, inputs):
	activation = weights[-1] #bias
	for i in range(len(weights)-1): #leaves out last input in row which is the expected output
		activation += weights[i] * inputs[i]
	return activation

#sigmoid function used to help get output for node/neuron
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))
    
# Forward propagate through network for an output
def forward_propagate(network, row):
    inputs = row  
    #go through each layer
    for layer in network:
        new_inputs = [] #hold inputs for next layer
        for neuron in layer:
            #generate output for neuron
            activation = activate(neuron['weights'], inputs) 
            neuron['output'] = transfer(activation)  
            #add neuron output to list          
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    #go through layers in reverse
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1: #in hidden layer/s 
			for j in range(len(layer)): #for each neuron
				error = 0.0
                #take neuron errors from the layer ahead of current 
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else: #in output layer
			for j in range(len(layer)):
				neuron = layer[j]
                #add error of each output neuron to to list 
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
            #calculate delta for each neuron
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1] #take out expected value from row
		if i != 0: #if not the first layer use outputs from previous layer instead
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [(row[-1]) for i in range(n_outputs)]
			expected[0] = normalizeExp_value(expected[0])
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

#network is a list of arrays (layers) and neurons are dictionaries
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    #create layers 
    #weights initialized to random values 
    hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
    network.append(hidden_layer)
    output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def showNetwork(NN):
    for layer in NN:
        print("layer: ", layer )



#-1 because last input = expected value
nInputs = len(trainingData[0])-1
nOutputs = 1
network = initialize_network(nInputs, 19, nOutputs)

expectedArr=[]
for i in range (0,len(trainingData)):
    expectedArr.append(trainingData[i][-1])

expMin= min(expectedArr)
expMax = max(expectedArr)

#showNetwork(network)
minmax = dataset_minmax(trainingData)
normalize_dataset(trainingData, minmax)
#print(data)

"""
for row in data:
    #print(row[:-1])
    output = forward_propagate(network, row[:-1])
    print(deNormalizeOutput(output[0], expMin,expMax))

backward_propagate_error(network, expectedArr)
for layer in network:
	print(layer)
"""
for layer in network:
	print(layer)


train_network(network, trainingData, 0.6, 20000, nOutputs)
#for layer in network:
#	print(layer)

def predict(network, row):
	outputs = forward_propagate(network, row)
	return deNormalizeOutput(outputs[0], expMin, expMax)

for row in validationData:
	prediction = predict(network, row)
	#print('Expected=%d, Got=%d' % (row[-1], prediction))
	print('expected=', row[-1], 'got=', prediction)

for layer in network:
	print(layer)