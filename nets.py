import numpy as np
import pandas as pd

class NeuralNet():
	'''
	input layer takes a vector/matrix full of data
	hidden layer is a list that gives the length of hidden layer
	output layer is the dimension of the output layer (based on the model)
	'''

	def __init__(self, input_layer, hidden_layers, output_layer):
		if isinstance(input_layer, np.ndarray):
			self.input_layer = input_layer
		
		if isinstance(hidden_layers, list):
			self.hidden_layers = hidden_layers
		
		if isinstance(output_layer, int):
			self.output_layer = output_layer


	'''Returns list of empty theta matrices based on dimensions'''
	def initialize_theta(self, epsilon):
		input_length = np.size(self.input_layer, 0)
		layer_dims = [input_length] + self.hidden_layers + [self.output_layer]
		return [np.random.rand(layer_dims[i+1], layer_dims[i]) * ((2 * epsilon) - epsilon) for i in range(len(layer_dims[0:-1]))]

	'''Returns np.ndarray of outputs using sigmoid function'''
	def sigmoid(self, z):
		assert isinstance(z, np.ndarray), 'input must be an np.ndarray'
		return 1 / (1 + np.exp(-z))

	'''Returns results of each layer of neural net until computing final output function'''
	def compute_output(self, theta, n):
		a = []
		for i in range(0, n):
			if i == 0:
				temp_z = np.dot(theta[i], self.input_layer)
				temp_a = self.sigmoid(-temp_z)
				a.append(temp_a)
			else:
				temp_z = np.dot(theta[i], a[-1])
				temp_a = self.sigmoid(-temp_z)
				a.append(temp_a)
			return a

