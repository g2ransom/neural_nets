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
		layer_dims = [input_length] + self.hidden_layers
		return [np.random.rand(layer_dims[i+1], layer_dims[i]) * ((2 * epsilon_init) - epsilon_init) for i in range(len(layer_dims[0:-1]))]