import numpy as np
import pandas as pd

class NeuralNet():
	'''
	input layer takes a vector/matrix full of data
	hidden layer is a list that gives the length of hidden layer
	output layer is the dimension of the output layer (based on the model)
	'''


	def __init__(self, input_layer, hidden_layers, output_layer, y, epsilon):
		if isinstance(input_layer, np.ndarray) and input_layer.ndim == 2:
			self.input_layer = input_layer
		
		if isinstance(hidden_layers, list) and len(filter(lambda x: x <= 0, hidden_layers)) == 0:
			self.hidden_layers = hidden_layers
		
		if isinstance(output_layer, int) and output_layer > 0:
			self.output_layer = output_layer

		if isinstance(y, np.ndarray) and len(input_layer) == len(y):
			self.y = y

		self.m = len(y)
		
		if epsilon > 0:
			self.epsilon = epsilon

		input_length = np.size(self.input_layer, 1)
		layer_dims = [input_length] + self.hidden_layers + [self.output_layer]
		self.theta = [np.random.rand(layer_dims[i+1], layer_dims[i]+1) * ((2 * self.epsilon) - self.epsilon)
						for i in range(len(layer_dims[0:-1]))]

		self.theta_n = len(self.theta)


	'''Returns np.ndarray of outputs using sigmoid function'''
	def sigmoid(self, z):
		if isinstance(z, np.ndarray):
			return 1 / (1 + np.exp(-z))


	'''Returns results of each layer of neural net until computing final output function'''
	def compute_output(self, input_layer):
		if isinstance(input_layer, np.ndarray):
			a = []
			a.append(input_layer)
			for i in range(self.theta_n):
				if i == 0:
					X = input_layer
					if X.ndim == 1:
						X = np.insert(X, 0, 1, axis=0)
					else:
						X = np.insert(X, 0, 1, axis=1)
						X = np.transpose(X)
					temp_z = np.dot(self.theta[i], X)
					temp_a = self.sigmoid(-temp_z)
					if temp_a.ndim == 1:
						temp_a = np.insert(temp_a, 0, 1, axis=0)
					else:
						temp_a = np.insert(temp_a, 0, 1, axis=1)
					a.append(temp_a)
				elif i == (self.theta_n-1):
					temp_z = np.dot(self.theta[i], a[-1])
					temp_a = self.sigmoid(-temp_z)
					a.append(temp_a)
				else:
					temp_z = np.dot(self.theta[i], a[-1])
					temp_a = self.sigmoid(-temp_z)
					if temp_a.ndim == 1:
						temp_a = np.insert(temp_a, 0, 1, axis=0)
					else:
						temp_a = np.insert(temp_a, 0, 1, axis=1)
					a.append(temp_a)
			return a


	def compute_cost(self, h, lambda_value=0):
		if isinstance(h, np.ndarray):
			J_temp = (1 / self.m) * np.sum(np.multiply(-self.y, np.log(h)) - np.multiply((1 - y), np.log(1-h)))
			if lambda_value == 0:
				lambda_sum = 0
			else:
				lambda_sum = (lambda_value / (2 * self.m)) * sum([np.sum(np.power(self.theta[i], 2)) for i in range(self.theta_n - 1)])
			return J_temp + lambda_sum



	def sigmoid_gradient(self, z):
		if isinstance(z, np.ndarray):
			return np.multiply(self.sigmoid(-z), (1 - self.sigmoid(-z)))


	def backprop(self, learning_rate, iterations=1):
		if isinstance(learning_rate, float):
			theta_grad = []
			theta = self.theta
			for z in range(self.theta_n):
				theta_grad.append(np.zeros(theta[z].shape))
			if  isinstance(iterations, int):
				for epoch in range(0, iterations):
					for i in range(self.m):
						if self.input_layer.ndim == 1:
							a = self.compute_output(self.input_layer[i])
						else:
							a = self.compute_output(self.input_layer[i, :])
						if self.y.ndim == 1:
							yy = self.y[i]
						else:
							yy = self.y[i, :]

						'''compute the error for each layer'''
						dels = []
						for l in reversed(range(self.theta_n)):
							if l == self.theta_n - 1:
								error_l = a[-1] - yy
								del_temp = error_l * self.sigmoid_gradient(np.dot(theta[l], a[l]))
								dels.append(del_temp)
							else:
								del_temp = np.dot(np.transpose(theta[l+1]), dels[-1]) * self.sigmoid_gradient(np.dot(theta[l], a[l]))
								dels.append(del_temp)

							'''compute weight update'''
							theta_grad[l] = theta_grad[l] + np.dot(del_temp, np.transpose(a[l-1]))
							theta[l] = theta[l] - (learning_rate * theta_grad[l])
		return theta
			
			




