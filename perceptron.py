# The perceptron
# The perceptron is an algorithm for learning a binary classifier. 
# A binary classifier is a function that maps its input to binary output.
# It is defined as, f(x) = activation_function(sum_i_j(x_i * w_i) + b), 
# where w_i is the i_th weight and x_i the i_th feature and b the bias
#
# The activation functions are applied to the perceptron as decorators
# to allow for quick experimentation with different activations.
from typing import List, Any
from activations import sigmoid, relu


def perceptron(input: List[List[Any]], weights: List[float], bias: float):
	"""Perceptron function implementation.

	A perceptron is defined as f(x) = activation_func(sum_i_j(x_i * w_i) + b)
	Args:
		input: a list of input features
		weights: a list of weights. Must be of same length as input_i
		bias: bias term
	
	Returns:
		A list of (features_i_j * weights_i + bias) values
	"""
	out = []
	for feature in input:
		out.append(
			sum((x * weight) for x, weight in zip(feature, weights)) + bias
		)

	return out


if __name__=="__main__":
	or_circuit = [[1, 1], [0, 1], [1, 0], [0, 0]]
	weights = [0.5, 0.5]
	bias = 1.0
	res = sigmoid(perceptron(or_circuit, weights, bias))
	print(f"Output probabilities: {res}")
