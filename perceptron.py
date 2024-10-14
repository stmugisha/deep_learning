# The perceptron
# The perceptron is an algorithm for learning a binary classifier. 
# A binary classifier is a function that maps its input to binary output.
# It is defined as, f(x) = activation_function(sum_i_j(x_i * w_i) + b), 
# where w_i is the i_th weight and x_i the i_th feature and b the bias
#
# The activation functions are applied to the perceptron as decorators
# to allow for quick experimentation with different activations.
from typing import List
from activations import sigmoid


#@sigmoid 
def perceptron(input: List[List[float]], weights: List[float], bias: float):
	"""Perceptron function implementation.

	A perceptron is defined as f(x) = sum_i_j(x_i * w_i) + b
	"""
	out = []
	for feature in input:
		out.append(
			sum(x * weight) for x, weight in zip(feature, weights) + bias
		)

	return out

