# Implementations of various activation functions.
from typing import Union
from functools import wraps
import math
import numpy as np



def sigmoid(x: Union[float, np.ndarray]):
	"""The sigmoid activation function.

	It is a monotonic function (entirely non-decreasing or non-increasing) 
	whose graph follows a logistic function.
	It's defined as, delta(x) = 1 / (1 + e^(-x)), where x is the input value

	Function domain: [0, 1]

	Args:
		x: a float or a list of float values.

	Returns:
		List or float of sigmoid transformed values 
	"""
	if isinstance(x, np.ndarray):
		return 1 / (1 + np.exp(-x))
	elif isinstance(x, float):
		return 1 / (1 + math.exp(-x))
	else:
		raise ValueError("Sigmoid input must be a float or a 1D array of float values.")


def relu(x: Union[float, np.ndarray]):
	"""Relu activation function.
	
	Defined as, f(x) = 
	"""
	if isinstance(x, np.ndarray):
		return np.maximum(x, 0)
	elif isinstance(x, float):
		return max(x, 0)
	else:
		raise ValueError("ReLU input must be a float or a 1D array of float values")


def tanh(x: np.ndarray):
	"""Hyperbolic tangent activation function.

	It is defined as, tanh(x) = sinh(x)/cosh(x) = (e^x - e^-x) / (e^x + e^-x)
	= (e^2x - 1) / (e^2x + 1)
	"""
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x: np.ndarray):
	"""Softmax activation function.
	
	It is defined as f(x) = e^z_i / sum_i_j(e^x_i)
	"""
	return np.exp(x) / np.sum(np.exp(x))


def activation(function: str = "sigmoid"):
	"""Activations decorator factory.
	
	Defaults to sigmoid if no activation function is provided.
	Args:
		function: a string specifying activation function to apply.
		Available functions are "sigmoid", "tanh"
	"""
	activation_functions = {
		"sigmoid": sigmoid,
		"relu": relu,
		"tanh": tanh,
		"softmax": softmax
	}

	def decorator(func):
		@wraps(func)
		def activate(*args, **kwargs):
			"""Activate input values."""
			#get result from function to which activation is applied
			result = func(*args, **kwargs)
			# get result after applying activation
			return activation_functions[function](result) 
		return activate

	return decorator
