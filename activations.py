# Implementations of various activation functions.
from typing import List, Union
from functools import wraps
import math


def sigmoid(x: Union[float, List[float]]):
	"""The sigmoid activation function.

	It is a function whose graph follows a logistic function.
	It's defined as, delta(x) = 1 / (1 + e^(-x)), where x is the input value
	Function domain: [0, 1]
	Args:
		x: a float or a list of float values.
	"""
	if isinstance(x, list):
		return [1 / (1 + math.exp(-i)) for i in x]
	elif isinstance(x, float):
		return 1 / (1 + math.exp(-x))
	else:
		raise ValueError("Input must be a float or a list of float values.")


def relu(x):
	"""Relu activation function.
	
	"""
	raise NotImplementedError


def tanh(x: List[float]):
	"""Hyperbolic tangent activation function.

	It is defined as, delta(x) = 
	"""
	raise NotImplementedError


def softmax(x: List[float]):
	"""Softmax activation function.
	
	It is defined as f(x) = e^z_i / sum_i_j(e^x_i)
	"""
	raise NotImplementedError


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
