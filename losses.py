"""
Implementations of various loss functions
"""
from typing import List, Union
import numpy as np


def mse(
	true_values: List[float], 
	predicted_values: List[float]
):
	"""Mean squared error loss function.

	Defined as the average of the sum of the square differences between
	true and predicted value pairs.
	Args:
		true_values: Array of true labels
		predicted_values: Array of predicted labels from a model.

	Returns:
		A float of the error between true and predicted values.
	"""
	assert len(true_values) == len(predicted_values), \
		"Shape mismatch! True and Predicted values must be arrays of the same length."
	if isinstance(true_values, np.ndarray):
		return float(np.nanmean((true_values - predicted_values) ** 2))
	else: # list
		return sum([(t - p) ** 2 for t, p in zip(true_values, predicted_values)]) / len(true_values)


def mse_delta(
	true_values: np.ndarray, 
	predicted_values: np.ndarray
):
	"""Derivative of the mean squared error function.
	The derivative is taken with respect to the predicted value.
	Therefore, if t is the true and predicted values,
	differentiating (t - p)^2 by chain rule yields 2*(-1)*(t - p)

	Args:
		Numpy arrays of the true and predicted values.
	"""
	return np.nanmean(-2 * (true_values - predicted_values))
	# or return np.nanmean(2 * (predicted_values - true_values))


def bce_loss():
	"""Binary Cross Entropy Loss function.
	"""
	pass
