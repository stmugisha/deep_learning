"""
Implementations of various loss functions
"""
from typing import List, Union
import numpy as np


def mean_squared_error(
	true_values: List[Union[float, int]], 
	predicted_values: List[Union[float, int]]
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

