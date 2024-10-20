"""
Multi Layer Perceptron (FeedForward Neural Network).
It is a network of multiple fully connected neurons/perceptrons.
"""
from typing import List, Any
import numpy as np
from activations import relu, sigmoid


def perceptron_np(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray):
	"""Perceptron in numpy."""
	return np.dot(inputs, weights) + bias


class MLP:
	"""Multi Layer Perceptron."""
	def __init__(
		self, 
		input_dim: int,  
		hidden_dim: int,
		out_dim: int, 
	) -> None:
		self.input_dim = input_dim
		self.out_dim = out_dim
		self.hidden_dim = hidden_dim
		self.w1 = np.random.uniform(low=0.0, high=0.5, size=(input_dim, hidden_dim))
		self.w2 = np.random.uniform(low=0.0, high=0.5, size=(hidden_dim, out_dim))
		self.bias = np.random.random(size=out_dim)
	
	def forward(self, x: np.ndarray) -> np.ndarray:
		"""Forward pass."""
		layer1 = perceptron_np(x, self.w1, self.bias)
		x1 = relu(layer1)
		x2 = perceptron_np(layer1, self.w2, self.bias)
		return sigmoid(x2)

	def __call__(self, x):
		return self.forward(x)




if __name__=="__main__":
	or_circuit = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
	in_dim = or_circuit.shape[1]
	model = MLP(in_dim, 10, 1)
	out = model(or_circuit)
	print(f"Output probabilities: \n{out}")
