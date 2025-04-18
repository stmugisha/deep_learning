"""
Backpropagation implementation.
A neural network learns by means of updates to its weights for each data processed by  
inorder to minimize a given objective function.
The mechanism of updating network weights is what is termed as backpropagation.
"""
import numpy as np
from activations import relu, sigmoid
from mlp import perceptron_np
from losses import mse, mse_delta


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