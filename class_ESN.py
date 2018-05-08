import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


class ESN():
	def __init__ (self, n_inputs=1, n_outputs=1, n_reservoir=500, sr = 0.95, sparsity=0, noise=0., feedback =False, 
		verbose=False, random_state = None, f_out=lambda x: x, f_out_inverse=lambda x: x, 
		binary_W = False, binary_states = False) :
		
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_reservoir = n_reservoir
		self.sr = sr
		self.sparsity = sparsity
		self.noise = noise
		self.verbose = verbose
		self.feedback = feedback
		self.f_out = f_out
		self.f_out_inverse = f_out_inverse
		self.binary_W = binary_W 
		self.binary_states = binary_states

		if random_state is None:
			self.random_state_ = np.random.mtrand._rand
		else:
			self.random_state_ = np.random.RandomState(random_state)

		self.create_weights()

	def create_weights(self):
		#random reservoir weights:
		W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
		W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
		
		if self.binary_W:
			W = np.sign(W)
		
		radius = np.max(np.abs(np.linalg.eigvals(W)))
		self.W = W * self.sr / radius

		# random input weights:
		self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
		if self.binary_W:
			self.W_in = np.sign(self.W_in)
		# random feedback (teacher forcing) weights:
		self.W_fb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1
		if self.binary_W:
			self.W_fb = np.sign(self.W_fb)

	def _step(self, x, u, y, feedback=False):
		#creates the preactivation
		if self.feedback:
			S  = (np.dot(self.W, x) + np.dot(self.W_in, u) + np.dot(self.W_fb, y))
		else:
			S = (np.dot(self.W, x) + np.dot(self.W_in, u))

		if self.binary_states:
			return np.sign(S + self.noise*(self.random_state_.rand(self.n_reservoir) - 0.5))
		else:
			return (np.tanh(S) + self.noise*(self.random_state_.rand(self.n_reservoir) - 0.5))
		


	def fit(self, u, y_true, inspect=False, settling = 100, regularization_coef = 0):
		"""
		Collect the network's reaction to training data, train readout weights.

		Args:
		inputs: array of dimensions (N_training_samples x n_inputs)
		outputs: array of dimension (N_training_samples x n_outputs)
		inspect: show a visualisation of the collected reservoir states

		Returns:
		the network's output on the training data, using the trained weights
		"""


		# transform any vectors of shape (x,) into vectors of shape (x,1):
		if u.ndim < 2:
			u = np.reshape(u, (len(u), -1))

		if y_true.ndim < 2:
			y_true = np.reshape(y_true, (len(y_true), -1))

		if self.verbose:
			print("evolving...")
		# step the reservoir through the given input,output pairs:
		x = np.zeros((y_true.shape[0], self.n_reservoir))
		
		for n in range(1, u.shape[0]):
			x[n,:] = self._step( x[n - 1],  u[n, :],  y_true[n - 1, :] )

		# learn the weights, i.e. find the linear combination of collected
		# network states that is closest to the target output
		if self.verbose:
			print("fitting...")

		# include the raw inputs:
		x_ext = np.hstack((x, u))
		# Solve for W_out:
		
		#ridge = linear_model.Ridge(alpha= regularization_coef, fit_intercept = False)
		#ridge.fit(x_ext[settling:, :], self.f_out_inverse( y_true[settling:, :]))
		#self.W_out = np.array(ridge.coef_)


		self.W_out = np.dot(np.linalg.pinv(x_ext[settling:, :]), self.f_out_inverse(y_true[settling:, :] ) ).T

	

		# remember the last state for later:
		self.last_x = x[-1, :]
		self.last_u = u[-1, :]
		self.last_y = y_true[-1, :]

		# optionally visualize the collected states
		if inspect:
			# (^-- we depend on matplotlib only if this option is used)
			plt.figure(figsize=(x.shape[0] * 0.05, x.shape[1] * 0.05))
			plt.title("Training set")
			plt.imshow(x.T, aspect='auto', interpolation='nearest')
			plt.colorbar()
			plt.show()

		if self.verbose:
			print("training error:")
			# apply learned weights to the collected states:
		pred_train = self.f_out(np.dot(x_ext, self.W_out.T))
		if self.verbose:
			print(np.sqrt(np.mean((pred_train[settling:] - y_true[settling:])**2)))
		return pred_train

	def predict(self, u, keep_going=True, inspect = True ):
		"""
		Apply the learned weights to the network's reactions to new input.

		Args:
		inputs: array of dimensions (N_test_samples x n_inputs)
		continuation: if True, start the network from the last training state

		Returns:
		Array of output activations
		"""

		if u.ndim < 2:
			u = np.reshape(u, (len(u), -1))
		n_steps = u.shape[0]

		if keep_going:
			last_x = self.last_x
			last_u = self.last_u
			last_y = self.last_y
		
		else:
			last_x = np.zeros(self.n_reservoir)
			last_u = np.zeros(self.n_inputs)
			last_y = np.zeros(self.n_outputs)

		u = np.vstack( [last_u,u] )
		x = np.vstack( [last_x, np.zeros( (n_steps, self.n_reservoir))])
		y = np.vstack([last_y, np.zeros( (n_steps, self.n_outputs))])

		for n in range(n_steps):
			x[n + 1, :] = self._step(x[n, :], u[n + 1, :], y[n, :])
			y[n + 1, :] = self.f_out( np.dot(self.W_out,  np.concatenate([x[n + 1, :], u[n + 1, :]])) )

		if inspect:
			plt.figure(figsize=(x.shape[0] * 0.05, x.shape[1] * 0.05))
			plt.title("Test set")
			plt.imshow(x.T, aspect='auto', interpolation='nearest')
			plt.colorbar()
			plt.show()

		return y[1:]








