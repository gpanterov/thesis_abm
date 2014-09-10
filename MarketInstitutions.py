"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np

class Population(object):
	traders = []

class Market(object):
	price_history = []

	def take_orders(self):
		pass
	def clear_market(self):
		pass

def expected_utility1(a, x0, p0, x, p, W0, rf, mu, sigma2):
	"""
	Returns expected exponential utility function when the asset
	follows normal distribution
	Parameters
	----------
	a: float
		Risk aversion parameter
	x0: array
		Initial positions in asset
	p0: array
		Prices at which initial positions were made
	x: float
		Additional (new) position in asset
	p: float
		Current asset price
	W0: float
		Initial cash
	rf: float
		1 + risk-free rate of return
	mu: float 
		Expected return of asset
	sigma2: float
		Variance of asset return

	Notes
	-----
	Expected utility is
	E(U) = -np.exp(-a * (mu_C - 0.5 * a * sigma2_C)
	where mu_c is the mean of the final wealth (C) and sigma2_c is the variance
	of the final wealth.
	Final wealth (C) is given by:
	C = np.sum(x * V - np.sum(x0 * p0) + x * (v - p) + (W0 - x * p) * rf
	where V is the realized price of the asset. V is normally distributed with mean mu and var sigma2

	References
	----------
	http://www.tau.ac.il/~spiegel/teaching/corpfin/mean-variance.pdf
	http://en.wikipedia.org/wiki/Exponential_utility

	"""
	A = np.sum(x0 * p0)
	B = (W0 - x * p) * rf

	# Expected value of C**2
	EC2 = (sigma2 + mu) * (np.sum(x0) + x) ** 2 + \
			2 * mu * (x + np.sum(x0)) * (B - A - p * x) + \
			(A - B + p * x) ** 2
	mu_C = mu * np.sum(x0) - np.sum(x0 * p) + x * mu - x * p + B
	sigma2_C = EC2 - mu_C**2
	EU = -np.exp(-a * (mu_C - 0.5 * a * sigma2_C))
	return EU


class Trader(object):
	def __init__(self, endowment, risk_aversion, price_distro):
		self.endowment = endowment
		self.risk_aversion = risk_aversion
		self.price_distro = price_distro
		self.current_positions = []
		self.current_cash = endowment

	def expected_utility(self):
		x0 = np.sum(self.current_positions)
		
		
