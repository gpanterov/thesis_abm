"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize


def expected_utility1(a, x0, p0, x, p, W0, rf, mu, sigma2):
	"""
	Returns expected exponential utility function when the asset
	follows normal distribution. Traders can trade repatedly and keep
	a portfolio.
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
	EC2 = (sigma2 + mu**2) * (np.sum(x0) + x) ** 2 + \
			2 * mu * (x + np.sum(x0)) * (B - A - p * x) + \
			(A - B + p * x) ** 2
	mu_C = mu * np.sum(x0) - A + x * mu - x * p + B
	sigma2_C = EC2 - mu_C**2
	EU = -np.exp(-a * (mu_C - 0.5 * a * sigma2_C))
	return EU

def expected_utility2(a, x, p, W0, rf, mu, sigma2, trader_type):
	"""
	Returns expected exponential utility function when the asset
	follows normal distribution. Traders trade only once
	Parameters
	----------
	a: float
		Risk aversion parameter
	x: float
		Additional (new) position in asset
	p: float
		Current asset price
	W0: float
		Initial cash
	rf: float
		risk-free rate of return
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
	C =  x * (v - p) + (W0 - x * p) * rf
	where v is the realized price of the asset. 
	v is normally distributed with mean mu and var sigma2

	References
	----------
	http://www.tau.ac.il/~spiegel/teaching/corpfin/mean-variance.pdf
	http://en.wikipedia.org/wiki/Exponential_utility

	"""
	if trader_type == "buyer":
		B = (W0 - x * p) * rf
	elif trader_type == "seller":
		B = (W0 + x * p) * rf
	else:
		print "Wrong Type"
		raise
	# Expected value of C**2
	EC2 = (sigma2 + mu**2) * x**2 + (B - x * p) * (B - x * p + 2 * mu * x)
	mu_C = mu * x - x * p + B
	sigma2_C = EC2 - mu_C**2
	EU = -np.exp(-a * (mu_C - 0.5 * a * sigma2_C))
	return EU


def penalty(x, p, W0, trader_type):
	if trader_type == "buyer":
		#cannot buy more than you have cash
	 	return (p*x - W0 > 0) * 1e3 + (x < 0) * 1e3
	if trader_type == "seller":	
		#cannot sell more than you own
		return (p*x + W0 < 0) * 1e3 + (x >= 0) * 1e3


sample_distro_params = {'pop_size':100,
			'endowment': lambda : np.random.normal(10000, 1000),
			'risk_aversion':lambda : np.random.normal(1e-3, 5e-4),
			'price_distro': lambda : {'mu':30., 'sigma2':9.}}

class Population(object):
	traders = []
	def __init__(self, market, distro_params):
		self.market = market
		self.distro_params = distro_params
	
	def create_population(self):
		for i in range(len(self.distro_params['pop_size'])):
			W0 = self.distro_params['endowment']()
			a = self.distro_params['risk_aversion']()
			price_distro = self.distro_params['price_distro']()
			if trader_indicator == 1:
				trader_type == "buyer"
			else:
				trader_type == "seller"
			
			t = Trader(self.market, W0, a, price_distro, trader_type)	
			self.traders.append(t)
class Market(object):
	def __init__(self, price_history, rf):
		self.price_history = price_history
		self.rf = rf
		self.buy_trades = []
		self.sell_trades = []

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		print "Submitting trade of size", x
		if x > 0:
			self.buy_trades.append(x)
		if x < 0:
			self.sell_trades.append(x)

	def clear_market(self):
		"""
		Qd = a - b * P
		Qs = u + z * P
		"""
		Qs = - np.sum(self.sell_trades)
		Qd = np.sum(self.buy_trades)
		

class Trader(object):
	def __init__(self, market, endowment, risk_aversion, 
					price_distro):
		self.market = market
		self.endowment = endowment
		self.risk_aversion = risk_aversion
		self.price_distro = price_distro

	def maximize_utility(self, p, rf ):
		a = self.risk_aversion
		W0 = self.endowment
		mu = self.price_distro['mu']
		sigma2 = self.price_distro['sigma2']
		if mu >= p:
			trader_type = "buyer"
		else:
			trader_type = "seller"
		obj_func = lambda  x:  - expected_utility2(a, x, p, W0, rf, mu, sigma2, 
						trader_type) + penalty(x, p, W0, trader_type)
		if trader_type == "buyer":
			x_initial = 1.
		if trader_type == "seller":
			x_initial = -1.
		res = minimize(obj_func, x_initial, method = 'nelder-mead',
				options={'disp': True})
		return res

	def create_trade(self):
		rf = self.market.rf
		p = self.market.get_last_price()
		res = self.maximize_utility(p, rf)
		x = res['x'][0]
		if x > 0.5 or x < -0.5:
			self.market.submit_trade(x)

