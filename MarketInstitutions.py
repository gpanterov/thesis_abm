"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

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

def gen_price_distro(noise_trader, mu, sigma2, stdev_mu, stdev_sig):
	if noise_trader:
		mu = np.random.normal(mu, stdev_mu)
		sigma2 = np.random.normal(sigma2, stdev_sig)
		
	assert mu > 0
	assert sigma2 > 0
	return {'mu':mu, 'sigma2':sigma2}
		

sample_distro_params = {'pop_size':1000,
			'endowment': lambda : np.random.normal(10000, 1000),
			'risk_aversion':lambda : np.random.normal(1e-3, 5e-4),
			'noise_trader' : lambda : np.random.binomial(1, 0.3),
			'price_distro': lambda nt: gen_price_distro(nt, 30., 9., 4, 1)}

class Population(object):
	def __init__(self, market, distro_params):
		self.market = market
		self.distro_params = distro_params
	
	def create_population(self):
		self.traders = []
		for i in range(self.distro_params['pop_size']):
			W0 = self.distro_params['endowment']()
			a = self.distro_params['risk_aversion']()
			nt = self.distro_params['noise_trader']()
			price_distro = self.distro_params['price_distro'](nt)

			t = Trader(self.market, W0, a, price_distro)	
			self.traders.append(t)

	def trade(self):
		for t in self.traders:
			t.create_trade()
		


class Market(object):
	def __init__(self, price_history, rf):
		self.price_history = price_history
		self.Qs = []
		self.Qd = []
		self.rf = rf
		self.buy_trades = []
		self.sell_trades = []

	def estimate_supply_demand(self, pop, N, max_price=100.):
		for i in range(N):
			print "Simulation number ", i
			p = np.random.normal(30., 5.)
			self.price_history.append(p)
			pop.create_population()
			pop.trade()
			self.Qs.append(-np.sum(self.sell_trades))
			self.Qd.append(np.sum(self.buy_trades))
			
			
		Qs = np.array(self.Qs[-N : ])
		Qd = np.array(self.Qd[-N : ])
		P = np.array(self.price_history[-N :])
		print Qd
		print Qs
		X = sm.add_constant(np.log(P))
		model_s = sm.OLS(np.log(Qs), X).fit()
		model_d = sm.OLS(np.log(Qd), X).fit()

		self.Qs_params = model_s.params
		self.Qd_params = model_d.params
		
		# Delete generated values
		self.Qs[-N : ] = []
		self.Qd[-N : ] = []
		self.price_history[-N : ] = []
		return model_s, model_d
	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		#print "Submitting trade of size", x
		if x > 0:
			self.buy_trades.append(x)
		if x < 0:
			self.sell_trades.append(x)

	def update_price(self):
		"""
		The market maker updates the price based on the supply and demand
		estimates. THe market maker upda

		model_s.params = array([ 13.69711137,  -0.19148352])
		model_d.params=array([ 13.72370539,   0.08500943])

		"""
		Qs = - np.sum(self.sell_trades)
		Qd = np.sum(self.buy_trades)
		p = self.get_last_price()
		
		excess_supply = (Qs - Qd) / (Qs + 1)
		new_price = (1 - excess_supply) * p 
		self.price_history.append(new_price)
		print "-" * 50
		print "Old price is: ", p
		print "Qs: ", Qs
		print "Qd: ", Qd
		print "New price: ", new_price
		self.sell_trades = []
		self.buy_trades = []

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
				options={'disp': False})
		return res

	def create_trade(self):
		rf = self.market.rf
		p = self.market.get_last_price()
		res = self.maximize_utility(p, rf)
		x = res['x'][0]
		if x > 2.5 or x < -2.5:
			self.market.submit_trade(x)

class SimpleTrader(Trader):
	def __init__(self, market, mu):
		self.market = market
		self.mu = mu

	def create_trade(self):
		p = self.market.get_last_price()
		if self.mu > p:
			# Buy!
			x = 1.
		else:
			x = -1.
		self.market.submit_trade(x)

class SimpleMarket(Market):
	def __init__(self, price_history):
		self.price_history = price_history
		self.Qs = []
		self.Qd = []
		self.buy_trades = []
		self.sell_trades = []

class SimplePopulation(Population):

	def __init__(self, market, pop_size, mu, stdev_noise, noise_prob):
		self.market = market
		self.pop_size = pop_size
		self.mu = mu
		self.stdev_noise = stdev_noise
		self.noise_prob = noise_prob

	def create_population(self):
		self.traders = []
		for i in range(self.pop_size):
			nt = np.random.binomial(1, self.noise_prob)
			if nt == 1:
				mu = self.mu
			else:
				mu = np.random.normal(self.mu, stdev_noise)

			assert mu > 0
			
			t = SimpleTrader(self.market, mu)	
			self.traders.append(t)


