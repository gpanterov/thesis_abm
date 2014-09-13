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
		



		


class Market(object):
	def __init__(self, price_history, rf):
		self.price_history = price_history
		self.Qs = []
		self.Qd = []
		self.rf = rf
		self.buy_trades = []
		self.sell_trades = []
		self.inventories = []

	def end_trading_round(self):
		self.sell_trades = []
		self.buy_trades = []

	def estimate_supply_demand(self, pop, N):
		for i in range(N):
			if i % 10 == 0:
				print "Simulation number ", i
			p = np.random.normal(30., 5.)
			self.price_history.append(p)
			pop.create_population()
			pop.trade()
			self.Qs.append(-np.sum(self.sell_trades))
			self.Qd.append(np.sum(self.buy_trades))
			self.end_trading_round()
			
			
		Qs = np.array(self.Qs[-N : ])
		Qd = np.array(self.Qd[-N : ])
		P = np.array(self.price_history[-N :])
		X = sm.add_constant(P)
		model_s = sm.OLS(Qs, X).fit()
		model_d = sm.OLS(Qd, X).fit()

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
		self.end_trading_round()



class SimpleTrader(object):
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
		self.expectedQs = []
		self.expectedQd = []
		self.buy_trades = []
		self.sell_trades = []
		self.inventories = []


	def calculate_equilibrium_price(self, pop, N=100):
		self.prices = []
		for i in np.arange(1, N, 1):
			if i % 10 == 0:
				print "Simulation number ", i
			p = float(i)
			self.price_history.append(p)
			self.prices.append(p)
			pop.create_population()
			pop.trade()
			self.Qs.append(-np.sum(self.sell_trades))
			self.Qd.append(np.sum(self.buy_trades))
			self.end_trading_round()
			
			
		Qs = np.array(self.Qs)
 		Qd = np.array(self.Qd)

		self.base = np.column_stack ((np.array(self.prices), (Qs - Qd)**2))
		indx = np.argsort(self.base[:, 1])
		equilibrium_price = self.base[:, 0][indx[0]]	
		# Delete generated values
		self.Qs = []
		self.Qd = []
		self.price_history = [equilibrium_price]

	def update_price(self):
		"""
		The market maker updates the price based on the supply and demand
		estimates. THe market maker upda
		
		(Log)
		Qd_params = [34.7665, -7.986281]
		Qs_params = [-21.01799, 8.49184]

		(Levels)
		Qs_params = [-16004.9490137 ,    701.58337161]
		Qd_params = [ 26004.9490137 ,   -701.58337161]
		"""
		Qs = - np.sum(self.sell_trades)
		Qd = np.sum(self.buy_trades)

		self.Qs.append(Qs)
		self.Qd.append(Qd)

		p = self.get_last_price()
	
#		self.inventories.append( Qs - Qd)
#		I = np.sum(self.inventories)	
#		new_price = (I - (Qd_params[0] - Qs_params[0])) / \
#					(Qd_params[1] - Qs_params[1])
		g = 0.02


		if Qs > Qd:
			# more supply - price should go down
			new_price = (1 - g) * p
		else:
			# more demand - prices go up
			new_price = (1 + g) * p

		self.price_history.append(new_price)
		print "-" * 50
		print "Old price is: ", p
		print "Qs: ", Qs
		print "Qd: ", Qd
		#print "Inventory of the market maker: ", I
		print "New price: ", new_price
		self.end_trading_round()

			

class SimplePopulation(object):

	def __init__(self, market, pop_size, mu, stdev_mu, stdev_noise, noise_prob):
		self.market = market
		self.pop_size = pop_size
		self.mu = mu
		self.stdev_noise = stdev_noise
		self.noise_prob = noise_prob
		self.stdev_mu = stdev_mu
	def create_population(self):
		self.traders = []
		for i in range(self.pop_size):
			nt = np.random.binomial(1, self.noise_prob)
			if nt != 1:
				mu = np.random.normal(self.mu, self.stdev_mu)
			else:
				p = self.market.get_last_price()
				mu = np.random.normal(p, self.stdev_noise)
				mu = np.max([5., mu])


			
			t = SimpleTrader(self.market, mu)	
			self.traders.append(t)

	def trade(self):
		for t in self.traders:
			t.create_trade()
