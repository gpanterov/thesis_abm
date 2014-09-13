"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm



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

class SimpleMarket(object):
	def __init__(self, price_history):
		self.price_history = price_history
		self.Qs = []
		self.Qd = []
		self.expectedQs = []
		self.expectedQd = []
		self.buy_trades = []
		self.sell_trades = []
		self.inventories = []

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		#print "Submitting trade of size", x
		if x > 0:
			self.buy_trades.append(x)
		if x < 0:
			self.sell_trades.append(x)

	def end_trading_round(self):
		self.sell_trades = []
		self.buy_trades = []

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
