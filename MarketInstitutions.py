"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm


def log_util(trade_type, P, prob,  X=1., v1=1., v2=0.):
	if trade_type == "buy":
		return prob * np.log(X + v1 - P) + (1 - prob) * np.log(X + v2 - P)
	elif trade_type == "sell":
		return prob * np.log(X + P - v1) + (1 - prob) * np.log(X + P - v2)
	else:
		return np.log(X)

class SimpleTrader(object):
	def __init__(self, market, prob):
		self.market = market
		self.prob = prob
		self.own_trades = []

	def create_trade(self):
		P = self.market.get_last_price()

		util_buy = log_util("buy", P, self.prob)
		util_sell = log_util("sell", P, self.prob)
		util_none = log_util(None, P, self.prob)
		if util_buy > util_none:
			# Buy
			x = 1.
		elif util_sell > util_none:
			x = -1.
		else:
			return None
		self.own_trades.append(x)
		self.market.submit_trade(x)

#		if self.mu > p:
#			# Buy!
#			x = 1.
#		else:
#			x = -1.
#		self.own_trades.append(x)
#		self.market.submit_trade(x)

	def update_expectations(self):
		pass


				


class SimpleMarket(object):
	def __init__(self, price_history):
		self.price_history = price_history
		self.inventory = 0
		self.inventory_history = []

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		self.inventory += - x
		self.inventory_history.append(self.inventory)

		if np.abs(self.inventory) > 5:  # Update price if inventory grows too much
			self.update_price()


	def update_price(self, g=0.01):
		p = self.get_last_price()
		inventory_grow = np.abs(self.inventory_history[-1]) > np.abs(self.inventory_history[-2])

		if self.inventory > 0 and inventory_grow:
			# Positive inventory means a lot of sellers - price should go down
			new_price = (1 - g) * p
			self.price_history.append(new_price)
		elif self.inventory < 0 and inventory_grow:
			# Negative inventory means a lot of buyers- prices go up
			new_price = (1 + g) * p
			self.price_history.append(new_price)
			
		else:
			return None

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
			#print self.market.get_last_price(), self.market.inventory
