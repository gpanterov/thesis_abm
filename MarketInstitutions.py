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
		self.own_trades = []

	def create_trade(self):
		p = self.market.get_last_price()
		if self.mu > p:
			# Buy!
			x = 1.
		else:
			x = -1.
		self.own_trades.append(x)
		self.market.submit_trade(x)

class SimpleMarket(object):
	def __init__(self, price_history):
		self.price_history = price_history
		self.inventory = 0

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		self.inventory += - x
		if np.abs(self.inventory) > 10:  # Update price if inventory grows too much
			self.update_price()


	def update_price(self, g=0.01):
		p = self.get_last_price()

		if self.inventory > 0:
			# Positive inventory means a lot of sellers - price should go down
			new_price = (1 - g) * p
		else:
			# Negative inventory means a lot of buyers- prices go up
			new_price = (1 + g) * p

		self.price_history.append(new_price)

			

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

