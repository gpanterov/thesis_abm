"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import random
from bisect import bisect

def cdf(weights):
	total=sum(weights) * 1.
	result=[]
	cumsum=0
	for w in weights:
		cumsum += w 
		result.append(cumsum/total)
	return result

def choice(population, cdf_vals):
	"""
	Returns a random element of population sampled according
	to the weights cdf_vals (produced by the func cdf)
	Inputs
	------
	population: list, a list with objects to be sampled from
	cdf_vals: list/array with cdfs (produced by the func cdf)
	Returns
	-------
	An element from the list population
	"""
	assert len(population) == len(cdf_vals)
	x = random.random()
	idx = bisect(cdf_vals,x)
	return population[idx]


def log_util(trade_type, P, prob,  X=1., v1=1., v2=0.):
	if trade_type == "buy":
		return prob * np.log(X + v1 - P) + (1 - prob) * np.log(X + v2 - P)
	elif trade_type == "sell":
		return prob * np.log(X + P - v1) + (1 - prob) * np.log(X + P - v2)
	else:
		return np.log(X)

def exp_util(trade_type, P, prob, a=1e-3, X=1., v1=1., v2=0.):
	if trade_type == "buy":
		return prob * ( 1 - np.exp(- a * (X + v1 - P))) + \
					(1 - prob) * (1 - np.exp(- a * (X + v2 - P)))
	elif trade_type == "sell":
		return prob * ( 1 - np.exp(- a * (X + P - v1))) + \
					(1 - prob) * (1 - np.exp(- a * (X + P - v2)))

	else:
		return 1 - np.exp(-a * X)

class SimpleTrader(object):
	def __init__(self, market, prob, pop_size):
		self.market = market
		self.prob = prob
		self.own_trades = []
		self.pop_size = pop_size
		self.get_trade_incentive(exp_util)

	def get_trade_incentive(self, util_func):
		P = self.market.get_last_price()
		self.util_buy = util_func("buy", P, self.prob)
		self.util_sell = util_func("sell", P, self.prob)
		self.util_none = util_func(None, P, self.prob)

		self.trade_incentive = np.max((self.util_buy, self.util_sell, self.util_none)) / \
															 self.util_none - 1
		return self.trade_incentive
	def create_trade(self):
		self.get_trade_incentive(exp_util)

		if self.util_buy == (self.trade_incentive + 1) * self.util_none:
			# Buy
			x = 1.
		elif self.util_sell == (self.trade_incentive + 1) * self.util_none:
			x = -1.
		else:
			return None
		self.own_trades.append(x)
		self.market.submit_trade(x)

	def update_expectations(self, n=10):
		prices = np.array(self.market.price_history)

		if len(prices) < 3:
			self.prob = np.random.binomial(1, 0.5)
		else:
			price_change = prices[1:] - prices[:-1]
			num_hikes = 1. * np.sum(price_change[-n:] > 0) / n
			self.prob = np.random.binomial(1, num_hikes)
			if np.sum(price_change[-n:]) == 0:
				self.prob = np.random.binomial(1, 0.5)

class SimpleMarket(object):
	def __init__(self, price_history, max_price=0.95, min_price=0.05):
		self.price_history = price_history
		self.inventory = 0
		self.inventory_history = []
		self.max_price = max_price
		self.min_price = min_price

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		self.inventory += - x
		self.inventory_history.append(self.inventory)

		if np.abs(self.inventory) > 1:  # Update price if inventory grows too much
			self.update_price()
		else:
			self.price_history.append(self.get_last_price())


	def update_price(self, g=0.01):
		p = self.get_last_price()
		inventory_grow = np.abs(self.inventory_history[-1]) > np.abs(self.inventory_history[-2])

		if self.inventory > 0 and inventory_grow:
			# Positive inventory means a lot of sellers - price should go down
			new_price = (1 - g) * p
		elif self.inventory < 0 and inventory_grow:
			# Negative inventory means a lot of buyers- prices go up
			new_price = (1 + g) * p
			
		else:
			new_price = p
		new_price = np.max((self.min_price, new_price))
		new_price = np.min((self.max_price, new_price))
		self.price_history.append(new_price)

