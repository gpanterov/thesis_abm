"""
9/9/2014
Script contains the market institutions for the market simulation
"""
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import random
from bisect import bisect
import pandas as pd



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

def exp_util(trade_type, P, prob, a=1e-2, X=1., v1=1., v2=0.):
	if trade_type == "buy":
		return prob * ( 1 - np.exp(- a * (X + v1 - P))) + \
					(1 - prob) * (1 - np.exp(- a * (X + v2 - P)))
	elif trade_type == "sell":
		return prob * ( 1 - np.exp(- a * (X + P - v1))) + \
					(1 - prob) * (1 - np.exp(- a * (X + P - v2)))

	else:
		return 1 - np.exp(-a * X)


def compute_RSI(price_history, n):
	prices = np.array(price_history)
	price_changes = np.log(prices[1:]) - np.log(prices[:-1])
	U = (price_changes > 0) * price_changes
	D = (price_changes < 0) * np.abs(price_changes)
	
	Uema = pd.ewma(U, n)
	Dema = pd.ewma(D, n)

	RS = 1. * Uema / Dema
	RSI = 100 - 100. / (1 + RS)
	return RSI


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

		self.trade_incentive = np.max((self.util_buy, self.util_sell, self.util_none)) - \
															 self.util_none
		return self.trade_incentive

	def create_trade(self):
		self.get_trade_incentive(exp_util)

		if self.util_buy == self.trade_incentive + self.util_none:
			# Buy
			x = 1.
		elif self.util_sell == self.trade_incentive +  self.util_none:
			x = -1.
		else:
			x = 0.
			print self.util_buy - self.util_none, self.util_sell - self.util_none

		self.own_trades.append(x)
		self.market.submit_trade(x)

	def update_expectations(self):
		pass

class RSI_Trader(SimpleTrader):
	def __init__(self, market, pop_size, n, overbought_level, oversold_level):
		self.market = market
		self.own_trades = []
		self.pop_size = pop_size
		self.n = n
		self.overbought_level = overbought_level
		self.oversold_level = oversold_level

	def update_expectations(self):
		if len(self.market.price_history) < self.n:
			self.prob = self.market.get_last_price()
			return None

		RSI = compute_RSI(self.market.price_history, self.n)
		if RSI[-1] >= self.overbought_level:
			self.prob = 0.
		elif RSI[-1] <= self.oversold_level:
			self.prob = 1.
		else:
			self.prob = self.market.get_last_price()


class SimpleMarket(object):
	def __init__(self, price_history, max_price=0.95, min_price=0.05):
		self.price_history = price_history[:]
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

class Simulation(object):
	def __init__(self, market, all_traders, util_func = exp_util):
		self.util_func = util_func
		self.market = market
		self.types_of_traders = []
		self.all_traders = all_traders
		self.update_all_traders()

	def update_all_traders(self):
		for trader in self.all_traders:
			trader.update_expectations()

	def run(self, num_trades):
		self.traders_sizes = [t.pop_size for t in self.all_traders]
		for i in range(num_trades):
			trade_incentives = [i.get_trade_incentive(self.util_func) for i in self.all_traders]
			trade_probs = np.array(self.traders_sizes) * np.array(trade_incentives)
			cdf_vals = cdf(trade_probs)
			trader = choice(self.all_traders, cdf_vals)
			trader.create_trade()
			self.update_all_traders()

	def resample_prices(self, sampling_freq=5):
		prices = np.array(self.market.price_history)
		delta = (prices[1:] - prices[0:-1]) / prices[0:-1]
		s = pd.Series(delta)
		freq = s.index / sampling_freq

		returns = s.groupby([freq]).mean()
		
		prices = pd.Series(prices)	
		freq2 = prices.index / sampling_freq
		prices = prices.groupby([freq2]).mean()
		return prices, returns


