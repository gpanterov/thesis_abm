"""
Module for trader classes

"""

import numpy as np
from scipy.optimize import minimize

def exponential_utility(wealth, a):
	return 1 - np.exp( - a * wealth)


def traderEU_discrete(trade_size, market_price, prob, util_func):
	"""
	Expected utility of a trader with discrete probabilities
	Parameters:
	-----------
	trade_size: float
		Number of shares to buy or sell (negative)
	market_price: float
	prob: array-like
		An array of the probabilities for the price of the asset
	util_func: lambda
		Utility function of the trader. 
		Takes expected wealth/profit as only argument

	Returns:
	--------
	expected_utility: float
	"""

	# Create support for the probability space
	S = np.linspace(0.01, 0.99, len(prob))
	P = np.array(prob)

	# Create vector of terminal profit values 
	# for each possible price realization
	Profit = trade_size * (S - market_price)

	expected_utility = np.sum(P * util_func(Profit))
	return expected_utility

def MaxEU_discrete_trader(market_price, prob, util_func):
	""" 
	Maximize utlity for discrete trader.
	Returns:
	-------
	trade_size	
 	"""
	obj_func = lambda x: - traderEU_discrete(x, market_price, prob, util_func)
	res = minimize(obj_func, 10, method='nelder-mead')
	return res.x


	


def traderEU_normal_exp(trade_size, market_price, mu_price, sigma_price, a):
	"""
	Expected utility of a trader with exponential utility and
	normally distributed price
	References:
	----------
	[1] http://www.tau.ac.il/~spiegel/teaching/corpfin/mean-variance.pdf
	"""
	# 1) Create the mean and variance of the expected profits
	mu_profit = trade_size * (mu_price - market_price)
	# Using that Var(aX) = a**2 * Var(x)
	var_profit = (trade_size ** 2 * sigma_price ** 2)
	expected_utility = 1 - np.exp( - a * (mu_profit - a * var_profit / 2.))
	return expected_utility






def AR(params):
	params = np.array(params)
	n = len(params)
	P = np.random.normal(0.5,0.05, (n+1,))
	P = np.linspace(0.4, 0.42, n+1)
	R = np.log(P[1:]) - np.log(P[:-1])
	R = list(R)
	P = list(P)
	for i in range(50):
		r = np.sum(params*R[-n:]) + np.random.normal(0,0.01)
		R.append(r)
		p = P[-1] * (1 + r)
		P.append(p)
	return P[n+1: ], R[n:]

class DiscreteTrader(object):

	def __init__(self, market, prob, pop_size, util_func):
		self.market = market
		self.prob = np.array(prob)
		self.pop_size = pop_size
		self.deal_sizes = []
		self.deal_prices = []
		self.deal_times = []
		self.util_func = util_func


		self.trader_name = "Discrete Trader"


	def create_trade(self):
		# Maximize utility to determine trade size and direction
		market_price = self.market.get_last_price()
		obj_func = lambda x: - traderEU_discrete(x, 
							market_price, self.prob, self.util_func)
		x0 = 1.
		res = minimize(obj_func, x0, method = 'nelder-mead')
		x = res.x
		# Record trade
		self.deal_sizes.append(x)
		self.deal_prices.append(market_price)
		# Submit trade to market
		self.market.submit_trade(x)

	def update_expectations(self):
		pass

class BaseContinuousTrader(object):
	def create_trade(self):

		market_price = self.market.get_last_price()
		x = (self.mu - market_price) / (self.a * self.sig**2)

		# Record trade
		self.deal_sizes.append(x)
		self.deal_prices.append(market_price)
		self.deal_times.append(self.market.time)
		# Submit trade to market
		self.market.submit_trade(x, self.trader_name)

	def update_expectations(self):
		pass

class IntelligentTrader(BaseContinuousTrader):
	""" Trader with continuous price distribution"""
	def __init__(self, market, pop_size, mu, sig, a, trader_name):
		self.market = market
		self.pop_size = pop_size
		self.mu = mu
		self.base_mu = mu
		self.sig = sig
		self.a = a

		self.deal_sizes = []
		self.deal_prices = []
		self.deal_times = []


		self.trader_name = trader_name
	def update_expectations(self):
		self.mu = self.base_mu #+ np.random.normal(0,0.01)

class NoiseTrader(BaseContinuousTrader):
	def __init__ (self, market, pop_size, ar_params, sig, a, trader_name):
		self.market = market
		self.pop_size = pop_size
		self.ar_params = np.array(ar_params)
		self.a = a
		self.sig = sig

		self.deal_sizes = []
		self.deal_prices = []
		self.deal_times = []
		self.trader_name = trader_name

	def update_expectations(self):
		if self.market.time % 5 == 0:
			self.u = np.random.normal(0, 0.1)
		n = len(self.ar_params)
		assert len(self.market.price_history) >= n + 1
		P = np.array(self.market.price_history[-n - 1:])
		R = np.log(P[1:]) - np.log(P[:-1])
		r = np.sum(self.ar_params * R)
		self.mu = P[-1] * (1 + r) + np.random.normal(0, 0.01) + self.u
