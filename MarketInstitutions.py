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
from scipy.optimize import minimize


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

	if np.sum(np.isnan(cdf_vals)) > 0:
		print "Zero incentive (pick one at random)"
		i = np.random.randint(0, len(cdf_vals))
		return population[i]

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

def exp_util_v2(P, price, trade_type, V=[0.1, 0.5, .9], a=1e-2):
	P = np.array(P)
	V = np.array(V)
	if trade_type == "buy":
		return np.sum(P * (1 - np.exp(- a * (V - price))))
	elif trade_type == "sell":
		return np.sum(P * (1 - np.exp(- a * (price - V))))

def CE(P, Q=[0.33,0.33,0.34]):
	P = np.array(P)
	Q = np.array(Q)
	ce =  np.sum(P * np.log(P / Q))

	return ce

def max_ent(price, trade_type, Q):
	obj_func = lambda x: CE(x, Q)
	P0 = [0.3,0.2,0.5]

	cons = ({'type':'ineq',
		'fun':lambda x: exp_util_v2(x, price, trade_type)},
		{'type':'eq',
		'fun':lambda x: np.sum(x) - 1},
	)
	res = minimize(obj_func, P0, method='SLSQP', constraints=cons)
	return res.x


def exp_util_v3(P, price, trade_type, V=[0.1, 0.5, .9], a=1e-2):
	""" Returns zero if utility is negative """
	P = np.array(P)
	V = np.array(V)
	if trade_type == "buy":
		return np.max((0, np.sum(P * (1 - np.exp(- a * (V - price))))))
	elif trade_type == "sell":
		return np.max((0, np.sum(P * (1 - np.exp(- a * (price - V))))))


def exp_util_v4(P, price, trade_type, V=[0.1, 0.5, .9], a=1e-2):
	""" Returns zero if utility is negative """
	P = np.array(P)
	V = np.array(V)
	if trade_type == "buy":
		if np.sum(P * (1 - np.exp(- a * (V - price)))) > 0:
			return 1.
		else:
			return 0.
	elif trade_type == "sell":
		if np.sum(P * (1 - np.exp(- a * (price - V)))) > 0:
			return 1.
		else:
			return 0.


def cons_func(x, price, b, util_func=exp_util_v3):
	denom = b * util_func(x, price, "buy") + b * util_func(x, price, "sell") + \
				0.5 * (1 -b) * util_func([0.,0.,1.], price, "buy") + 0.5*(1 - b)*util_func([1.,0.,0.], price, "sell") 
	fundamental_buys = b * util_func(x, price, "buy") / denom 
	fundamental_sells = b * util_func(x, price, "sell")/denom 

	noise_buys = 0.5*(1 -b) * util_func([0.,0.,1.], price, "buy") / denom
	noise_sells = 0.5*(1 - b)*util_func([1.,0.,0.], price, "sell") / denom
	
	expected_net_trades = fundamental_buys - fundamental_sells + noise_buys - noise_sells
#	return fundamental_buys, fundamental_sells, noise_buys, noise_sells
	return expected_net_trades

def max_ent_v2(price, Q, b, actual, N):
	obj_func = lambda x: CE(x, Q)
	P0 = [0.3,0.2,0.5]
	cons = ({'type':'eq',
		'fun': lambda x: N * cons_func(x, price, b) - actual},
		{'type':'eq',
		'fun':lambda x: np.sum(x) - 1},
	)
	res = minimize(obj_func, P0, method='SLSQP', constraints=cons)
	return res

def cons_func_multi(x, prices, b):
	expected_net_trades = 0
	for  price in prices:
		denom = b * exp_util_v3(x, price, "buy") + b * exp_util_v3(x, price, "sell") + \
					0.5 * (1 -b) * exp_util_v3([0.,0.,1.], price, "buy") + 0.5*(1 - b)*exp_util_v3([1.,0.,0.], price, "sell") 
		fundamental_buys = b * exp_util_v3(x, price, "buy") / denom 
		fundamental_sells = b * exp_util_v3(x, price, "sell")/denom 

		noise_buys = 0.5*(1 -b) * exp_util_v3([0.,0.,1.], price, "buy") / denom
		noise_sells = 0.5*(1 - b)*exp_util_v3([1.,0.,0.], price, "sell") / denom
	
		expected_net_trades += fundamental_buys - fundamental_sells + noise_buys - noise_sells
#	return fundamental_buys, fundamental_sells, noise_buys, noise_sells
	return expected_net_trades

def max_ent_multi(price, PRICES, Q, b, actual, N):
	obj_func = lambda x: CE(x, Q)
	P0 = [0.3,0.2,0.5]
	cons = ({'type':'eq',
		'fun': lambda x: N * cons_func(x, price, b) + N*cons_func_multi(x, PRICES,b) - actual},
		{'type':'eq',
		'fun':lambda x: np.sum(x) - 1},
	)
	res = minimize(obj_func, P0, method='SLSQP', constraints=cons)
	return res

def calculate_changes(some_list):
	x = np.array(some_list)
	assert np.sum(x>0) == len(x)  # all values must be positive
	return np.log(x[1:]) - np.log(x[:-1])
#obj_func = lambda x: CE(x)
#
#P0 = [0.3,0.2, 0.5]


#res = minimize(obj_func, P0, method='SLSQP', constraints=cons)

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
		self.pop_size = pop_size
		self.own_trades = []
		self.trades_prices = []
		self.trader_name = "Simple Trader"


	def get_trade_incentive(self, util_func):
		P = self.market.get_last_price()
		self.util_buy = util_func("buy", P, self.prob)
		self.util_sell = util_func("sell", P, self.prob)
		self.util_none = util_func(None, P, self.prob)

		self.trade_incentive = np.max((self.util_buy, self.util_sell, self.util_none)) - \
															 self.util_none
		return self.trade_incentive

	def create_trade(self):
		ti = self.get_trade_incentive(exp_util)

		if self.util_buy == self.trade_incentive + self.util_none:
			# Buy
			x = 1.
		elif self.util_sell == self.trade_incentive +  self.util_none:
			x = -1.
		else:
			x = 0.


		self.own_trades.append(x)
		self.trades_prices.append(self.market.get_last_price())
		self.market.submit_trade(x)

	def calculate_profits(self):
		if len(self.own_trades) < 2:
			return None
		profits = np.array(self.own_trades) * np.array(self.trades_prices)
		return np.sum(profits)

	def update_expectations(self):
		pass

class RSI_Trader(SimpleTrader):
	def __init__(self, market, pop_size, n, overbought_level, oversold_level):
		self.market = market
		self.pop_size = pop_size
		self.n = n
		self.overbought_level = overbought_level
		self.oversold_level = oversold_level
		self.own_trades = []
		self.trades_prices = []

	def update_expectations(self):
		if len(self.market.price_history) < self.n:
			self.prob = self.market.get_last_price() + np.random.normal(0, 0.01)
			return None

		RSI = compute_RSI(self.market.price_history, self.n)
		if RSI[-1] >= self.overbought_level:
			self.prob = 0.
			#print "Overbought", RSI[-1]
		elif RSI[-1] <= self.oversold_level:
			self.prob = 1.
			#print "Oversold", RSI[-1]
		else:
			self.prob = self.market.get_last_price()+ np.random.normal(0, 0.01)



class Trend_Trader(SimpleTrader):
	def __init__(self, market, pop_size, n_short, n_long) :
		self.market = market
		self.pop_size = pop_size
		self.n_short = n_short
		self.n_long = n_long
		self.own_trades = []
		self.trades_prices = []
		
	def update_expectations(self):
		if len(self.market.price_history) < self.n_long:

			self.prob = self.market.get_last_price() + np.random.normal(0, 0.01)
			return None
		prices = np.array(self.market.price_history)
		ema_short = pd.ewma(prices, self.n_short)
		ema_long = pd.ewma(prices, self.n_long)

		if ema_short[-1] > ema_long[-1] and ema_short[-2] < ema_long[-2]: # Buy signal
			self.prob = 1.
			#print "Trend trader - Short EMA crossed long from below (buy)"
		elif ema_short[-1] < ema_long[-1] and ema_short[-2] > ema_long[-2]: # Sell signal
			self.prob = 0.
			#print "Trend trader - short EMA crossed long from above (sell)"
		else:
			pass
		
					
class Contrarian(SimpleTrader):
	def __init__(self, market, pop_size, n) :
		self.market = market
		self.pop_size = pop_size
		self.n = n
		self.own_trades = []
		self.trades_prices = []

	def update_expectations(self):
		if len(self.market.price_history) < self.n:
			self.prob = self.market.get_last_price() + np.random.normal(0, 0.01)
			return None

		prices = np.array(self.market.price_history)
		ema = pd.ewma(prices, self.n)
		self.prob = ema[-1]



class RandomTrader(SimpleTrader):
	def __init__(self, market, pop_size):
		self.market = market
		self.pop_size = pop_size
		self.own_trades = []
		self.trades_prices = []
		self.trader_name = "Random Trader"

	def update_expectations(self):
		self.prob = np.random.binomial(1, 0.5) * 1.


class TrendFollower(SimpleTrader):
	def __init__(self, market, pop_size, n=5, threshold=0.8):
		self.market = market
		self.pop_size = pop_size
		self.n = n
		self.threshold = threshold
		self.own_trades = []
		self.trades_prices = []
	
	def update_expectations(self):
		if len(self.market.price_history) < self.n:
			self.prob = np.random.binomial(1, 0.5) * 1.
			return None
		x = calculate_changes(self.market.price_history)
		share_positive_changes = 1. * np.sum(x[-self.n : ] > 0) / self.n
		share_negative_changes = 1. * np.sum(x[-self.n : ] < 0) / self.n
		if share_positive_changes > self.threshold:
			self.prob = 1.
		elif share_negative_changes > self.threshold:
			self.prob = 0.
		else:
			self.prob = np.random.binomial(1, 0.5) * 1.

			


class SimpleMarket(object):
	def __init__(self, price_history, max_price=0.95, min_price=0.05, tick=0.01):
		self.price_history = price_history[:]
		self.inventory = 0
		self.inventory_history = [0]
		self.max_price = max_price
		self.min_price = min_price
		self.tick = tick
		self.update_timer = 0
		#self.prior = max_ent(self.price_history[-1], "buy", [0.33,0.33,0.34])
		self.prior = np.array([0.33,0.33,0.34])
		self.New_Prices = []

	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x):
		self.inventory += - x
		self.inventory_history.append(self.inventory)
		self.update_timer +=1

		if self.update_timer >= 1:
			#self.update_price_multiple_periods(10)
			self.update_price(x)
			self.update_timer = 0
		else:
			self.price_history.append(self.get_last_price())

	def update_price_multiple_periods(self, N):
		p = self.get_last_price()
		self.New_Prices.append(p)
		actual = -self.inventory_history[-1] + self.inventory_history[-N-1]
		res = max_ent_v2(p, self.prior, 0.8, actual, N)
		#res = max_ent_multi(p, self.New_Prices, [0.33,0.33,0.34], 0.8, actual, N)
		if res.success:
			self.prior = res.x
		else:
			print res.message
		new_price = np.sum(self.prior * np.array([0.01, 0.5, 0.99]))
		self.price_history.append(new_price)

	def update_price(self, x):
		p = self.get_last_price()
		if x > 0:
			new_p = max_ent(p, "buy", self.prior)
		elif x < 0:
			new_p = max_ent(p, "sell", self.prior)
		else:
			self.price_history.append(p)
			return None
		self.prior = new_p[:]
		new_price = np.sum(new_p * np.array([0.1, 0.5, 0.9]))

		if new_price > p and new_price - p >= 0.5 * self.tick:
			new_price = p + self.tick
		elif new_price < p and p - new_price >= 0.5 * self.tick:
			new_price = p - self.tick
		elif new_price > p and new_price - p < 0.5 * self.tick:
			new_price = p + self.tick
		elif new_price < p and p - new_price < 0.5 * self.tick:
			new_price = p - self.tick

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
		for i, t in enumerate(range(num_trades)):
			trade_incentives = [i.get_trade_incentive(self.util_func) for i in self.all_traders]
			#trade_probs = np.array(self.traders_sizes) * np.array(trade_incentives)
			trade_probs = np.array(self.traders_sizes)
			self.trade_probs = trade_probs / np.sum(trade_probs)
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

