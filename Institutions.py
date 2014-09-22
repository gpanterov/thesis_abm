import numpy as np
from scipy.optimize import minimize


def cross_ent(P, Q=[0.33,0.33,0.34]):
	P = np.array(P)
	Q = np.array(Q)
	ce =  np.sum(P * np.log(P / Q))

	return ce

def entropy(P):
	P = np.array(P)
	return - np.sum(P * np.log(P))

def expected_size(P, market_prices, prob_noise, mu_noise, sig_noise, a_int, a_noise):
	"""
	The expected trade size give market parameters for noise and intelligent
	traders
	"""
	market_prices = np.array(market_prices)
	mu_support = np.array([0.01, 0.5, 0.99])
	sig_support = np.array([0.01, 0.1, 0.19])

	# Calculate the mean and st dev for an intellgent trader
	mu_int = np.sum(np.array(P[0:3]) * mu_support)
	sig_int = np.sum(np.array(P[3:6]) * sig_support)

	# Expected size for an intelligent and noise traders
	size_int = (mu_int - market_prices) / (a_int * sig_int**2)
	size_noise = (mu_noise - market_prices) / (a_noise * sig_noise**2)
	avg_size_per_period = prob_noise * size_noise + (1 - prob_noise) * size_int
	
	return np.mean(avg_size_per_period) 

def MaxEnt_market_maker(market_prices, avg_sizes, 
			prob_noise, mu_noise, sig_noise, a_int, a_noise, 
				Q = [0.33, 0.33, 0.34, 0.33, 0.33, 0.34]):
	"""
	Maximum entropy problem for the market maker
	"""
	obj_func = lambda P: - entropy(P[0:3]) - entropy(P[3:6])
#	obj_func = lambda P: cross_ent(P[0:3], Q[0:3]) + cross_ent(P[3:6], Q[3:6])

	# COnstraints
	cons = ({'type':'eq',
	'fun':lambda x: expected_size(x, market_prices, 
				prob_noise, mu_noise, sig_noise, a_int, a_noise) - np.mean(avg_sizes)},
	{'type':'eq',
	'fun':lambda x: np.sum(x[0:3]) - 1},
	{'type':'eq',
	'fun':lambda x: np.sum(x[3:]) - 1 },)

	# Optimization
	P0 = [0.33, 0.33, 0.34, 0.33, 0.33, 0.34]
	res = minimize(obj_func, P0, method='SLSQP', constraints=cons)
	return res



class Market(object):
	def __init__(self, price_history, prob_noise, sig_noise, a_int, a_noise):
		self.price_history = price_history[:]
	
		self.time = 0
		self.inventory = 0
		self.outstanding_trades = []
		self.orders = []

		self.prob_noise = prob_noise
		self.sig_noise = sig_noise
		self.a_int = a_int
		self.a_noise = a_noise

		self.avg_sizes=[]
		self.market_prices = [price_history[-1]]

		self.prior = [0.33, 0.33, 0.34, 0.33,0.33,0.34]
	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x, trader_name):
		self.outstanding_trades.append(x)
		self.orders.append([self.time, self.get_last_price(), x, trader_name])

		# Every 10 trades the market maker updates the price and clears the market
		if len(self.outstanding_trades) >= 10:
			self.update_price()

	def get_mu_noise(self):
		self.mu_noise = self.get_last_price()
		return self.get_last_price()

	def update_price(self):
		p = self.get_last_price()
		net_pos = np.sum(self.outstanding_trades)
		self.inventory += - net_pos
		avg_size = 1. * net_pos / len(self.outstanding_trades)
		self.avg_sizes.append(avg_size)
		avg_sizes = self.avg_sizes[-1:]
		market_prices = self.market_prices[-1:]
		#mu_noise = self.get_mu_noise()
		res = MaxEnt_market_maker(market_prices, avg_sizes, 
			self.prob_noise, self.mu_noise, 
			self.sig_noise, self.a_int, self.a_noise, self.prior)
		self.prior = res.x
		mu_hat = np.sum(res.x[0:3] * np.array([0.01, 0.5, 0.99]))
		sig_hat = np.sum(res.x[3:6] *  np.array([0.01, 0.1, 0.19]))
		self.price_history.append(mu_hat)
		self.market_prices.append(mu_hat)
		self.outstanding_trades = []
