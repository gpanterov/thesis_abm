
import numpy as np
import pandas as pd
import bayes as bayes
reload(bayes)
from scipy.stats import norm


class SimulationParams(object):
	##############
	# Parameters #
	##############

	# probability of informed trading
	alpha = 0.9

	# Price tick
	tick = 0.01

	# informed trader idiosyncratic shock (standard deviation)
	s_i = 0.01

	# shocks to informed traders beliefs about asset price in each day (new info)
	# ----------------------------------------------------------------------------

	## NOISE TRADERS

	# AR parameters
	ar_params = [0.3, 0.3, 0.4]

	# noise trader idiosyncratic shock (standard deviation)
	s_n = 0.01
	#------------------------------------------------------------------------------


	def __init__(self, num_days, num_trades, price_history= [0.5, 0.51, 0.52, 0.53]):

		# number of trading days
		self.num_days = num_days

		# number of trades per day
		self.num_trades = num_trades

		# price history
		self.price_history = price_history

		self.init_IT()
		self.init_NT()
		self.init_MM()

	def init_IT(self):
		"""Initialize informed trader"""
		self.shocks_i = [0.01] * self.num_days

		# starting belief of informed trader
		self.mu_i = self.price_history[-1]

	

	def init_NT(self):
		""" Initialize Noise Trader"""
		# starting belief for noise trader
		self.mu_n = AR(self.ar_params, self.price_history)


	def init_MM(self):
		"""
		Initialize Market maker
		"""
		# prior of market maker
		self.prior = bayes.compute_uniform_prior(100)
		# market maker belief about alpha
		self.alpha_mm = self.alpha

		# market maker belief about idiosyncratic shock to noise traders
		self.s_n_mm = self.s_n

		# market maker belief about AR params of noise trader
		self.ar_params_mm = self.ar_params

		# market maker belief about idiosincratic shock to informed traders
		self.s_i_mm = self.s_i

		# Bayesian
		self.b = bayes.BayesDiscrete(self.s_i_mm, self.s_n_mm)

class SimulationRecords(object):
	# Trade related
	def __init__(self):
		self.mu_i = []
		self.mu_i_mm = []
		self.all_trades = []
		self.entropies = []

		# day related
		self.daily_return = []
		self.daily_volatility = []
		self.closing_price = []
		self.opening_price = []


def generate_trade(P, market_price, mu_i, mu_n):
	""" Generates a single (0,1) trade. P is a parameter class SimulationParams"""		
	if np.random.uniform() < P.alpha: # informed trader arrives
		prob_buy = (1 - norm.cdf(market_price, mu_i, P.s_i)) 
	else:  # noise trader arrives
		prob_buy = (1 - norm.cdf(market_price, mu_n, P.s_n))
	return np.random.binomial(1, prob_buy)

def MM_update(P, trade, prior, price_history):
	mu_n_mm = AR(P.ar_params_mm, price_history) # belief about noise traders res. price
	new_prior = P.b.calculate_posterior(trade, prior, price_history[-1], mu_n_mm, P.alpha_mm)
	mu_i_mm = np.sum(new_prior * P.b.support)
	return mu_i_mm, new_prior


def AR(params, P):
	"""
	Returns the next value in an AR process for the returns
	Parameters
	----------
	params: array 
			AR params
	P:	array
		Prices (history)
	"""
	P = np.array(P)
	params = np.array(params)
	n = len(params)
	R = P[1:] - P[:-1]
	r = np.sum(params*R[-n:])
	p = P[-1] + r
	return p

def resample_prices_ret(price_history, sampling_freq=10):
	prices = np.array(price_history)
	delta = (prices[1:] - prices[0:-1]) / prices[0:-1]
	s = pd.Series(delta)
	freq = s.index / sampling_freq

	returns = s.groupby([freq]).mean()
	
	prices = pd.Series(prices)	
	freq2 = prices.index / sampling_freq
	prices = prices.groupby([freq2]).mean()
	return prices, returns

