
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
	alpha = 0.6

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
		self.initialize()

	def initialize(self):
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

def MM_loss(p, support, prob, p0, alpha=1e-3):
	I = (support >=p) * 1.
	I[I==0] = -1

	loss = - (support - p) * I #- 1e-5 * (p!=p0)
	return np.sum(prob * (1 - np.exp(-alpha*loss)))

def MM_loss_log(p, support, prob, p0):
	I = (support >=p) * 1.
	I[I==0] = -1

	loss =  (support - p) * I + 1e-5
	return np.sum(prob * np.log(loss)) - 2e-1 * (p!=p0)

def day_trade(P, num_trades, mu_i, prior, price_history, Records):
	""" Completes one day of trading"""
	daily_prices = [price_history[-1]]
	for i in range(num_trades):
		# Noise traders update prices
		mu_n = AR(P.ar_params, price_history)

		# Generate a trade
		trade = generate_trade(P, price_history[-1], mu_i, mu_n)

		# Market maker updates prices
		mu_i_mm, prior = MM_update(P, trade, prior, price_history)

		if i%2==0 and i > 1:
			if mu_i_mm > price_history[-1]:
				new_price = min(price_history[-1] + 3*P.tick, mu_i_mm)
			else:
				new_price = max(price_history[-1] - 3*P.tick, mu_i_mm)
			new_price = np.round(new_price, 2)
			price_history.append(new_price)
			daily_prices.append(new_price)

		# Keep recod
		Records.mu_i.append(mu_i)
		Records.entropies.append(np.sum(-prior*np.log2(prior + 1e-5)))
		Records.mu_i_mm.append(mu_i_mm)
		Records.all_trades.append(trade)

	# calculates daily performance
	daily_returns = np.array(daily_prices[1:]) - np.array(daily_prices[:-1])
	Records.closing_price.append(daily_prices[-1])
	Records.opening_price.append(daily_prices[0])
	Records.daily_return.append(Records.closing_price[-1] - Records.opening_price[-1])
	Records.daily_volatility.append(np.std(daily_returns))


	return prior, price_history

def simulate(P, num_sim, num_days, num_trades):
	""" Simulates num_sim times multi-day trading (num_days)"""
	SimRet = pd.DataFrame(index = range(num_days), columns = range(num_sim))
	SimVol = pd.DataFrame(index = range(num_days), columns = range(num_sim))
	for i in range(num_sim):
		prior = P.prior[:]
		price_history = P.price_history[:]
		mu_i = P.mu_i
		Records = SimulationRecords()
		for day in range(num_days):
			mu_i = mu_i + P.shocks_i[day]
			prior, price_history = day_trade(P, num_trades, mu_i, prior, price_history, Records)
		SimRet[i] = Records.daily_return 
		SimVol[i] = Records.daily_volatility
	return SimRet, SimVol

def fit(BaseRet, BaseVol, SimRet, SimVol):
	""" Estimates the goodness of fit of the simulation """

	mad_ret = np.abs(SimRet.values - BaseRet.values)
	mad_ret = np.mean(mad_ret, axis=0)

	mad_vol = np.abs(SimVol.values - BaseVol.values)
	mad_vol = np.mean(mad_vol, axis=0)
	num_sim = len(SimRet.columns)
	Res = pd.DataFrame(index=range(num_sim), columns=['mad_ret', 'mad_vol'])
	Res['mad_vol'] = mad_vol
	Res['mad_ret'] = mad_ret

	return Res

