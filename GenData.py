"""
This module generates trading data
"""

import numpy as np
import pandas as pd
import bayes as bayes
reload(bayes)
from scipy.stats import norm




# seed
#np.random.seed(12345)

################
# Trader tools #
################

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
	params = np.array(params)
	n = len(params)
	R = np.log(P[1:]) - np.log(P[:-1])
	r = np.sum(params*R[-n:])
	p = P[-1] * (1. + r)
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



##############
# Parameters #
##############

# price history
price_history = [0.5, 0.501, 0.502, 0.503]

# number of trading days
num_days = 10

# number of trades per day
num_trades = 2

# probability of informed trading
alpha = 0.6


#-----------------------------------------------------------------------------


## INFORMED TRADERS

# starting belief
mu_i = price_history[-1]

# informed trader idiosyncratic shock (standard deviation)
s_i = 0.001

# shocks to informed traders beliefs about asset price in each day (new info)
shocks_i = np.random.normal(0.0, 0.001, size=(num_days,))
# ----------------------------------------------------------------------------

## NOISE TRADERS

# AR parameters
ar_params = [0.3, 0.3, 0.4]

# starting belief
mu_n = AR(ar_params, price_history)

# noise trader idiosyncratic shock (standard deviation)
s_n = 0.001
#------------------------------------------------------------------------------



## MARKET MAKER


# prior of market maker
prior = bayes.compute_uniform_prior(100)

# market maker belief about alpha
alpha_mm = alpha

# market maker belief about AR params of noise trader
ar_params_mm = ar_params

# market maker belief about idiosincratic shock to informed traders
s_i_mm = s_i

# market maker belief about idiosyncratic shock to noise traders
s_n_mm = s_n


b = bayes.BayesDiscrete(s_i_mm, s_n_mm)
#------------------------------------------------------------------------------


######################
# Trading Simulation #
######################
MM = {'all_trades' : [],
		'mu_n' : [],
		'mu_i' : [],
		'entropies' : []}
Trader_i = {'mu':[]}

for day in range(num_days):
	mu_i = mu_i + shocks_i[day]
	for i in range(num_trades):
		if np.random.uniform() < alpha: # informed trader arrives
			prob_buy = (1 - norm.cdf(price_history[-1], mu_i, s_i)) 
		else:  # noise trader arrives
			prob_buy = (1 - norm.cdf(price_history[-1], mu_n, s_n))

		trade = np.random.binomial(1, prob_buy)
		MM['all_trades'].append(trade)

		# Market maker updates prices
		mu_n_mm = AR(ar_params_mm, price_history) # belief about noise traders res. price
		prior = b.calculate_posterior(trade, prior, price_history[-1], mu_n_mm, alpha_mm)
		mu_i_mm = np.sum(prior * b.support)
		price_history.append(mu_i_mm)
		# Noise traders update prices
		mu_n = AR(ar_params, price_history)

		# Keep recod
		Trader_i['mu'].append(mu_i)
		MM['entropies'].append(np.sum(-prior*np.log2(prior + 1e-5)))
		MM['mu_i'].append(mu_i_mm)


# Resample prices
prices, ret = resample_prices_ret(price_history, sampling_freq=num_trades)
