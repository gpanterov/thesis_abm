import numpy as np
import pandas as pd
from scipy.optimize import minimize

############################################################################
#################### Data Generation #######################################
############################################################################

class default_params(object):
	Sigma_u = 10 ** 2 # Variance of noise trader order flow

	# Market Maker
	Lambda = 0.5  
	phi = -0.2
	y_bar = 10.
	I_bar = 10.

	# Informed trader
	Sigma_0 = 1.5 ** 2
	alpha = 5e-2 # risk aversion

# Market maker price rule
def mm_price_change(x_t, u_t, I, params):
	""" Returns market pice change given the new order flow"""
	return (params.Lambda/params.y_bar) * (x_t + u_t) + \
								 params.phi * (I / params.I_bar)

# Informed trader optimal demand
def compute_xstar(p_last, p_own, params):
	""" Return optimal trade for informed trader """
	Sigma_MM = (params.Lambda / params.y_bar)**2 * params.Sigma_u
	return (p_own - p_last) / ((2 * params.Lambda / params.y_bar) + \
			params.alpha * (params.Sigma_0 + Sigma_MM))

# Generate raw data

def create_price_vector(informed_prices, price_durations, num_trades):
	price_durations = list(price_durations)
	informed_prices = list(informed_prices)
	assert num_trades > np.sum(price_durations)
	assert len(informed_prices) == len(price_durations) + 1

	P = []
	for i, t in enumerate(price_durations):
		P += [informed_prices[i]] * int(t)
	P += informed_prices[-1:] * int(num_trades - np.sum(price_durations))
	
	return np.array(P)

def raw_data(informed_prices, price_durations, p_start, num_trades, params):
	""" 
	Generates syntetic trading data

	informed_prices: array
				vector of prices for the informed trader
	"""
	price_history = [p_start]
	X=[]
	U = []
	Y = []

	Inv = [0]

	informed_prices = create_price_vector(informed_prices, price_durations, num_trades)
	for p in informed_prices:
		# Noise trader order
		p_last = price_history[-1]
		u_t = np.random.normal(0, params.Sigma_u**0.5)
		x_t = compute_xstar(p_last, p, params)
		I_last = Inv[-1]
		I_new = I_last - (x_t + u_t)
		p_new = p_last + mm_price_change(x_t, u_t, I_last, params)
		price_history.append(p_new)
		Inv.append(I_new)
		X.append(x_t)
		U.append(u_t)
		Y.append(x_t + u_t)
	return price_history, Inv, X, U, Y


###############################################################################
############################ Estimation #######################################
###############################################################################
def resample_series(series, sampling_freq=5):
	series = np.array(series)
	series = pd.Series(series)	
	freq = series.index / sampling_freq
	series = series.groupby([freq]).mean()
	return series.values


def simple_likelihood(price_history, Inventory, P_OWN, params):
	""" Likelihood of observing a certain price series """
	P_LAST = np.array(price_history[:-1])
	P = np.array(price_history[1:])
	DELTA_P = P - P_LAST

	I_LAST = np.array(Inventory[:-1])
	XSTAR = compute_xstar(P_LAST, P_OWN, params)

	MU_DP = (params.Lambda / params.y_bar) * XSTAR + \
						params.phi * I_LAST / params.I_bar
	Sigma_dp = (params.Lambda / params.y_bar)**2 * params.Sigma_u
	LN_F = -0.5 * np.log(2*np.pi) - np.log(Sigma_dp**0.5) - \
		(DELTA_P - MU_DP)**2 / (2 * Sigma_dp)  
	return LN_F 



class SimpleLikelihood(object):
	def __init__(self, price_history, Inventory, lfunc, num_trades, p_durations, params_class):
		self.price_history = np.array(price_history)
		self.Inv = np.array(Inventory)
		self.params_class = params_class
		self.lfunc = lfunc
		self.num_trades = num_trades
		self.p_durations = p_durations

	def obj_func(self, x):

		# Instantiate a parameter class
		_params = self.params_class()
		# Enter the values for endog. variables
		#_params.Lambda = x[0]
		#_params.phi = x[1]
		#_params.Sigma_u = x[2]

		_params.Lambda=x[0]
		_params.phi = x[1]
		p_own = x[2:]
		P_OWN = create_price_vector(p_own, self.p_durations, self.num_trades)
		res = self.lfunc(self.price_history, self.Inv, P_OWN, _params)
		return -np.sum(res) 

	def optimize(self):

		p0 = [50, 50]
		x0 = [0.2, -0.1] + p0 

		res = minimize(self.obj_func, x0, method='nelder-mead',
			options={'xtol':1e-8, 'disp':True, 'maxfev':5000, 'maxiter':5000} )
		return res


