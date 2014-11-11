import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

############################################################################
#################### Data Generation #######################################
############################################################################

class default_params(object):
	Sigma_u = 10 ** 2 # Variance of noise trader order flow

	# Market Maker
	Lambda = 0.5  
	phi = 0.
	y_bar = 10.
	I_bar = 10.

	# Informed trader
	Sigma_0 = 0
	alpha = 5e-2 # risk aversion

	Sigma_e = 0.02**2
	Sigma_n = 3**2

# Market maker price rule
def mm_price_change(x_t, u_t, I, params):
	""" Returns market pice change given the new order flow"""
	return (params.Lambda/params.y_bar) * (x_t + u_t) + \
								 params.phi * (I / params.I_bar)

# Informed trader optimal demand
def compute_xstar(p_last, p_own, params):
	""" Return optimal trade for informed trader """
	Sigma_MM = (params.Lambda / params.y_bar)**2 * params.Sigma_u + params.Sigma_e
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
		e = np.random.normal(0, params.Sigma_e**0.5)
		p_new = p_last + mm_price_change(x_t, u_t, I_last, params) + e
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


def likelihood_ratio(old_price, new_price, shock_time, lfunc,
							price_history, Inventory, params):
	N = len(price_history)
	P1 = np.ones(N) * old_price
	P2 = np.concatenate((np.ones(shock_time)* old_price, 
							np.ones(N - shock_time)*new_price))

	assert len(P1) == len(P2)
	P1 = P1[1:]
	P2 = P2[1:]

	L1 = lfunc(price_history, Inventory, P1, params)
	L2 = lfunc(price_history, Inventory, P2, params)
	# see http://en.wikipedia.org/wiki/Likelihood-ratio_test
	D = 2 * (np.sum(L1) - np.sum(L2))
	return D

def normal_log_density(x, mu, sig):
	return -np.log(sig) - 0.5 * np.log(2*np.pi) - (x - mu) **2 / (2*sig**2)

class SimpleLikelihood(object):
	def __init__(self, price_history, Inventory, lfunc, num_trades, params_class):
		self.price_history = np.array(price_history)
		self.Inv = np.array(Inventory)
		self.params_class = params_class
		self.lfunc = lfunc
		self.num_trades = num_trades

	def obj_func(self, x):
		num_prices = len(self.p_durations) + 1
		# Instantiate a parameter class
		_params = self.params_class()

		# Enter the values for endog. variables
		_params.Lambda=x[0]
		_params.phi = x[1]
		_params.Sigma_u = x[2]
		_params.Sigma_0 = x[3]
		p_own = x[-num_prices:]
		P_OWN = create_price_vector(p_own, self.p_durations, self.num_trades)
		res = self.lfunc(self.price_history, self.Inv, P_OWN, _params)
		#prior_Sigmau = normal_log_density(x[2], 10, 2)
		prior_Sigmau = 0

		prior_Lambda = normal_log_density(x[0], 1., 0.3)
		prior_prices = normal_log_density(np.mean(p_own), 100, 12)
		prior_prices2 = normal_log_density(np.std(p_own), 0., 0.2)
		prior_phi = normal_log_density(x[1], 0., 0.001)

		all_priors = prior_Lambda + prior_phi + prior_Sigmau + \
					prior_prices #+ prior_prices2 	
		return -np.sum(res + all_priors ) 

	def optimize(self):

		p0 = [50] * (len(self.p_durations) + 1)
		x0 = [0.2, -0.1, 70, 20] + p0 

		res = minimize(self.obj_func, x0, method='nelder-mead',
			options={'xtol':1e-8, 'disp':True, 'maxfev':5000, 'maxiter':5000} )
		return res


	def optimize_all_set(self):
		sampling_freq=5
		ret = np.array(self.price_history[1:]) - np.array(self.price_history[:-1])
		N = len(ret)
		ret_re = resample_series(ret, sampling_freq)
		ret_re = np.abs(ret_re)
		n = len(ret_re)
		# Split the data into 3 sub-periods. For each subperiod identify the biggest
		# change. The time of the biggest change will be a candidate for a price_shock

		slicer = np.round(np.linspace(0, n, 4))

		ret1 = ret_re[0 : slicer[1]]
		ret2 = ret_re[slicer[1] : slicer[2]]
		ret3 = ret_re[slicer[2] :]

		indx1 = np.argsort(ret1)
		indx2 = np.argsort(ret2)
		indx3 = np.argsort(ret3)
		p_shocks = [indx1[-1], indx2[-1] + slicer[1], indx3[-1] + slicer[2]]
		p_shocks = np.sort(p_shocks)
		p_shocks *= 1. * sampling_freq
		assert p_shocks[-1] <= N

		self.p_durations = p_shocks - np.concatenate(([0],p_shocks[:-1]))
		res=self.optimize()
		return res

	
def plot(sampling_freq, price_history, X, U, Y):
	Vol = np.abs(X) + np.abs(U)

	prices = resample_series(price_history, sampling_freq)
	volume = resample_series(Vol, sampling_freq)
	oflow = resample_series(Y, sampling_freq)
	 
	fig = plt.figure()
	ax1 = fig.add_subplot(3,1,1)
	ax1.plot(prices)
	ax1.set_title("Price History")

	ax2 = fig.add_subplot(3,1,2)
	ax2.set_title("Volume")
	ax2.plot(volume)

	ax3 = fig.add_subplot(3,1,3)
	ax3.set_title("Net order flow")
	ax3.plot(oflow)

	plt.show()


def func_vol(Sigma, price_history, Vol, informed_prices, price_durations,
			params):
	"""
	This is the volume function. It should return (close to) zero if
	the Sigma parameter is the true parameter. Used to estimate Sigma_u.

	Example:
	--------
	import scipy.optimize as opt
	x = opt.fsolve(func_vol, 10., args=(price_history, Vol, 
					informed_prices, price_durations,params))

	"""
	V = np.mean(Vol)
	T = len(Vol)
	P_last = price_history[:-1]
	P = tools.create_price_vector(informed_prices, price_durations, T)
	numerator = np.sum(np.abs(P - P_last))
	z = params.Lambda/params.y_bar
	a = params.alpha
	w = np.sqrt(2/np.pi)
	denom = 2 * z + a*params.Sigma_0 + a*z**2*Sigma
	res = Sigma**0.5 * w + (1./T) * (1./denom) * numerator - V
	return res
