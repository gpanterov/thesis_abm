import numpy as np
import pandas as pd
import scipy.stats as stats
import ABMtools as abmtools
from scipy.stats import norm
import scipy.stats as stats
import math

def folded_pdf(x, mu, sigma):
	"""
	Returns the pdf of a folded normal random variable with mu, sigma

	Reference: 
	---------
	http://en.wikipedia.org/wiki/Folded_normal_distribution
	"""

	z = 1. / (sigma * np.sqrt(2 * np.pi)) # helper term
	pdf = z * np.exp(-(x - mu)**2 / (2 * sigma**2)) + \
		z *  np.exp(-(-x - mu)**2 / (2 * sigma**2)) 
	return pdf

def normal_log_density(x, mu, sig):
	return -np.log(sig) - 0.5 * np.log(2*np.pi) - (x - mu) **2 / (2*sig**2)

def normal_density(x, mu, sig):

	return (1./(sig * (2 * np.pi)**0.5)) * np.exp(- (x - mu)**2 / (2.*sig**2))


def likelihood1(pdelta, vol,  Lambda, alpha, Sigma_u, Sigma_e,
				P_informed, P_last, Sigma_0, y_bar):
	"""
	Returns the log-likelihood for the join probability of price_change
	and volume (non-signed)
	"""

	z = Lambda / y_bar
	xstar = (P_informed - P_last) / \
		(2 * z + alpha * (Sigma_0 + z**2 * Sigma_u + Sigma_e))

	mu1 = (1. / z) * pdelta - xstar
	sig1 = (1. / z) * Sigma_e**0.5
	l1 = stats.norm.pdf(vol - np.abs(xstar), mu1, sig1) + \
			stats.norm.pdf(-vol + np.abs(xstar), mu1, sig1)

	mu2 = z * xstar
	sig2 = (z**2 * Sigma_u + Sigma_e)**0.5

	ll2 = normal_log_density(pdelta, mu2, sig2)

	ll= np.log(l1) + ll2
	return np.sum(ll)


def likelihood2(pdelta, y, Lambda, alpha, Sigma_u, Sigma_e, Sigma_n,
				P_informed, P_last, Sigma_0, y_bar):
	"""
	Returns the log-likelihood for the join probability of price_change
	and signed volume 
	"""

	z = Lambda / y_bar
	mu1 = z * y
	sig1 = (z**2 * Sigma_n + Sigma_e) **0.5

	xstar = (P_informed - P_last) / \
		(2 * z + alpha * (Sigma_0 + z**2 * Sigma_u + Sigma_e))
	mu2 = xstar
	sig2 = (Sigma_u + Sigma_n)**0.5

	ll= normal_log_density(pdelta, mu1, sig1) + normal_log_density(y, mu2, sig2)
	return np.sum(ll)




def obj_func1(params, price_history, vol,
					price_durations, y_bar):

	"""
	Objective function for optimization
	"""
	num_trades = len(price_history) - 1
	P_last = price_history[:-1]
	pdelta = np.array(price_history[1:]) - P_last

	K = len(price_durations) + 1
	informed_prices = params[-K:]	
	P_informed = abmtools.create_price_vector(informed_prices,
									 price_durations, num_trades)
	x = params[:-K]

	Sigma_0 = x[4]
	lfunc1 = lambda x: likelihood1(pdelta, vol,  
		x[0], x[1], x[2], x[3], P_informed, P_last, Sigma_0, y_bar) \
		+ np.log(norm.pdf(x[0], 0.5, 0.1)) \
		+ np.log(norm.pdf(x[1], 5e-2, 1e-1)) \
		+ np.log(norm.pdf(x[2], 100., 15.))  \
		+ np.log(norm.pdf(x[3], 0.02**2, 0.01)) \
		+ np.log(norm.pdf(x[4], 10, 5))

	return -lfunc1(x)

def log_posterior_1(params0, price_history, vol,
					price_durations, y_bar):

	"""
	Threshold function for metropolis hastings. It is the the log of posterior
	distribution
	"""
	num_trades = len(price_history) - 1
	P_last = price_history[:-1]
	pdelta = np.array(price_history[1:]) - P_last

	K = len(price_durations) + 1
	informed_prices = params0[-K:]	
	P_informed = abmtools.create_price_vector(informed_prices,
									 price_durations, num_trades)
	x = params0[:-K]

	Sigma_0 = x[4]
	lfunc1 = lambda x: likelihood1(pdelta, vol,  
		x[0], x[1], x[2], x[3], P_informed, P_last, Sigma_0, y_bar) \
		+ np.log(norm.pdf(x[0], 0.3, 0.1)) \
		+ np.log(norm.pdf(x[1], 5e-2, 1e-1)) \
		+ np.log(norm.pdf(x[2], 80., 15.))  \
		+ np.log(norm.pdf(x[3], 0.02**2, 0.01)) \
		+ np.log(norm.pdf(x[4], 10, 5))

	return lfunc1(x)






def metropolis_hastings(x0, sigmas, lfunc, N=1000, types_cont=False):
	"""
	Samples from a given distribution according to the
	Metropolis Hastings algorithm

	Parameters:
	-----------

	lfunc: likelihood function (lambda function that takes x0 as argument only)
	x0: (Kx1) array of starting values
	sigmas: list
			standard deviations for the drawing function q
	N: number of samples to return

	Returns:
	--------
	sample: (KxN) array of samples 

	Example:
	--------
	import BayesianTools as btools

	lfunc = lambda x: btools.likelihood_pdelta(pdelta, 
		x[0], x[1], x[2], P_informed, P_last, Sigma_0, y_bar) \
		+ np.log(norm.pdf(x[2], 0.05, 1e-2))  \
		+ np.log(norm.pdf(x[0], 0.5, 0.1)) \
		+ np.log(norm.pdf(x[1], 50., 50))

	x0 = [0.3,60., 1e-2]
	sample = btools.metropolis_hastings(x0, lfunc) 

	"""

	poisson_log_pmf = lambda mu, k: -mu + k * math.log(mu) - math.log(math.factorial(k))

	num_params = len(x0)
	sample = np.empty((N, num_params))
	Xcurrent = list(x0)

	if not types_cont:
		types_cont = [True] * num_params

	old_likelihood = lfunc(Xcurrent)
	for i in range(N):
		#print Xcurrent
		for j in range(num_params):
			Xtemp = Xcurrent[:]
			# draw a new value
			if types_cont[j]:
				draw = np.random.normal(Xcurrent[j], sigmas[j])
				c = 0
			else:
				#draw = stats.poisson.rvs(Xcurrent[j])
				#c = poisson_log_pmf(Xcurrent[j], draw) - poisson_log_pmf(draw, Xcurrent[j])
				draw = np.random.normal(Xcurrent[j], sigmas[j])
				draw = min(draw, 0.99)
				draw = max(draw, 0.01)
				c = 0
			Xtemp[j] = draw
			if draw > 0: 
				new_likelihood = lfunc(Xtemp)
			else: # negative values for the parameters have zero likelihood
				new_likelihood = -np.inf


			a = new_likelihood - old_likelihood + c
			if a >= 0:
				Xcurrent[j] = draw
				old_likelihood = new_likelihood
			else:
				u = np.random.uniform()
				prob = np.exp(a)
				if prob > u:
					Xcurrent[j] = draw
					old_likelihood = new_likelihood
				else:
					pass
		sample[i, :] = np.array(Xcurrent)	

	return sample
