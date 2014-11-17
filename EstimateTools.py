import numpy as np
import pandas as pd
import scipy.stats as stats
import ABMtools as abmtools
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import minimize
import math

import BayesianTools as btools
reload(btools)
import ABMtools as abmtools
reload(abmtools)



def log_likelihood(Lambda, alpha, Sigma_0, Sigma_u, Sigma_e, P_informed, 
					P_last, pdelta, vol):

	z = Lambda
	xstar = (P_informed - P_last) / \
		(2 * z + alpha * (Sigma_0 + z**2 * Sigma_u + Sigma_e))

	mu1 = (1. / z) * pdelta - xstar
	sig1 = (1. / z) * Sigma_e**0.5
	l1 = stats.norm.pdf(vol - np.abs(xstar), mu1, sig1) + \
			stats.norm.pdf(-vol + np.abs(xstar), mu1, sig1)

	mu2 = z * xstar
	sig2 = (z**2 * Sigma_u + Sigma_e)**0.5

	ll2 = btools.normal_log_density(pdelta, mu2, sig2)

	ll= np.log(l1) + ll2
	return np.sum(ll)

def split_window(x, Lambda, alpha, Sigma_0, Sigma_u, Sigma_e,
						P_last, pdelta, vol): 
	x = list(x)
	cutoff =  len(pdelta)/2
	Pi = [x[0]] * cutoff + [x[1]] * cutoff
	return log_likelihood(Lambda, alpha, Sigma_0, Sigma_u, Sigma_e, Pi, 
					P_last, pdelta, vol)

def single_window(x, Lambda, alpha, Sigma_0, Sigma_u, Sigma_e,
						P_last, pdelta, vol): 
	x = list(x)
	Pi = x * len(pdelta)
	return log_likelihood(Lambda, alpha, Sigma_0, Sigma_u, Sigma_e, Pi, 
					P_last, pdelta, vol)


def get_trading_signals(price_history, vol,  Lambda, alpha, 
						Sigma_0, Sigma_u, Sigma_e, window_size=20):
	"""
	Produces a set of trading signals for a given price series by 
	estimating discrete change in prices of informed trader
	
	Example:
	--------
	import EstimateTools as etools

	window_size=16
	Xsample, Xsample_sd = etools.get_trading_signals(price_history, vol,  Lambda, alpha, 
						Sigma_0, Sigma_u, Sigma_e, window_size)

	plt.figure(2)
	plt.subplot(3,1,1)
	plt.plot(price_history[window_size:])
	plt.plot(Xsample[:,1])

	plt.subplot(3,1,2)
	plt.plot(Xsample_sd[:,1])

	plt.subplot(3,1,3)
	plt.plot(np.abs(Xsample[:,1] - Xsample[:,0]))
	plt.show()


	"""
	pdelta = np.array(price_history[1:]) - np.array(price_history[:-1])
	num_trades = len(price_history) - 1

	
	for i in range(0, num_trades - window_size):
		if i%25 == 0:
			print "Trading window is at position ", i
		l = i
		u = l + window_size
		Pl = np.array(price_history[l:u])
		pd = np.array(pdelta[l:u])
		v = vol[l:u]

		# Split window estimation
		x0=[np.mean(Pl)]*2
		sigmas=[1,1]
		func = lambda x: split_window(x, Lambda, alpha, Sigma_0, Sigma_u, Sigma_e,
						Pl, pd, v)

# 		# Single window estimation 
#		x0 = [np.mean(Pl)]
#		sigmas = [1]
#		func = lambda x: single_window(x, Lambda, alpha, Sigma_0, Sigma_u, Sigma_e,
#						Pl, pd, v)


		sample =btools.metropolis_hastings(x0, sigmas, func)
		x_sample = np.mean(sample[50:, :], axis=0)
		x_sample_sd = np.std(sample[50:, :], axis=0)
		if i == 0:
			Xsample = x_sample
			Xsample_sd = x_sample_sd
		else:
			Xsample = np.row_stack((Xsample, x_sample))
			Xsample_sd = np.row_stack((Xsample_sd, x_sample_sd))
	return Xsample, Xsample_sd

class TradingModel(object):

	"""
	This class produces the MCMC estimates and the optimization estimates

	Example:
	--------
	import EstimateTools as etools
	model = etools.TradingModel(price_history, vol)
	model.MCMC()
	model.MAP()

	print model.map_estimate
	print model.posterior_means

	"""
	def __init__(self, price_history, vol, price_durations):

		assert len(price_history) == len(vol) + 1

		self.price_history = price_history
		self.vol = vol

		self.P_last = np.array(price_history[:-1])
		self.pdelta = np.array(price_history[1:]) - self.P_last
		self.num_trades = len(self.pdelta)

		self.get_price_durations(price_durations)
		self.get_priors()

		self.posterior_means = False

	def MCMC(self, N=1000):
		"""
		Draws a random sample from the posterior distribution
		"""
		x0 = [self.Lambda_mu, self.alpha_mu, self.Sigma0_mu, self.Sigmau_mu, self.Sigmae_mu] + \
				[np.mean(self.price_history)] * (len(self.price_durations) + 1)
		sigmas = [self.Lambda_sig, self.alpha_sig, self.Sigma0_sig, self.Sigmau_sig, self.Sigmae_sig] + \
					[1] * (len(self.price_durations) + 1)
		ll0 = self.log_posterior(x0)
		print ll0
		self.posterior_sample = btools.metropolis_hastings(x0, sigmas, self.log_posterior , N) 
		self.posterior_means = np.mean(self.posterior_sample[50:, :], axis=0)

	def MAP(self):
		""" Maximum a posteriori estimation """
		obj_func = lambda x: - self.log_posterior(x)
		x0 = self.posterior_means
		self.optimization_results = minimize(obj_func, x0, method='nelder-mead', 
									options={'disp':True})
		self.map_estimate = self.optimization_results.x

	def log_posterior(self, params):
		K = len(self.price_durations) + 1
		x = params[:-K]

		Lambda = x[0]
		alpha = x[1]
		Sigma_0 = x[2]
		Sigma_u = x[3]
		Sigma_e = x[4]

		informed_prices = params[-K:]
		P_informed = abmtools.create_price_vector(informed_prices,
									 self.price_durations, self.num_trades)

		ll = log_likelihood(Lambda, alpha, Sigma_0, Sigma_u, 
						Sigma_e, P_informed, self.P_last, self.pdelta, self.vol)

		log_priors = np.log(norm.pdf(Lambda, self.Lambda_mu, self.Lambda_sig)) \
		+ np.log(norm.pdf(alpha, self.alpha_mu, self.alpha_sig)) \
		+ np.log(norm.pdf(Sigma_0, self.Sigma0_mu, self.Sigma0_sig))  \
		+ np.log(norm.pdf(Sigma_u, self.Sigmau_mu, self.Sigmau_sig)) \
		+ np.log(norm.pdf(Sigma_e, self.Sigmae_mu, self.Sigmae_sig))

		return ll #+ log_priors

	def get_price_durations(self, price_durations):
		self.price_durations =  price_durations

	def get_priors(self):
		self.Lambda_mu = 0.025
		self.Lambda_sig = 0.05

		self.alpha_mu = 0.05
		self.alpha_sig = 0.03
		
		self.Sigma0_mu = 10
		self.Sigma0_sig = 3

		self.Sigmau_mu = 0.75 * np.var(self.vol) / (1 - 2/np.pi)  # Approximate how much noise trading
		self.Sigmau_sig = 0.1 * np.var(self.vol) / (1 - 2/np.pi)

		self.Sigmae_mu = 0.0004
		self.Sigmae_sig = 0.001

