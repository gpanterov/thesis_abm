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


class TradingModel(object):

	"""
	This class produces the MCMC estimates and the optimization estimates

	Example:
	--------
	import EstimateTools as etools
	model = etools.TradingModel(price_history, vol)
	model.MCMC()
	model.MAP()
	"""
	def __init__(self, price_history, vol):

		assert len(price_history) == len(vol) + 1

		self.price_history = price_history
		self.vol = vol

		self.P_last = np.array(price_history[:-1])
		self.pdelta = np.array(price_history[1:]) - self.P_last
		self.num_trades = len(self.pdelta)

		self.get_price_durations()
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

		return ll + log_priors

	def get_price_durations(self):
		self.price_durations =  [100, 80, 90]

	def get_priors(self):
		self.Lambda_mu = 0.025
		self.Lambda_sig = 0.05

		self.alpha_mu = 0.05
		self.alpha_sig = 0.03
		
		self.Sigma0_mu = 10
		self.Sigma0_sig = 3

		self.Sigmau_mu = 100
		self.Sigmau_sig = 15

		self.Sigmae_mu = 0.0004
		self.Sigmae_sig = 0.001

