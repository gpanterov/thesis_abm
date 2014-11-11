import numpy as np
import pandas as pd
import scipy.stats as stats


def normal_log_density(x, mu, sig):
	return -np.log(sig) - 0.5 * np.log(2*np.pi) - (x - mu) **2 / (2*sig**2)

def likelihood2(pdelta, y, Lambda, alpha, Sigma_u, Sigma_e, Sigma_n,
				P_informed, P_last, Sigma_0, y_bar):
	
	z = Lambda / y_bar
	mu1 = z * y
	sig1 = (z**2 * Sigma_n + Sigma_e) **0.5

	xstar = (P_informed - P_last) / \
		(2 * z + alpha * (Sigma_0 + z**2 * Sigma_u + Sigma_e))
	mu2 = xstar
	sig2 = (Sigma_u + Sigma_n)**0.5

	ll= normal_log_density(pdelta, mu1, sig1) + normal_log_density(y, mu2, sig2)
	return np.sum(ll)

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

def likelihood_pdelta(pdelta, Lambda, Sigma_u, alpha, Sigma_e, 
				P_informed, P_last, Sigma_0, y_bar):
	"""
	Returns the log-likelihood of the price change (pdelta)
	"""
	z = Lambda / y_bar
	xstar = (P_informed - P_last) / \
		(2 * z + alpha * (Sigma_0 + z**2 * Sigma_u + Sigma_e))
	mu = z * xstar
	sig = (z**2 * Sigma_u + Sigma_e) **0.5

	pdf = stats.norm.pdf(pdelta, mu, sig)
	ll = normal_log_density(pdelta, mu, sig)
	return np.sum(ll)


def metropolis_hastings(x0, sigmas, lfunc, N=1000):
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
	num_params = len(x0)
	sample = np.empty((N, num_params))
	Xcurrent = x0[:]

	# This function generates the new values
	q = np.random.normal
	# starting values

	old_likelihood = lfunc(Xcurrent)
	for i in range(N):
		for j in range(num_params):
			#assert old_likelihood == lfunc(Xcurrent)
			old_likelihood = lfunc(Xcurrent)
			Xtemp = Xcurrent[:]
			# draw a new value
			draw = q(Xcurrent[j], sigmas[j])
			Xtemp[j] = draw
			try:
				new_likelihood = lfunc(Xtemp)
			except ValueError:
				new_likelihood = -np.inf

			a = new_likelihood - old_likelihood
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
