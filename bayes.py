import numpy as np
from scipy.stats import norm
# Calculate for prior from price distribution
def compute_uniform_prior(Ns):
	return np.ones(Ns) / Ns
class BayesNormal(object):
	# Enter some default values
	a = 1e-2
	sig = 0.1
	a_sig2 = a * sig**2

	sig_ei = 0.01
	sig_en = 0.01

	n = 10 # number of trades on which the market maker averages

	def expected_trade(self, p, mu):
		# Calculates the expected trade for a trader with expectation mu at price p
		return (mu - p) / self.a_sig2

	def compute_prior_from_params(self, p, mu_i, mu_n, alpha):
		# Calculate the mean and var of the prior 
		# given the believes about the price mu_i, mu_n and alpha
		X_i = self.expected_trade(p, mu_i)
		X_n = self.expected_trade(p, mu_n)

		mu = (alpha * X_i + (1 - alpha) * X_n) * self.n

		var_s = ((alpha/self.a_sig2)**2 * self.sig_ei**2 + \
				 ((1-alpha) / (self.a_sig2))**2 * self.sig_en**2) * self.n**2

		return mu, var_s


class BayesDiscrete(object):

	sig_ei = 0.01
	sig_en = 0.01

	def calculate_likelihood_buy(self, p, mu_i, mu_n, alpha):
		# calculate likelihood of a buy given the parameters
		return alpha * (1 - norm.cdf(p, mu_i, self.sig_ei)) + \
				(1 - alpha) * (1 - norm.cdf(p, mu_n, self.sig_en))

	def calculate_likelihood_sell(self, p, mu_i, mu_n, alpha):
		return alpha * norm.cdf(p, mu_i, self.sig_ei) + \
				(1 - alpha) * norm.cdf(p, mu_n, self.sig_en)

	def calculate_posterior(self, trade, prior, p, mu_n, alpha):
		Ns = len(prior)
		s = np.linspace(0, 1, Ns)
		self.support = s
		if trade == 1:
			posterior = self.calculate_likelihood_buy(p, s, mu_n, alpha) * prior
		else:
			posterior = self.calculate_likelihood_sell(p, s, mu_n, alpha) * prior

		return posterior / np.sum(posterior) 
		
