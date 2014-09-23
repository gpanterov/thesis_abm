import numpy as np
from scipy.optimize import minimize



def normal_likelihood(x, mu, s):
	"""
	Loglikelihood of a normal distribution
	"""
	x = np.array(x)
	return (1/((2 * np.pi)**0.5 * s)) * np.exp( - (x - mu)**2 / (2*s**2))

def trade_ll(X, P, alpha, mu_i, mu_n, sig_int, sig_noise):
	"""
	Likelihood of a trade X occurring	
	"""
	X = np.array(X)
	P = np.array(P)
	mu_n = np.array(mu_n)

	MU_i = (mu_i - P) / (1e-2*0.01)
	s_i = ((1/(1e-2*0.01))**2 * sig_int)**0.5

	MU_n = (mu_n - P) / (1e-2*0.01)

	s_n = ((1/(1e-2*0.01))**2 * sig_noise)**0.5
	L = alpha * normal_likelihood(X, MU_i, s_i) + (1-alpha)*normal_likelihood(X, MU_n, s_n)
	return np.log(L)

def max_ll(X, P, alpha, mu_n, sig_int, sig_noise):
	obj_func = lambda mu_i: -np.sum(trade_ll(X, P, alpha, mu_i, mu_n, sig_int, sig_noise))	
	res = minimize(obj_func, 0.5, method='nelder-mead')
	return res

class Market(object):
	def __init__(self, price_history, prob_noise, sig_noise, a_int, a_noise):
		self.price_history = price_history[:]
	
		self.time = 0
		self.inventory = 0
		self.outstanding_trades = []
		self.orders = []

		self.prob_noise = prob_noise
		self.sig_noise = sig_noise
		self.a_int = a_int
		self.a_noise = a_noise

		self.avg_sizes=[]
		self.market_prices = [price_history[-1]]

		self.prior = [0.33, 0.33, 0.34, 0.33,0.33,0.34]
		self.mu_noises=[]


	def get_last_price(self):
		return self.price_history[-1]

	def submit_trade(self, x, trader_name):
		self.outstanding_trades.append(x)
		self.orders.append([self.time, self.get_last_price(), x, trader_name])

		# Every 10 trades the market maker updates the price and clears the market
		if len(self.outstanding_trades) >= 10:
			self.update_price()

	def get_mu_noise(self):
		self.mu_noise = self.get_last_price()
		return self.get_last_price()

	def update_price(self):
		self.time += 1
		p = self.get_last_price()
		net_pos = np.sum(self.outstanding_trades)
		self.inventory += - net_pos
		avg_size = 1. * net_pos / len(self.outstanding_trades)
		self.avg_sizes.append(avg_size)
		self.mu_noises.append(self.get_mu_noise())
		
		n_back = 1
		avg_sizes = self.avg_sizes[-n_back:]
		market_prices = self.market_prices[-n_back:]
		mu_noises = self.mu_noises[-n_back:]


		res = MaxEnt_market_maker(market_prices, avg_sizes, 
			self.prob_noise, mu_noises, 
			self.sig_noise, self.a_int, self.a_noise, self.prior)
		if not res.success:
			print res.message
		self.prior = res.x
		mu_hat = np.sum(res.x[0:3] * np.array([0.01, 0.5, 0.99]))
		sig_hat = np.sum(res.x[3:6] *  np.array([0.01, 0.1, 0.19]))
		if mu_hat > p and mu_hat - p > 0.02:
			mu_hat = p + 0.02
		if mu_hat < p and p - mu_hat > 0.02:
			mu_hat = p - 0.02

		self.price_history.append(mu_hat)
		self.market_prices.append(mu_hat)
		self.outstanding_trades = []
