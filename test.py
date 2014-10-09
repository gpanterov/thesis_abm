import numpy as np
import Traders as trd
reload(trd)
import Institutions as inst
reload(inst)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
from bisect import bisect
import bayes as bayes
reload(bayes)
from scipy.stats import norm

##############
# Parameters #
##############
p = 0.5
mu_i = 0.7
mu_n = 0.40
mu_n_hat = 0.4
alpha = 0.6
alpha_hat = 0.4

# SImulatiohn 

b = bayes.BayesDiscrete()
#prior = [0.1, 0.25, 0.3, 0.25, 0.1]
prior = bayes.compute_uniform_prior(100)

#p = np.linspace(0, 1, 100)
#p = np.concatenate((p,p,p))
#np.random.shuffle(p)

p = [0.5]
ntrades = 200
trades = []
entropies=[]
for i in range(ntrades):
	if p[i] > 0.6:
		mu_n = 0.3
	elif p[i] < 0.4:
		mu_n = 0.7
	else:
		mu_n = p[i]
	if i > 50:
		mu_i = 0.5
	if np.random.uniform() <alpha:
		prob_buy = (1 - norm.cdf(p[i], mu_i, 0.01)) 
	else:
		prob_buy = (1 - norm.cdf(p[i], mu_n, 0.01))
	
	trade = np.random.binomial(1,prob_buy)
	trades.append(trade)
	prior = b.calculate_posterior(trades[i], prior, p[i], p[i], alpha_hat)
	p.append(np.sum(prior * b.support))
	entropies.append(np.sum(-prior*np.log2(prior + 1e-5)))





