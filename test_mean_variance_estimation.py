"""
This script tests whether my calculations for the mean and variance on paper are correct
I simulate final wealth based on generated terminal price and compare the empirical with the
theoretical (estimated on paper) mean and variance.
The results are very close
"""

import numpy as np
import MarketInstitutions as mi
reload(mi)


a = 1e-3
x0 = np.array([10., 12., 3.])
p0 = np.array([75.3, 64.1, 30.0])
x = 13.
p = 27.0
W0 = 1000.
rf = 1.
mu = 35.
sigma2 = 3.


N = 10000
V = np.random.normal(mu, np.sqrt(sigma2), size = (N,))

def final_wealth(v, a, x0, p0, x, p, W0, rf, mu, sigma2):
	C = np.sum(x0 * v) - np.sum(x0 * p0) + x * (v - p) + (W0 - x * p) * rf
	return C

C = []
for v in V:
	C.append(final_wealth(v, a, x0, p0, x, p, W0, rf, mu, sigma2))
C = np.array(C)
mu_C_sim = np.mean(C)
sigma2_C_sim = np.var(C)

A = np.sum(x0 * p0)
B = (W0 - x * p) * rf

# Expected value of C**2
EC2 = (sigma2 + mu**2) * (np.sum(x0) + x) ** 2 + \
		2 * mu * (x + np.sum(x0)) * (B - A - p * x) + \
		(A - B + p * x) ** 2
mu_C = mu * np.sum(x0) - A + x * mu - x * p + B
sigma2_C = EC2 - mu_C**2

print "Trades can trade more than once:"
print "Simulated mean and variance: ", mu_C_sim, sigma2_C_sim
print "Estimated mean and variance: ", mu_C, sigma2_C
print "\n"

def final_wealth_norepeat(v, a, x, p, W0, rf, mu, sigma2):
	C =  x * (v - p) + (W0 - x * p) * rf
	return C

C = []
for v in V:
	C.append(final_wealth_norepeat(v, a, x, p, W0, rf, mu, sigma2))
C = np.array(C)
mu_C_sim = np.mean(C)
sigma2_C_sim = np.var(C)

B = (W0 - x * p) * rf

# Expected value of C**2
EC2 = (sigma2 + mu**2) * x**2 + (B - x * p) * (B - x * p + 2 * mu * x)
mu_C = mu * x - x * p + B
sigma2_C = EC2 - mu_C**2

print "Traders can trade only once: "
print "Simulated mean and variance: ", mu_C_sim, sigma2_C_sim
print "Estimated mean and variance: ", mu_C, sigma2_C

