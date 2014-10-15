import numpy as np
import pandas as pd
from scipy.optimize import minimize

##############
# Parameters #
##############
p = 47.  # starting price

# Noise Trader
Sigma_u = 10 ** 2 # Variance of noise trader order flow

# Market Maker
Inv = [0]
Lambda = 0.2  
phi = -0.2
y_bar = 10.
I_bar = 10.

def compute_price(x_t, u_t, I_, p):
	return p + (Lambda/y_bar) * (x_t + u_t) + phi * (I_ / I_bar)

# Informed trader
p0 = 40. 
Sigma_0 = 3 ** 2
alpha = 1e-2  # risk aversion
Sigma_MM = (Lambda / y_bar)**2 * Sigma_u


# Informed trader optimal order
def compute_x_star(p_last, Lambda, Sigma_0, Sigma_MM, p0):
	""" Return optimal trade for informed trader """
	return (p0 - p_last) / ((2 * Lambda / y_bar) + \
				alpha * (Sigma_0 + Sigma_MM))
	

price_history = [p]
X=[]
U = []
Y = []
num_trades = 1000
for t in range(num_trades):
	# Noise trader order
	p_last = price_history[-1]
	u_t = np.random.normal(0, Sigma_u**0.5)
	x_t = compute_x_star(p_last,Lambda, Sigma_0, Sigma_MM, p0)
	I_last = Inv[-1]
	p_new = compute_price(x_t, u_t, I_last, p_last)
	price_history.append(p_new)
	I_new = I_last - (x_t + u_t)
	Inv.append(I_new)
	X.append(x_t)
	U.append(u_t)
	Y.append(x_t + u_t)

# Estimation
def likelihood(p, p_last, Lambda, phi, Sigma_u, I_last, 
							Sigma_0, Sigma_MM, p0):
	x_star = compute_x_star(p_last, Lambda, Sigma_0, Sigma_MM, p0)
	mu_dp = (Lambda / y_bar) * x_star + phi * I_last/I_bar
	Sigma_dp = (Lambda / y_bar)**2 * Sigma_u
	ln_f = -0.5 * np.log(2*np.pi) - np.log(Sigma_dp**0.5) - \
			(p - mu_dp)**2 / (2*Sigma_dp)  
	return ln_f 


def obj_func(params, P, Inv) :
	Lambda = params[0]
	phi = params[1]
	Sigma_u = params[2]
	Sigma_0 = params[3]
	p0 = params[4]

	p = np.array(P[1:])
	p_last = np.array(P[:-1])
	delta_p = p - p_last
	I_last = np.array(Inv[:-1])

	Sigma_MM = (Lambda / y_bar)**2 * Sigma_u
	
	res = likelihood(delta_p, p_last, Lambda, phi, Sigma_u, I_last,
				Sigma_0, Sigma_MM, p0)
	return np.sum(res) 

def optimize(P, Inv):
	func = lambda x: -obj_func(x, P, Inv) 
	x0 = [0.3,-0.1, 50, 10, 40]
	#x0 = [Lambda, phi, Sigma_u, Sigma_0, p0]
	res = minimize(func, x0, method='nelder-mead',
		options={'xtol':1e-8, 'disp':True, 'maxfev':5000, 'maxiter':5000} )
	return res

res = optimize(price_history, Inv)
true_params = [Lambda, phi, Sigma_u, Sigma_0, p0]
print true_params
print res.x
Lambda1 = res.x[0]
phi1=res.x[1]
Sigma_u1=res.x[2]
Sigma_01 = res.x[3]
p01 = res.x[4]
Sigma_MM1 = (Lambda1 / y_bar)**2 * Sigma_u1

n=5
p = price_history[n]
p_last = price_history[n-1]
I_last = Inv[n-1]
delta_p = p - p_last

l = likelihood(p, p_last, Lambda, phi, Sigma_u, I_last, 
							Sigma_0, Sigma_MM, p0)
l1 = likelihood(p, p_last, Lambda1, phi1, Sigma_u1, I_last, 
							Sigma_01, Sigma_MM1, p01)

