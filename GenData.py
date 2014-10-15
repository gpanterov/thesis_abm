import numpy as np
import pandas as pd


##############
# Parameters #
##############
p = 47.  # starting price

# Noise Trader
Sigma_u = 10 ** 2 # Variance of noise trader order flow

# Market Maker
Inv = [0]
Lambda = 0.3  
phi = -0.01
y_bar = 10.
I_bar = 10.

def compute_price(x_t, u_t, I_, p):
	return p + Lambda/y_bar * (x_t + u_t) + phi * (I_ / I_bar)

# Informed trader
p0 = 45. 
Sigma_0 = 3 ** 2
alpha = 1e-2  # risk aversion
Sigma_MM = (Lambda / y_bar)**2 * Sigma_u


# Informed trader optimal order
def compute_x_star(p):
	""" Return optimal trade for informed trader """
	return (p0 - p) / ((2 * Lambda / y_bar) + alpha * (Sigma_0 + Sigma_MM))
	

price_history = [p]
X=[]
U = []
Y = []
num_trades = 100
for t in range(num_trades):
	# Noise trader order
	p_last = price_history[-1]
	u_t = np.random.normal(0, Sigma_u**0.5)
	x_t = compute_x_star(p_last)
	I_last = Inv[-1]
	p_new = compute_price(x_t, u_t, I_last, p_last)
	price_history.append(p_new)
	I_new = I_last - (x_t + u_t)
	Inv.append(I_new)
	X.append(x_t)
	U.append(u_t)
	Y.append(x_t + u_t)

