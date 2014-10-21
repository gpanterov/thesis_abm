import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
##############
# Parameters #
##############
p = 37.  # starting price

# Noise Trader
Sigma_u = 10 ** 2 # Variance of noise trader order flow

# Market Maker
Inv = [0]
Lambda = 0.5  
phi = -0.2
y_bar = 10.
I_bar = 10.

def compute_price(x_t, u_t, I_, p):
	return p + (Lambda/y_bar) * (x_t + u_t) + phi * (I_ / I_bar)

# Informed trader
p0 = 40. 
Sigma_0 = 1.5 ** 2
alpha = 5e-2 # risk aversion
Sigma_MM = (Lambda / y_bar)**2 * Sigma_u


# Informed trader optimal order
def compute_x_star(p_last, Lambda, Sigma_0, Sigma_MM, alpha, p0):
	""" Return optimal trade for informed trader """
	return (p0 - p_last) / ((2 * Lambda / y_bar) + \
				alpha * (Sigma_0 + Sigma_MM))
	

def gen_data(num_trades, p_start, Lambda, Sigma_0, Sigma_u, alpha, P0):
	price_history = [p_start]
	X=[]
	U = []
	Y = []
	Sigma_MM = (Lambda / y_bar)**2 * Sigma_u

	Inv = [0]

	for p0 in P0:
		for t in range(num_trades):
			# Noise trader order
			p_last = price_history[-1]
			u_t = np.random.normal(0, Sigma_u**0.5)
			x_t = compute_x_star(p_last,Lambda, Sigma_0, Sigma_MM, alpha, p0)
			I_last = Inv[-1]
			p_new = compute_price(x_t, u_t, I_last, p_last)
			price_history.append(p_new)
			I_new = I_last - (x_t + u_t)
			Inv.append(I_new)
			X.append(x_t)
			U.append(u_t)
			Y.append(x_t + u_t)
	return price_history, Inv, X, U, Y


# Estimation
def likelihood(p, p_last, Lambda, phi, Sigma_u, I_last, 
							Sigma_0, Sigma_MM, alpha, p0):
	x_star = compute_x_star(p_last, Lambda, Sigma_0, Sigma_MM, alpha, p0)
	mu_dp = (Lambda / y_bar) * x_star + phi * I_last/I_bar
	Sigma_dp = (Lambda / y_bar)**2 * Sigma_u
	ln_f = -0.5 * np.log(2*np.pi) - np.log(Sigma_dp**0.5) - \
			(p - mu_dp)**2 / (2*Sigma_dp)  
	return ln_f 


def obj_func(params, P, Inv) :
	Lambda = params[0]
	phi = params[1]
	Sigma_u = params[2]
#	Sigma_0 = params[2]
	p0 = params[3]

	p = np.array(P[1:])
	p_last = np.array(P[:-1])
	delta_p = p - p_last
	I_last = np.array(Inv[:-1])

	Sigma_MM = (Lambda / y_bar)**2 * Sigma_u
	
	res = likelihood(delta_p, p_last, Lambda, phi, Sigma_u, I_last,
				Sigma_0, Sigma_MM, alpha, p0)
	return np.sum(res) 

def optimize(P, Inv):
	func = lambda x: -obj_func(x, P, Inv) 
	x0 = [0.3,-0.1, 50,50]
	#x0 = [Lambda, phi, Sigma_u, Sigma_0, p0]
	res = minimize(func, x0, method='nelder-mead',
		options={'xtol':1e-8, 'disp':True, 'maxfev':5000, 'maxiter':5000} )
	return res

def resample_series(series, sampling_freq=5):
	series = np.array(series)
	series = pd.Series(series)	
	freq = series.index / sampling_freq
	series = series.groupby([freq]).mean()
	return series

# Generate data

num_trades=100
#P0 = [37., 41., 38.]
P0 = [37]
price_history, Inv, X, U, Y = gen_data(num_trades, p, Lambda, Sigma_0, 
								Sigma_u, alpha, P0)

# Estimate
res = optimize(price_history, Inv)
print res.x

Lambda_hat = res.x[0]
phi_hat = res.x[1]
Sigmau_hat = res.x[2]
p0_hat = res.x[3]

# Plot

Vol = np.abs(X) + np.abs(U)
sampling_freq=1
prices = resample_series(price_history, sampling_freq)
volume = resample_series(Vol, sampling_freq)
oflow = resample_series(Y, sampling_freq)
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax1.plot(prices)
ax1.set_title("Price History")

ax2 = fig.add_subplot(3,1,2)
ax2.set_title("Volume")
ax2.plot(volume)

ax3 = fig.add_subplot(3,1,3)
ax3.set_title("Net order flow")
ax3.plot(oflow)

#new_price, new_Inv, new_X, new_U, new_Y = gen_data(num_trades, p, Lambda_hat,
#							Sigma_0, Sigmau_hat, p0_hat)
#plt.plot(new_price) 
plt.show()


