import numpy as np
import matplotlib.pyplot as plt
import BayesianTools as btools
reload(btools)
import ABMtools as abmtools
reload(abmtools)
from scipy.optimize import minimize
from scipy.stats import norm
import time


class params_class(abmtools.default_params):
	def __init__(self):
		pass

params = params_class()

p_start = 40.
num_trades = 300
informed_prices = [37., 42.,30., 35] 
price_durations = [100, 80, 90]



price_history, Inv, X, U, Y = abmtools.raw_data(informed_prices, price_durations,
										p_start, num_trades, params)

P_informed = abmtools.create_price_vector(informed_prices, price_durations, num_trades)
P_last = price_history[:-1]
pdelta = np.array(price_history[1:]) - P_last
Lambda = params.Lambda
alpha = params.alpha
y_bar = params.y_bar
Sigma_u = params.Sigma_u
Sigma_0 = params.Sigma_0
Sigma_e = params.Sigma_e
Sigma_n = params.Sigma_n


nu = np.random.normal(0, params.Sigma_n**0.5, (len(Y),1))
y = Y + nu
vol = np.abs(X) + np.abs(U)

pdelta = pdelta
P_informed = P_informed
P_last = P_last

lfunc2 = lambda x: btools.likelihood2(pdelta, y,  
		x[0], x[1], x[2], x[3], x[4], P_informed, P_last, Sigma_0, y_bar) \
		+ np.log(norm.pdf(x[0], 0.5, 0.1)) \
		+ np.log(norm.pdf(x[1], 5e-2, 1e-1)) \
		+ np.log(norm.pdf(x[2], 100., 15.))  \
		+ np.log(norm.pdf(x[3], 0.02**2, 0.01)) \
		+ np.log(norm.pdf(x[4], 9, 3))


lfunc1 = lambda x: btools.likelihood1(pdelta, vol,  
		x[0], x[1], x[2], x[3], x[4], P_informed, P_last, Sigma_0, y_bar) \
		+ np.log(norm.pdf(x[0], 0.5, 0.1)) \
		+ np.log(norm.pdf(x[1], 5e-2, 1e-1)) \
		+ np.log(norm.pdf(x[2], 100., 15.))  \
		+ np.log(norm.pdf(x[3], 0.02**2, 0.01)) \
		+ np.log(norm.pdf(x[4], 9, 1))

lfunc_opt2 = lambda x: - lfunc2(x)

lfunc_opt1 = lambda x: - lfunc1(x)


x0 = [0.4, 6e-2, 90., 0.02**2, 10]
sigmas = [0.2, 6e-2, 20, 0.01, 5]

b = len(informed_prices) 

x0 += [40] * b
sigmas += [5] * b
N = 1000
sampling_func = lambda x: btools.sample1(x, price_history, vol,
					price_durations, Sigma_0, y_bar)

sample = btools.metropolis_hastings(x0, sigmas, sampling_func , N) 
res = minimize(btools.obj_func1, x0, method='nelder-mead', 
		args=(price_history, vol, price_durations,Sigma_0, y_bar))

print "Optimization was ", res.success
#x0 = [1., 1., 50., 0., 20.]
#sigmas = [0.2, 6e-2, 20, 0.01, 5]

#t = time.time()
#N = 1000
#sample = btools.metropolis_hastings(x0, sigmas, lfunc1, N) 
#print "It took ", time.time() - t, ' seconds to finish the sampling'
#
#
#t= time.time()
#res=minimize(lfunc_opt1, x0, method='nelder-mead')
#print "It took ", time.time() - t, ' to optimize'
#
#print res.success, res.x


