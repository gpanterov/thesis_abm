import numpy as np
import pandas as pd
import ABMtools as tools
reload(tools)
import matplotlib.pyplot as plt
import scipy.optimize as opt


class params_class(tools.default_params):
	def __init__(self):
		pass

params = params_class()

p_start = 40.
num_trades = 500
informed_prices = [37., 42., 40., 43] 
price_durations = [120, 100, 130]
price_history, Inv, X, U, Y = tools.raw_data(informed_prices, price_durations,
										p_start, num_trades, params)
# Plot

#model =  tools.SimpleLikelihood(price_history, Inv, tools.simple_likelihood, 
#							num_trades, params_class)
#res = model.optimize_all_set()
#plt.plot(price_history)
#
#price_durations = model.p_durations
#num_prices = len(price_durations) + 1
#informed_prices = res.x[-num_prices:]
#p_start = price_history[0]
#params = params_class()
#params.Lambda = res.x[0]
#params.Sigma_u = res.x[2]
#params.Sigma_0 = res.x[3]
#price_history2, Inv2, X, U, Y = tools.raw_data(informed_prices, price_durations,
#										p_start, num_trades, params)
#plt.plot(price_history2)
#plt.show()

#D = tools.likelihood_ratio(37., 50., 120, tools.simple_likelihood,
#							price_history[:150],Inv[:150], params)

def func_vol(Sigma, price_history, Vol, informed_prices, price_durations,
			params):
	V = np.mean(Vol)
	T = len(Vol)
	P_last = price_history[:-1]
	P = tools.create_price_vector(informed_prices, price_durations, T)
	numerator = np.sum(np.abs(P - P_last))
	z = params.Lambda/params.y_bar
	a = params.alpha
	w = np.sqrt(2/np.pi)
	denom = 2 * z + a*params.Sigma_0 + a*z**2*Sigma
	res = Sigma**0.5 * w + (1./T) * (1./denom) * numerator - V
	return res

Res = []
for j in range(100):
	price_history, Inv, X, U, Y = tools.raw_data(informed_prices, price_durations,
											p_start, num_trades, params)

	Vol = np.abs(X) + np.abs(U)

	x = opt.fsolve(func_vol, 10., args=(price_history, Vol, 
					informed_prices, price_durations,params))
#	res = func_vol(params.Sigma_u, price_history, Vol, informed_prices,
#					price_durations, params)
	Res.append(x)

print np.mean(Res)
