import numpy as np
import matplotlib.pyplot as plt
import BayesianTools as btools
reload(btools)
import ABMtools as abmtools
reload(abmtools)
from scipy.optimize import minimize
from scipy.stats import norm
import time
import sandbox as sbox
reload(sbox)
import EstimateTools as etools
reload(etools)


class params_class(abmtools.default_params):
	def __init__(self):
		pass

params = params_class()
params.Lambda = 0.025
params.Sigma_0 = 10
p_start = 37.
num_trades = 300
#informed_prices = [37.3, 41., 37., 35.2] 
informed_prices = [40.1, 40. ,40.1, 39.9]
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

x_true = [Lambda, alpha, Sigma_u, Sigma_e, Sigma_0]
x_true.extend(informed_prices)

nu = np.random.normal(0, params.Sigma_n**0.5, (len(Y),1))
y = Y + nu
vol = np.abs(X) + np.abs(U)



d = num_trades / len(informed_prices)
#d=95
#price_durations_hat = [d] * (len(informed_prices) - 1)
price_durations_hat = price_durations
model = etools.TradingModel(price_history, vol, price_durations_hat)
print "Model is using price durations: ", model.price_durations
model.MCMC()
model.MAP()
print [round(i,2) for i in model.posterior_means]
print [round(i,2) for i in model.map_estimate]


# Display some series stats #
print "\n"
print "Series stats"
print "-------------"
print "|X|: ", np.std(np.abs(X))
print "|U|: ", np.std(np.abs(U))
print "Volume: ", np.std(vol)
print "Price change volatility: ", np.std(pdelta)
print "-------------"
print "\n"

plt.close('all')
plt.figure(1)
plt.plot(price_history)
plt.plot(P_informed)
plt.show()


x = model.map_estimate
Lambda_hat = x[0]
alpha_hat = x[1]
Sigma0_hat = x[2]
Sigmau_hat = x[3]
Sigmae_hat = x[4]

window_size=20
Xsample, Xsample_sd = etools.get_trading_signals(price_history, vol,  Lambda_hat, alpha_hat, 
						Sigma0_hat, Sigmau_hat, Sigmae_hat, window_size)

plt.close(1)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(price_history[window_size:])
plt.plot(Xsample[:,1])

plt.subplot(3,1,2)
plt.plot(Xsample_sd[:,1])

plt.subplot(3,1,3)
plt.plot(np.abs(Xsample[:,1] - Xsample[:,0]))
plt.show()

plt.figure(3)
plt.plot(P_informed[window_size:])
plt.plot(Xsample[:,1])
plt.show()



