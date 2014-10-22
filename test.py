import numpy as np
import pandas as pd
import ABMtools as tools
reload(tools)
import matplotlib.pyplot as plt

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

model =  tools.SimpleLikelihood(price_history, Inv, tools.simple_likelihood, 
							num_trades, params_class)
res = model.optimize_all_set()
plt.plot(price_history)

price_durations = model.p_durations
num_prices = len(price_durations) + 1
informed_prices = res.x[-num_prices:]
p_start = price_history[0]
params = params_class()
params.Lambda = res.x[0]
params.Sigma_u = res.x[2]
params.Sigma_0 = res.x[3]
price_history2, Inv2, X, U, Y = tools.raw_data(informed_prices, price_durations,
										p_start, num_trades, params)
plt.plot(price_history2)
plt.show()

D = tools.likelihood_ratio(37., 50., 120, tools.simple_likelihood,
							price_history[:150],Inv[:150], params)
