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
num_trades = 300
informed_prices = [37., 42.,] 
price_durations = [200.]
price_history, Inv, X, U, Y = tools.raw_data(informed_prices, price_durations,
										p_start, num_trades, params)

best_fun = np.inf
best_duration = 0
best_res = 0
for i in range(50, 250, 32):
	model = tools.SimpleLikelihood(price_history, Inv, tools.simple_likelihood, 
							num_trades, [i], params_class)
	res = model.optimize()
	print res.x
	if res.fun < best_fun:
		best_fun = res.fun
		best_duration = i
		best_res = res
