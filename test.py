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
num_trades = 250
informed_prices = [37., 41., 38.] * 2
price_history, Inv, X, U, Y = tools.raw_data(informed_prices, 
										p_start, num_trades, params)

lfunc = tools.moments_likelihood
model = tools.SimpleLikelihood(price_history, Inv, 
					lfunc, num_trades, params_class)
res = model.optimize()

