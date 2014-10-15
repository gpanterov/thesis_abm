"""
This module generates trading data
"""
import numpy as np
import pandas as pd
import TraderTools as tools
reload(tools)


# seed
#np.random.seed(12345)

num_days = 3
num_trades = 300
start_price_history = [0.5,0.51,0.52,0.51]

P = tools.SimulationParams(num_days, num_trades, start_price_history)
prior = P.prior[:]
price_history = P.price_history[:]
mu_i = P.mu_i
P.shocks_i = [0.01,0.01,0.01]

######################
# Trading Simulation #
######################

#BaseRet, BaseVol = tools.simulate(P, num_sim=1, num_days=num_days, num_trades=num_trades)
BaseRet = tools.simulate2(P, 1, num_days, num_trades)

true_params = P.shocks_i + [P.alpha]
num_sim = 20

models = []
shocks = [0.01, -0.01]
for shock1 in shocks:
	for shock2 in shocks:
		for shock3 in shocks:
			for alpha in np.arange(0.2,0.8,0.1):
				alpha = round(alpha, 1)
				P = tools.SimulationParams(num_days, num_trades, start_price_history)
				P.alpha = alpha
				P.alpha_mm = alpha
				P.shocks_i = [shock1, shock2, shock3]
				
				#SimRet, SimVol = tools.simulate(P, num_sim, num_days, num_trades)
				SimRet = tools.simulate2(P, num_sim, num_days, num_trades)
				mse_ret = np.mean((SimRet.values - BaseRet.values)**2, axis=0)
				current_fit = np.mean(mse_ret)
				current_params = P.shocks_i + [P.alpha]
				#current_fit = Res2['mad_ret']['mean']
				#current_vol = Res2['mad_vol']['mean']
				models.append(current_params + [current_fit])
				#print current_params, current_fit
models = pd.DataFrame(models)
sorted_models = models.sort(column=models.columns[-1])
sorted_models['rank'] = range(len(models))

print true_params
print "--" * 20
print sorted_models
#print "True params are: ", true_params
#print "Estimated params are: ", best_params

