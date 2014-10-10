"""
This module generates trading data
"""
import numpy as np
import pandas as pd
import TraderTools as tools
reload(tools)


def day_trade(P, num_trades, mu_i, prior, price_history, Records):
	daily_prices = [price_history[-1]]
	for i in range(num_trades):
		# Noise traders update prices
		mu_n = tools.AR(P.ar_params, price_history)

		# Generate a trade
		trade = tools.generate_trade(P, price_history[-1], mu_i, mu_n)

		# Market maker updates prices
		mu_i_mm, prior = tools.MM_update(P, trade, prior, price_history)

		if i%5==0 and i > 1:
			if mu_i_mm > price_history[-1]:
				new_price = min(price_history[-1] + 3*P.tick, mu_i_mm)
			else:
				new_price = max(price_history[-1] - 3*P.tick, mu_i_mm)
			new_price = np.round(new_price, 2)
			price_history.append(new_price)
			daily_prices.append(new_price)

		# Keep recod
		Records.mu_i.append(mu_i)
		Records.entropies.append(np.sum(-prior*np.log2(prior + 1e-5)))
		Records.mu_i_mm.append(mu_i_mm)
		Records.all_trades.append(trade)

	# calculates daily performance
	daily_returns = np.array(daily_prices[1:]) - np.array(daily_prices[:-1])
	Records.closing_price.append(daily_prices[-1])
	Records.opening_price.append(daily_prices[0])
	Records.daily_return.append(Records.closing_price[-1] - Records.opening_price[-1])
	Records.daily_volatility.append(np.std(daily_returns))


	return prior, price_history


# seed
#np.random.seed(12345)
num_days = 10
num_trades = 200
start_price_history = [0.5,0.51,0.52,0.51]

P = tools.SimulationParams(num_days, num_trades, start_price_history)
prior = P.prior[:]
price_history = P.price_history[:]
mu_i = P.mu_i


######################
# Trading Simulation #
######################
BaseRecords = tools.SimulationRecords()

for day in range(num_days):
	# new information arrives to the informed traders
	mu_i = mu_i + P.shocks_i[day]
	prior, price_history = day_trade(P, num_trades, mu_i, prior, price_history, BaseRecords)

BaseRet = pd.Series(index = range(num_days), data=BaseRecords.daily_return)
BaseVol = pd.Series(index = range(num_days), data=BaseRecords.daily_volatility)

num_sim = 5
SimRet = pd.DataFrame(index = range(num_days), columns = range(num_sim))
SimVol = pd.DataFrame(index = range(num_days), columns = range(num_sim))
for i in range(num_sim):
	print "Simulation ", i	
	Records = tools.SimulationRecords()
	for day in range(num_days):
		mu_i = mu_i + P.shocks_i[day]
		prior, price_history = day_trade(P, num_trades, mu_i, prior, price_history, Records)
	SimRet[i] = Records.daily_return 
	SimVol[i] = Records.daily_volatility

mse_ret = np.abs(SimRet.sub(BaseRet, axis=0))
mse_ret = mse_ret.mean(axis=0)

mse_vol = np.abs(SimVol.sub(BaseVol, axis=0))
mse_vol = mse_vol.mean(axis=0)

Res = pd.DataFrame(index=range(num_sim), columns=['mse_ret', 'mse_vol'])
Res['mse_vol'] = mse_vol
Res['mse_ret'] = mse_ret

print Res
