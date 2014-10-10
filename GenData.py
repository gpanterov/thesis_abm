"""
This module generates trading data
"""
import numpy as np
import pandas as pd
import TraderTools as tools
reload(tools)


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

BaseRet, BaseVol = tools.simulate(P, num_sim=1, num_days=num_days, num_trades=num_trades)

num_sim = 2
P = tools.SimulationParams(num_days, num_trades, start_price_history)
P.alpha = 1.
P.alpha_mm = .3
#P.ar_params_mm = [-0.3, 0.1, -0.2]
#P.shocks_i = [0.02, -0.01] * (num_days/2)
SimRet, SimVol = tools.simulate(P, num_sim, num_days, num_trades)

Res = tools.fit(BaseRet, BaseVol, SimRet, SimVol)
