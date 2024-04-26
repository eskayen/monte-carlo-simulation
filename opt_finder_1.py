import numpy as np
import time

# runtime(seconds) = years * sims / 100
NUM_SIMS = 1000 # adjust if too slow
INITIAL_MONEY = 1000000
NUM_YEARS = 3 # result will depend on years
P_THRESH = 0.05 # desired min percent profit
P_PROB = 0.5 # desired min probability of reaching the profit
L_THRESH = 0.1 # desired max percent loss
L_PROB = 0.1 # desired min probability of reaching the loss

# adjustable parameters: profit threshold & probability, loss threshold & probability, time span

#SIMULATOR
def sim(percent_A, percent_B, percent_C, percent_D):

    stock_allocs = {
        'A': percent_A / 100,
        'B': percent_B / 100,
        'C': percent_C / 100,
        'D': percent_D / 100
    }

    # set up parameters of the four stocks
    stock_params = {
        'A': {'dist': 'unif', 'a': -0.4, 'b': 0.6},
        'B': {'dist': 'unif', 'a': -0.1, 'b': 0.26},
        'C': {'dist': 'norm', 'mean': 0.08, 'stdev': 0.04},
        'D': {'dist': 'unif', 'a': 0.06, 'b': 0.08}
    }

    # perform monte carlo simulation
    portfolio_values = []  # array to contain total money returned from each simulation

    # profit and loss threshold amounts
    profit_val = INITIAL_MONEY * (1 + P_THRESH)
    prob_profit = 0
    loss_val = INITIAL_MONEY * (1 - L_THRESH)
    prob_loss = 0

    # loop through simulations
    for _ in range(NUM_SIMS):
        portfolio_value = INITIAL_MONEY  # the initial investment for a year / the value of the entire portfolio after all years
        # loop through years
        for _ in range(NUM_YEARS):
            yearly_yield = 0
            # loop through stocks
            for (stock, params) in stock_params.items():
                # randomly generate a return rate for the stock
                if params['dist'] == 'unif':
                    return_rate = np.random.uniform(params['a'], params['b'])
                elif params['dist'] == 'norm':
                    return_rate = np.random.normal(params['mean'], params['stdev'])
                # add stock yield (how much of the year's initial investment went to the stock times how much of the stock's investment was gained or lost) to the entire year's yield
                yearly_yield += portfolio_value * stock_allocs[stock] * return_rate
            # END STOCKS LOOP
            # determine the portfolio's value after a year / the initial investment for next year
            portfolio_value += yearly_yield
        # END YEARS LOOP
        # determine whether the final portfolio value was a profit or loss based on user thresholds
        if portfolio_value >= profit_val:
            prob_profit += 1 / NUM_SIMS
        if portfolio_value >= loss_val and portfolio_value < INITIAL_MONEY:
            prob_loss += 1 / NUM_SIMS
        # store the value of the entire portfolio after each simulation
        portfolio_values.append(portfolio_value)
    # END SIMS LOOP

    # analyze results
    mean_profit = np.mean(portfolio_values)
    std_dev = np.std(portfolio_values)

    return (mean_profit, std_dev, prob_profit, prob_loss)
# END sim


print(f'\nPortfolio with\n\tat least {P_PROB * 100}% chance of at least {P_THRESH * 100}% profit,\n\tat least {L_PROB * 100}% chance of at most {L_THRESH * 100}% loss,\n\tand highest Sharpe ratio after {NUM_YEARS} years:')

highest_sharpe = 0
portfolio = ()

start_time = time.time()

for percent_A in range(0, 101, 5):
    for percent_B in range(0, 101 - percent_A, 5):
        for percent_C in range(0, 101 - percent_A - percent_B, 5):
            for percent_D in range(0, 101 - percent_A - percent_B - percent_C, 5):
                if (percent_A + percent_B + percent_C + percent_D == 100):
                    # run simulation 1000 times for current permutation and get sharp ratio
                    (mean, stdev, p_prob, l_prob) = sim(percent_A, percent_B, percent_C, percent_D)
                    sharpe = mean / stdev
                    # if the simulation's probability of reaching/exceeding the desired profit reaches/exceeds the desired probability
                    # and the simulation's probability of reaching/exceeding the desired loss reaches/falls short of the desired probability
                    # and has the highest sharpe ratio
                    if (p_prob >= P_PROB and l_prob >= L_PROB and sharpe >= highest_sharpe):
                        highest_sharpe = sharpe
                        portfolio = (percent_A, percent_B, percent_C, percent_D, sharpe)

end_time = time.time()
runtime = end_time - start_time

if(portfolio == ()): portfolio = 'Does Not Exist'
print(f'\t\t{portfolio}\n')
print(f"Runtime: {runtime} seconds")