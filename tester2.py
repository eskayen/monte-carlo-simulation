import numpy as np
import time

# runtime(seconds) = years * sims / 100
NUM_SIMS = 1000 # adjust if too slow
INITIAL_MONEY = 1000000
NUM_YEARS = 1 # result will depend on years
STEP_SIZE = 5 # step size of permutations
ALLOWED_RISK = 0.2 # chance that you will lose any amount of money

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
    risk = 0

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
        # determine whether the final portfolio value was a profit or loss based on thresholds
        if portfolio_value < INITIAL_MONEY:
            risk += 1 / NUM_SIMS
        # store the value of the entire portfolio after each simulation
        portfolio_values.append(portfolio_value)
    # END SIMS LOOP

    # analyze results
    mean_profit = np.mean(portfolio_values)
    std_dev = np.std(portfolio_values)

    return (mean_profit, std_dev, risk)
# END sim


print(f'\nPortfolio with highest mean and lowest risk after {NUM_YEARS} years (max allowed risk = {ALLOWED_RISK * 100:.0f}%):')

#highest_sharpe = 0
portfolio = ()
highest_mean = INITIAL_MONEY
lowest_risk = 1

start_time = time.time()

for percent_A in range(0, 101, STEP_SIZE):
    for percent_B in range(0, 101 - percent_A, STEP_SIZE):
        for percent_C in range(0, 101 - percent_A - percent_B, STEP_SIZE):
            for percent_D in range(0, 101 - percent_A - percent_B - percent_C, STEP_SIZE):
                if (percent_A + percent_B + percent_C + percent_D == 100):
                    (mean, stdev, risk) = sim(percent_A, percent_B, percent_C, percent_D)
                    # search for the permutation which has the highest mean and lowest chance of loss/highest chance of profit
                    if (mean >= highest_mean and risk >= 0 and risk <= ALLOWED_RISK):
                        highest_mean = mean
                        lowest_risk = risk
                        portfolio = (percent_A, percent_B, percent_C, percent_D)
                        

end_time = time.time()
runtime = end_time - start_time
print(f'\t\t{portfolio}\n\t\tExpected Profit: ~{(highest_mean - INITIAL_MONEY) / INITIAL_MONEY * 100:.2f}%\n\t\tChance of Risk: ~{lowest_risk:.2f}%')
print(f"Runtime: {runtime} seconds")