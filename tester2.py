import numpy as np
import time

# runtime(seconds) = years * sims / 100
NUM_SIMS = 1000 # adjust if too slow
INITIAL_MONEY = 1000000
NUM_YEARS = 5 # result will depend on years

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
    profit_val = INITIAL_MONEY + 1
    loss_val = INITIAL_MONEY - 1
    prob_profit = 0
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
        # determine whether the final portfolio value was a profit or loss based on thresholds
        if portfolio_value >= profit_val and portfolio_value > INITIAL_MONEY:
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


print(f'\nPortfolio with highest mean and lowest risk after {NUM_YEARS} years:')

#highest_sharpe = 0
portfolio = ()
highest_mean = INITIAL_MONEY
highest_p = 0
lowest_l = 1

start_time = time.time()

for percent_A in range(0, 101, 5):
    for percent_B in range(0, 101 - percent_A, 5):
        for percent_C in range(0, 101 - percent_A - percent_B, 5):
            for percent_D in range(0, 101 - percent_A - percent_B - percent_C, 5):
                if (percent_A + percent_B + percent_C + percent_D == 100):
                    (mean, stdev, p_prob, l_prob) = sim(percent_A, percent_B, percent_C, percent_D)
                    # search for the permutation which has the highest mean and lowest chance of loss/highest chance of profit
                    if (mean >= highest_mean and p_prob >= highest_p and l_prob <= lowest_l):
                        highest_mean = mean
                        highest_p = p_prob
                        lowest_l = l_prob
                        portfolio = (percent_A, percent_B, percent_C, percent_D)

end_time = time.time()
runtime = end_time - start_time
print(f'\t\t{portfolio}\n\t\tProfit: ~{(highest_mean - INITIAL_MONEY) / INITIAL_MONEY * 100:.2f}%\n\t\tRisk: ~{lowest_l:.2f}%')
print(f"Runtime: {runtime} seconds")