from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd

# initial macros
INITIAL_INVESTMENT = 1000000
NUM_SIMULATIONS = 1000

## GLOBALS
# user entries
start_money = 0
num_sims = 0
num_years = 0
profit_threshold = 0
profit_prob = 0
loss_threshold = 0
loss_prob = 0

# stock parameters
stock_params = {
    'A': {'dist': 'unif', 'a': -0.4, 'b': 0.6},
    'B': {'dist': 'unif', 'a': -0.1, 'b': 0.26},
    'C': {'dist': 'norm', 'mean': 0.08, 'stdev': 0.04},
    'D': {'dist': 'unif', 'a': 0.06, 'b': 0.08}
}

# functions for updating stock labels for A, B, C, and D
def update_label_A(val):
    val = int(val)
    # if total of all four slider values are 100 or less, update this label
    if (val + scale_B.get() + scale_C.get() + scale_D.get()) <= 100:
        portion = int(entry_start_money.get()) * val // 100
        label_A.config(text=f'Allocation for Stock A: {val}% (${portion})')
    # if total would exceed 100, don't update this label
    else:
        scale_A.set(100 - (scale_B.get() + scale_C.get() + scale_D.get()))

def update_label_B(val):
    val = int(val)
    if (scale_A.get() + val + scale_C.get() + scale_D.get()) <= 100:
        portion = int(entry_start_money.get()) * val // 100
        label_B.config(text=f'Allocation for Stock B: {val}% (${portion})')
    else:
        scale_B.set(100 - (scale_A.get() + scale_C.get() + scale_D.get()))

def update_label_C(val):
    val = int(val)
    if (scale_A.get() + scale_B.get() + val + scale_D.get()) <= 100:
        portion = int(entry_start_money.get()) * val // 100
        label_C.config(text=f'Allocation for Stock C: {val}% (${portion})')
    else:
        scale_C.set(100 - (scale_A.get() + scale_B.get() + scale_D.get()))

def update_label_D(val):
    val = int(val)
    if (scale_A.get() + scale_B.get() + scale_C.get() + val) <= 100:
        portion = int(entry_start_money.get()) * val // 100
        label_D.config(text=f'Allocation for Stock D: {val}% (${portion})')
    else:
        scale_D.set(100 - (scale_A.get() + scale_B.get() + scale_C.get()))

# function for displaying the distribution graphs of all stocks in one figure
def display_stocks():
    (_, axes) = plt.subplots(2, 2)
    (xmin, xmax) = -50, 70

    x1 = np.linspace(-40, 60, 1000)
    axes[0, 0].hist(x1, density=True)
    axes[0, 0].set_xlim(xmin, xmax)
    axes[0, 0].set_ylim(0, 0.1)
    axes[0, 0].set_title('Stock A')

    x2 = np.linspace(-10, 26, 1000)
    axes[0, 1].hist(x2, density=True)
    axes[0, 1].set_xlim(xmin, xmax)
    axes[0, 1].set_ylim(0, 0.1)
    axes[0, 1].set_title('Stock B')

    x3 = np.random.normal(8, 4, 1000)
    axes[1, 0].hist(x3, density=True)
    axes[1, 0].set_xlim(xmin, xmax)
    axes[1, 0].set_ylim(0, 0.2)
    axes[1, 0].set_title('Stock C')

    x4 = np.linspace(6, 8, 1000)
    axes[1, 1].hist(x4, density=True)
    axes[1, 1].set_xlim(xmin, xmax)
    axes[1, 1].set_ylim(0, 0.6)
    axes[1, 1].set_title('Stock D')

    plt.tight_layout()
    plt.show()

# custom formatter for representing millions on histogram tick marks
def tick_formatter(x, pos):
    return f'{x / 1e6:.2f}M'

# update globals using user input
def get_inputs():
    # get entries: starting money, number of simulations, years, desired profit, desired loss
    global start_money; global num_sims; global num_years; global profit_threshold; global loss_threshold
    start_money = float(entry_start_money.get())
    num_sims = int(entry_num_sims.get())
    num_years = int(entry_num_years.get())
    profit_threshold = start_money * (100 + int(entry_profit.get())) / 100
    loss_threshold = start_money * (100 - int(entry_loss.get())) / 100
    
def get_stock_allocs():
    return {
        'A': scale_A.get() / 100,
        'B': scale_B.get() / 100,
        'C': scale_C.get() / 100,
        'D': scale_D.get() / 100
    }

def run_sim_default():
    stock_allocs = get_stock_allocs()
    global num_sims
    run_sim(stock_allocs, num_sims)

# function for main simulation logic
def run_sim(stock_allocs, num_sims):
    
    # get user inputs
    get_inputs()
    
    # perform monte carlo simulation
    portfolio_values = [] # array to contain total money returned from each simulation
    prob_profit = 0 # proability of profit
    prob_loss = 0 # probability of loss

    # loop through simulations
    for _ in range(num_sims):
        portfolio_value = start_money # the inital investment for a year / the value of the entire portfolio after all years
        # loop through years
        for _ in range(num_years):
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
            # determine the portfolio's value after a year / the initial investment for next year
            portfolio_value += yearly_yield
        # determine whether final portfolio value was a profit or loss based on user thresholds
        if portfolio_value >= profit_threshold:
            prob_profit += 1/num_sims
        if portfolio_value <= loss_threshold:
            prob_loss += 1/num_sims
        # store value of entire portfolio after each simulation
        portfolio_values.append(portfolio_value)
    
    # analyze results
    mean_profit = np.mean(portfolio_values)
    mean_percent_profit = (mean_profit - start_money) / start_money * 100
    std_dev = np.std(portfolio_values)
    
    # update result fields
    avg_profit.config(text=f'Average Profit: {mean_percent_profit:.2f}%')
    end_money.config(text=f'Average Ending Money: ${mean_profit:.2f}')
    standard_dev.config(text=f'Standard Deviation: ${std_dev:.2f}')

    profit_percent = int(entry_profit.get())
    if(profit_percent != 0):
        profit.config(text=f'Probability of at Least {profit_percent}% Profit: {prob_profit * 100:.2f}%')
    else:
        profit.config(text=f'Probability of Any Profit: {prob_profit * 100:.2f}%')
    
    loss_percent = int(entry_loss.get())
    if(loss_percent != 0):
        loss.config(text=f'Probability of at Least {loss_percent}% Loss: {prob_loss * 100:.2f}%')
    else:
        loss.config(text=f'Probability of Any Loss: {prob_loss * 100:.2f}%')

    # generate histogram
    if cbtn_graph_bool.get() == True:
        plt.figure(figsize=(8, 6), num='Simulation Results', clear=True)
        plt.grid(True)
        plt.hist(portfolio_values, bins=70, density=True, alpha=0.5, color='b')
        plt.xticks(rotation=-30, ha='left')
        plt.gca().xaxis.set_major_formatter(tkr.FuncFormatter(tick_formatter))
        plt.xlabel('Ending Money ($)')
        plt.ylabel('Probability (%)')
        plt.title('Monte Carlo Simulation Results')
        plt.axvline(mean_profit, color='r', linestyle='dashed', linewidth=1, label=f'Mean: ${mean_profit:.2f}')
        plt.axvline(mean_profit - std_dev, color='g', linestyle='dashed', linewidth=1, label=f'Std Dev: ±${std_dev:.2f}')
        plt.axvline(mean_profit + std_dev, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.show()

# function for calculating optimal portfolio
def find_opt():
    
    # get user inputs
    get_inputs()
    
    # first obtain mean (expected value) and standard deviation for each stock
    stocks_mean = []
    stocks_std = []
    # obtain expected returns and standard deviation
    for(_, param) in stock_params.items():
        if(param['dist'] == 'unif'):
            mean = (param['a'] + param['b']) / 2
            var = (pow(param['b']-param['a'], 2))/12
            std = pow(var, 0.5)
            stocks_mean.append(mean)
            stocks_std.append(std)
        elif(param['dist']=='norm'):
            stocks_mean.append(param['mean'])
            stocks_std.append(param['stdev'])
            
    print(stocks_mean)
    print(stocks_std)

    n = 100000

    # arrays holding the simulation results
    all_weights = np.zeros((n, 4))
    mean_arr = np.zeros(n)
    stdev_arr = np.zeros(n)
    sharpe_ratios = np.zeros(n)

    ## repeat for the number of simulations given
    for i in range(n):
        # create random weights for each stock that add up to 1
        weights = np.array(np.random.random(4))
        weights = weights / np.sum(weights)
        # save weights
        all_weights[i, :] = weights

        # gather the expected return and stdev with these new weights
        # store the mean and stdev information
        mean_arr[i] = np.dot(stocks_mean, weights)
        stdev_arr[i] = np.dot(stocks_std, weights)

        # store the calculated sharpe ratio
        sharpe_ratios[i] = mean_arr[i]/stdev_arr[i]

    max = sharpe_ratios.argmax()
    print("max sharpe ratio: ", sharpe_ratios[max], " at index ", max)
    print(all_weights[max])
    
    optimal_stocks = ("optimal stocks: % \n"
              "stock A: {:.4f}\n"
              "stock B: {:.4f}\n"
              "stock C: {:.4f}\n"
              "stock D: {:.4f}").format(all_weights[max][0], all_weights[max][1], all_weights[max][2], all_weights[max][3])
        
    if cbtn_opt_bool.get() == True:
        
        # plot the stuff
        plt.figure(figsize=(8,6), num="Sharpe Ratio Graph for Optimal Stock Allocation", clear=True)
        plt.scatter(x=stdev_arr, y=mean_arr, c=sharpe_ratios, cmap='PuRd')
        plt.xlabel('standard deviation (volatility)')
        plt.ylabel('return rate')
        plt.colorbar(label='Sharpe Ratio')
        t = plt.text(0.05, 0.09, optimal_stocks, fontsize=12,style='italic')
        t.set_bbox(dict(facecolor='purple', alpha=0.5, linewidth=0))
        plt.show()

def find_opt_2():
    print(f'\nPortfolio with\n\tat least {profit_prob * 100}% chance of {profit_threshold * 100}% or more profit,\n\tat least {loss_prob * 100}% chance of {loss_threshold * 100}% or more loss,\n\tand highest Sharpe ratio after {num_years} years:')

    # TESTING
    highest_sharpe = 0
    num_sims = 100
    opt_i = -1
    i = 0
    portfolio = ()

    for percent_A in range(0, 101, 5):
        for percent_B in range(0, 101 - percent_A, 5):
            for percent_C in range(0, 101 - percent_A - percent_B, 5):
                for percent_D in range(0, 101 - percent_A - percent_B - percent_C, 5):
                    if (percent_A + percent_B + percent_C + percent_D == 100):
                        (mean, stdev, p_prob, l_prob) = run_sim({'A' : percent_A, 'B' : percent_B, 'C' : percent_C, 'D' : percent_D}, num_sims)
                        sharpe = mean / stdev

                        if (p_prob >= profit_prob and l_prob <= loss_prob and sharpe >= highest_sharpe):
                            opt_i = i
                            highest_sharpe = sharpe
                            portfolio = (percent_A, percent_B, percent_C, percent_D, sharpe)

                        i += 1

    if(portfolio == ()): portfolio = 'Does Not Exist'
    print(f'\t\t{portfolio}\n')

# main window
canvas = Tk()
canvas.title('Monte Carlo Simulation')

# starting money entry
label_start_money = Label(canvas, text='Starting Money ($):')
label_start_money.grid(row=0, column=0, padx=10, pady=5, sticky='w')
text_SM = StringVar()
text_SM.set(INITIAL_INVESTMENT)
entry_start_money = Entry(canvas, textvariable=text_SM)
entry_start_money.grid(row=0, column=1, padx=10, pady=5, sticky='e')

# number of simulations entry
label_num_sims = Label(canvas, text='Number of Simulations:')
label_num_sims.grid(row=1, column=0, padx=10, pady=5, sticky='w')
text_NS = StringVar()
text_NS.set(NUM_SIMULATIONS)
entry_num_sims = Entry(canvas, textvariable=text_NS)
entry_num_sims.grid(row=1, column=1, padx=10, pady=5, sticky='e')

# number of years entry
label_num_years = Label(canvas, text='Time Period (Years):')
label_num_years.grid(row=2, column=0, padx=10, pady=5, sticky='w')
text_NY = StringVar()
text_NY.set('1')
entry_num_years = Entry(canvas, textvariable=text_NY)
entry_num_years.grid(row=2, column=1, padx=10, pady=5, sticky='e')

# desired profit entry
label_profit = Label(canvas, text='Desired Profit Amount (%):')
label_profit.grid(row=3, column=0, padx=10, pady=5, sticky='w')
text_DP = StringVar()
text_DP.set('0')
entry_profit = Entry(canvas, textvariable=text_DP)
entry_profit.grid(row=3, column=1, padx=10, pady=5, sticky='e')

# desired loss entry
label_loss = Label(canvas, text='Desired Loss Amount (%):')
label_loss.grid(row=4, column=0, padx=10, pady=5, sticky='w')
text_DL = StringVar()
text_DL.set('0')
entry_loss = Entry(canvas, textvariable=text_DL)
entry_loss.grid(row=4, column=1, padx=10, pady=5, sticky='e')

# desired profit probability entry
label_p_prob = Label(canvas, text='Desired Profit Probability (%):')
label_p_prob.grid(row=5, column=0, padx=10, pady=5, sticky='w')
text_p_prob = StringVar()
text_p_prob.set('0')
entry_p_prob = Entry(canvas, textvariable=text_p_prob)
entry_p_prob.grid(row=5, column=1, padx=10, pady=5, sticky='e')

# desired loss probability entry
label_l_prob = Label(canvas, text='Desired Loss Probability (%):')
label_l_prob.grid(row=6, column=0, padx=10, pady=5, sticky='w')
text_l_prob = StringVar()
text_l_prob.set('0')
entry_l_prob = Entry(canvas, textvariable=text_l_prob)
entry_l_prob.grid(row=6, column=1, padx=10, pady=5, sticky='e')

# scales for allocating percentages to each stock
label_A = Label(canvas, text='Allocation for Stock A: 0%')
label_A.grid(row=0, column=4, columnspan=2, padx=10)
scale_A = Scale(canvas, from_=0, to=100, orient='horizontal', length=300, command=update_label_A, resolution=1, tickinterval=25)
scale_A.grid(row=1, column=4, rowspan=2, columnspan=2, padx=10)

label_B = Label(canvas, text='Allocation for Stock B: 0%')
label_B.grid(row=3, column=4, columnspan=2, padx=10)
scale_B = Scale(canvas, from_=0, to=100, orient='horizontal', length=300, command=update_label_B, resolution=1, tickinterval=25)
scale_B.grid(row=4, column=4, rowspan=2, columnspan=2, padx=10)

label_C = Label(canvas, text='Allocation for Stock C: 0%')
label_C.grid(row=6, column=4, columnspan=2, padx=10)
scale_C = Scale(canvas, from_=0, to=100, orient='horizontal', length=300, command=update_label_C, resolution=1, tickinterval=25)
scale_C.grid(row=7, column=4, rowspan=2, columnspan=2, padx=10)

label_D = Label(canvas, text='Allocation for Stock D: 0%')
label_D.grid(row=9, column=4, columnspan=2, padx=10)
scale_D = Scale(canvas, from_=0, to=100, orient='horizontal', length=300, command=update_label_D, resolution=1, tickinterval=25)
scale_D.grid(row=10, column=4, rowspan=2, columnspan=2, padx=10)

# button for displaying the distribution graphs of all stocks
btn_dispay_stocks = Button(canvas, text='Show Stock Distributions', command=display_stocks)
btn_dispay_stocks.grid(row=18, column=4, columnspan=2)

# Create fields for results
end_money = Label(canvas, text='Average Ending Money:')
end_money.grid(row=7, column=0, columnspan=2, padx=10, pady=5)

avg_profit = Label(canvas, text='Average Profit:')
avg_profit.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

standard_dev = Label(canvas, text='Standard Deviation:')
standard_dev.grid(row=9, column=0, columnspan=2, padx=10, pady=5)

profit = Label(canvas, text='Probability of Min Desired Profit:')
profit.grid(row=10, column=0, columnspan=2, padx=10, pady=5)

loss = Label(canvas, text='Probability of Min Desired Loss:')
loss.grid(row=11, column=0, columnspan=2, padx=10, pady=5)

# button to run sim
btn_run = Button(canvas, text='Run Simulation', command=run_sim_default)
btn_run.grid(row=18, column=0, pady=10)

# checkbutton for displaying simulation results graph
cbtn_graph_bool = IntVar()
cbtn_graph = Checkbutton(canvas, text='Show Simulation Graph', variable=cbtn_graph_bool)
cbtn_graph.grid(row=20, column=0, pady=10)

# button to run optimal portfolio calculation
btn_opt = Button(canvas, text='Find Optimal Portfolio', command=find_opt)
btn_opt.grid(row=18, column=1, pady=10)

# check button for displaying optimal portfolio graph
cbtn_opt_bool = IntVar()
cbtn_opt = Checkbutton(canvas, text='Show Optimal Portfolio Graph', variable=cbtn_opt_bool)
cbtn_opt.grid(row=20, column=1, pady=10)

canvas.mainloop()
