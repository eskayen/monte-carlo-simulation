from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd

# initial macros
INITIAL_INVESTMENT = 1000000
NUM_SIMULATIONS = 1000

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
    return f'{x / 1e6:.1f}M'

# function for main simulation logic
def run_sim():
    # get entries: starting money, number of simulations, years, desired profit, desired loss
    start_money = float(entry_start_money.get())
    num_sims = int(entry_num_sims.get())
    years = int(entry_num_years.get())
    profit_val = start_money * (100 + int(entry_profit.get())) / 100
    loss_val = start_money * (100 - int(entry_loss.get())) / 100

    # determine what percentage of starting money went to which stock
    stock_allocs = {
        'A': scale_A.get() / 100,
        'B': scale_B.get() / 100,
        'C': scale_C.get() / 100,
        'D': scale_D.get() / 100
    }

    # set up parameters of the four stocks
    stock_params = {
        'A': {'dist': 'unif', 'a': -0.4, 'b': 0.6},
        'B': {'dist': 'unif', 'a': -0.1, 'b': 0.26},
        'C': {'dist': 'norm', 'mean': 0.08, 'stdev': 0.04},
        'D': {'dist': 'unif', 'a': 0.06, 'b': 0.08}
    }

    # perform monte carlo simulation
    portfolio_returns = [] # array to contain total money returned from each simulation
    for _ in range(num_sims):
        portfolio_return = start_money # begins at inital money and accumulates amount earned by each stock
        # iterate through the four stocks
        for (stock, params) in stock_params.items():
            percent_return = 0 # will represent the percentage of the starting money a stock will yield
            # for each year, randomly generate a percentage according to the stock's distribution and add to overall percent return
            for _ in range(years):
                if params['dist'] == 'unif':
                    percent_return += np.random.uniform(params['a'], params['b'])
                elif params['dist'] == 'norm':
                    percent_return += np.random.normal(params['mean'], params['stdev'])
            # once the stock had been evaluated throughout the time span, determine how much money was returned and accumulate
            portfolio_return += stock_allocs[stock] * (portfolio_return * percent_return)
        # store the money gained from the entire portfolio
        portfolio_returns.append(portfolio_return)

    # analyze results
    mean_profit = np.mean(portfolio_returns)
    mean_percent_profit = (mean_profit - start_money) / start_money * 100
    std_dev = np.std(portfolio_returns)

    # update result fields
    avg_profit.config(text=f'Average Profit: {mean_percent_profit:.2f}%')
    end_money.config(text=f'Average Ending Money: ${mean_profit:.2f}')
    standard_dev.config(text=f'Standard Deviation: ${std_dev:.2f}')

    # calculate probability of profit and loss
    prob_profit = 0
    prob_loss = 0
    for val in portfolio_returns:
        if val >= profit_val:
            prob_profit += 1/num_sims
        if val <= loss_val:
            prob_loss += 1/num_sims
    profit.config(text=f'Probability of at Least {int(entry_profit.get())}% Profit: {prob_profit * 100:.2f}%')
    loss.config(text=f'Probability of at Least {int(entry_loss.get())}% Loss: {prob_loss * 100:.2f}%')


    # generate histogram
    if cbtn_graph_bool.get() == True:
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.hist(portfolio_returns, bins=50, density=True, alpha=0.5, color='b')
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

    # get optimal portfolio
    if cbtn_opt_bool.get() == True:

        # first obtain mean (expected value) and standard deviation for each stock
        stocks_mean = []
        stocks_std = []

        for(stock, param) in stock_params.items():
            if(param['dist'] == 'unif'):
                mean = (param['a'] + param['b']) / 2
                var = ((param['b'] - param['a'])**2)/12
                std = var ** (0.5)
                stocks_mean.append(mean)
                stocks_std.append(std)
            elif(param['dist']=='norm'):
                stocks_mean.append(param['mean'])
                stocks_std.append(param['stdev'])

        # arrays holding the simulation results
        all_weights = np.zeros((num_sims, 4))
        mean_arr = np.zeros(num_sims)
        stdev_arr = np.zeros(num_sims)
        sharpe_ratios = np.zeros(num_sims)

        ## repeat for the number of simulations given
        for i in range(num_sims):
            # create random weights for each stock that add up to 1
            weights = np.array(np.random.random(4))
            weights = weights / np.sum(weights)
            # save weights
            all_weights[i, :] = weights

            # gather the expected return and stdev with these new weights
            # store the mean and stdev information
            mean_arr[i] = np.sum(stocks_mean * weights)
            stdev_arr[i] = np.sqrt(np.sum(stocks_std * weights))

            # store the calculated sharpe ratio
            sharpe_ratios[i] = mean_arr[i]/stdev_arr[i]

        print("largest sharpe ratio: ")
        print(sharpe_ratios.argmax())

        # plot the stuff
        plt.figure(figsize=(8,6))
        plt.scatter(stdev_arr, mean_arr, c=sharpe_ratios, cmap='plasma')
        plt.xlabel('stdev')
        plt.ylabel('mean')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()


# main window
canvas = Tk()
canvas.title('Monte Carlo Simulation')

# starting money entry
label_start_money = Label(canvas, text='Starting Money ($):')
label_start_money.grid(row=0, column=0, padx=10, pady=5, sticky='w')
default_text_SM = StringVar()
default_text_SM.set(INITIAL_INVESTMENT)
entry_start_money = Entry(canvas, textvariable=default_text_SM)
entry_start_money.grid(row=0, column=1, padx=10, pady=5)

# number of simulations entry
label_num_sims = Label(canvas, text='Number of Simulations:')
label_num_sims.grid(row=1, column=0, padx=10, pady=5, sticky='w')
default_text_NS = StringVar()
default_text_NS.set(NUM_SIMULATIONS)
entry_num_sims = Entry(canvas, textvariable=default_text_NS)
entry_num_sims.grid(row=1, column=1, padx=10, pady=5)

# number of years entry
label_num_years = Label(canvas, text='Time Period (Years):')
label_num_years.grid(row=2, column=0, padx=10, pady=5, sticky='w')
default_text_NY = StringVar()
default_text_NY.set('1')
entry_num_years = Entry(canvas, textvariable=default_text_NY)
entry_num_years.grid(row=2, column=1, padx=10, pady=5)

# desired profit entry
label_profit = Label(canvas, text='Desired Profit (%):')
label_profit.grid(row=3, column=0, padx=10, pady=5, sticky='w')
default_text_DP = StringVar()
default_text_DP.set('0')
entry_profit = Entry(canvas, textvariable=default_text_DP)
entry_profit.grid(row=3, column=1, padx=10, pady=5)

# desired loss entry
label_loss = Label(canvas, text='Desired Loss (%):')
label_loss.grid(row=4, column=0, padx=10, pady=5, sticky='w')
default_text_DL = StringVar()
default_text_DL.set('0')
entry_loss = Entry(canvas, textvariable=default_text_DL)
entry_loss.grid(row=4, column=1, padx=10, pady=5)

# button for displaying the distribution graphs of all stocks
btn_dispay_stocks = Button(canvas, text='Show Stock Distributions', command=display_stocks)
btn_dispay_stocks.grid(columnspan=2, pady=(15, 0))

# scales for allocating percentages to each stock
label_A = Label(canvas, text='Allocation for Stock A: 0%')
label_A.grid(column=0, columnspan=2, padx=10, pady=(15, 0))
scale_A = Scale(canvas, from_=0, to=100, orient='horizontal', length=200, command=update_label_A, resolution=5, tickinterval=25)
scale_A.grid(column=0, columnspan=2, padx=10)

label_B = Label(canvas, text='Allocation for Stock B: 0%')
label_B.grid(column=0, columnspan=2, padx=10)
scale_B = Scale(canvas, from_=0, to=100, orient='horizontal', length=200, command=update_label_B, resolution=5, tickinterval=25)
scale_B.grid(column=0, columnspan=2, padx=10)

label_C = Label(canvas, text='Allocation for Stock C: 0%')
label_C.grid(column=0, columnspan=2, padx=10)
scale_C = Scale(canvas, from_=0, to=100, orient='horizontal', length=200, command=update_label_C, resolution=5, tickinterval=25)
scale_C.grid(column=0, columnspan=2, padx=10)

label_D = Label(canvas, text='Allocation for Stock D: 0%')
label_D.grid(column=0, columnspan=2, padx=10)
scale_D = Scale(canvas, from_=0, to=100, orient='horizontal', length=200, command=update_label_D, resolution=5, tickinterval=25)
scale_D.grid(column=0, columnspan=2, padx=10, pady=(0, 10))

# Create fields for results
end_money = Label(canvas, text='Average Ending Money:')
end_money.grid(column=0, columnspan=2, padx=10, pady=5)

avg_profit = Label(canvas, text='Average Profit:')
avg_profit.grid(column=0, columnspan=2, padx=10, pady=5)

standard_dev = Label(canvas, text='Standard Deviation:')
standard_dev.grid(column=0, columnspan=2, padx=10, pady=5)

profit = Label(canvas, text='Probability of Min Desired Profit:')
profit.grid(column=0, columnspan=2, padx=10, pady=5)

loss = Label(canvas, text='Probability of Min Desired Loss:')
loss.grid(column=0, columnspan=2, padx=10, pady=5)

# button to run sim
btn_run = Button(canvas, text='Run Simulation', command=run_sim)
btn_run.grid(row=19, column=0, columnspan=2, pady=10)

# checkbutton for displaying graph
cbtn_graph_bool = IntVar()
cbtn_graph = Checkbutton(canvas, text='Show Simulation Graph', variable=cbtn_graph_bool)
cbtn_graph.grid(row=21, column=0, columnspan=1, pady=10)

## Optimal portfolio determination

# check button for getting optimal portfolio
cbtn_opt_bool = IntVar()
cbtn_opt = Checkbutton(canvas, text='Show Optimal Portfolio Graph', variable=cbtn_opt_bool)
cbtn_opt.grid(row=21, column=1,columnspan=2, pady=10)

canvas.mainloop()
