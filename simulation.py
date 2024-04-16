import matplotlib as plt
import numpy
from tkinter import *


# initial macros
INITIAL_INVESTMENT = 1000000
NUM_SIMULATIONS = 1000

# when user clicks on "Run Sim"
def run_sim():
    print("Running simulation")

# when user clicks on stock A
def stock_A_dist():
    print("Displaying stock A distribution graph")

# when user clicks on stock B
def stock_B_dist():
    print("Displaying stock B distribution graph")

# when user clicks on stock C
def stock_C_dist():
    print("Displaying stock C distribution graph")

# when user clicks on stock D
def stock_D_dist():
    print("Displaying stock D distribution graph")

# set up

# screen options
canvas = Tk()
canvas.title('Monte Carlo Simulation')
canvas.geometry('600x400')

# Monte Carlo Simulation label
intro_label = Label(canvas, text = "Monte Carlo Simulation")

# run sim button
run_sim_button = Button(canvas, text = "Run Sim", command=run_sim)

# run stock distribution graphs
run_dist_A_button = Button(canvas, text = "Show stock A distribution", command=stock_A_dist)
run_dist_B_button = Button(canvas, text = "Show stock B distribution", command=stock_B_dist)
run_dist_C_button = Button(canvas, text = "Show stock C distribution", command=stock_C_dist)
run_dist_D_button = Button(canvas, text = "Show stock D distribution", command=stock_D_dist)


# pack the components
intro_label.pack(expand=True, fill='both')
run_sim_button.pack(expand=True, fill='both')
run_dist_A_button.pack(expand=True, fill='both')
run_dist_B_button.pack(expand=True, fill='both')
run_dist_C_button.pack(expand=True, fill='both')
run_dist_D_button.pack(expand=True, fill='both')

# run screen
canvas.mainloop()
