"""
This file will contains a series of functions to plot the graph and save the plot

plot_graph() function has 5 parameters to generate a graph
    1. sequential_data: The dependent data want to plot
    2. name: The name of the file
    3. title: The title of the graph
    4. x_label: The x-axis's legend
    5. y_label: The y-axis's legend

"""

import matplotlib.pyplot as plt
import numpy as np

def plot_graph(sequential_data, name, title, x_label, y_label):
    
    # Plot the graph where sequential_data is 1d
    fig, ax = plt.subplots()
    ax.plot(sequential_data, range(len(sequential_data)))
    ax.set_title(title)
    ax.set(xlabel=x_label, ylabel=y_label)
    fig.savefig(name, dpi=150)
    


# Only for testing
if __name__ == "__main__":
    print("hello world")
    plot_graph([1,2,3,4,5], "test", "This is the test", "iteration", "accuracy")


    
