import numpy as np
import random
import time
import math
import copy
import matplotlib.pyplot as plt
import csv
import os
import json
import datetime
import matplotlib.ticker as ticker
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
import itertools
from scipy.integrate import trapz  # Using trapezoidal rule for discrete integration

# additional files to be present in working directory
import helper_function as hf
from Solve_YS_New_test import solve_ys
from Solve_YS_Original import solveYS_original
from Full_Model_fix_y import Full_Model_x_y, Full_Model_BinP, model_Px, MinCostFlowNetwork_ORTools_News, create_reference_matrix_and_count_switches
#from sensitivity_plots import generate_hourly_prices


# Get the directory of the current script
script_directory = os.path.abspath(os.path.dirname(__file__))

# Change the current working directory to the script directory
os.chdir(script_directory)


# Get current date and time
current_datetime = datetime.datetime.now()
folder_name = current_datetime.strftime("%Y%m%d_%H%M%S")
output_folder_path = os.path.join("output", folder_name)

# Create the directory
os.makedirs(output_folder_path, exist_ok=True)



# Define the variations

def generate_variations(*args):
    """
    Generate all combinations of options provided for each category
    and assign unique numbers to each combination.

    Args:
        *args: List of lists, where each list contains options for a category.
    
    Returns:
        dict: Dictionary with unique numbers as keys and combinations as values.
    """
    # Generate all combinations
    combinations = list(itertools.product(*args))
    
    # Assign unique numbers to each combination
    return {i + 1: combination for i, combination in enumerate(combinations)}


def get_regression_parameter(mu, config_data):
    
    slopes={}; intercepts={}

    for category in range(len(config_data.B_ell)):
        # Filter values by category
        y_values = [mu[j, k, t] for (j, k, t) in mu.keys() if j in config_data.B_ell[category]]
        x_values = [config_data.cost[t-1] for (j, k, t) in mu.keys() if j in config_data.B_ell[category]]
        
        # Perform linear regression for this category
        slope, intercept, r_value, _, _ = stats.linregress(x_values, y_values)
        slopes[category] = slope; intercepts[category] = intercept 
    return slopes, intercepts


def new_plot_mu_vs_cost_with_regression(mu, cost, iteration, h_x_plus_h_ys, config_data):
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'green', 'red']  # Colors for each category
    labels = ['First time window', 'Second time window', 'Third time window']  # Labels for each category
    

    for category in range(len(config_data.B_ell)):
        # Filter values by category
        y_values = [mu[j, k, t] for (j, k, t) in mu.keys() if j in config_data.B_ell[category]]
        x_values = [cost[t-1] for (j, k, t) in mu.keys() if j in config_data.B_ell[category]]
        
        # Perform linear regression for this category
        slope, intercept, r_value, _, _ = stats.linregress(x_values, y_values)

        # Plotting
        plt.scatter(x_values, y_values, label=f'$\mu$ for {labels[category]}', color=colors[category])
        line_x = np.linspace(min(x_values), max(x_values), 100)
        line_y = slope * line_x + intercept
        
        plt.plot(line_x, line_y, color=colors[category], 
                 label=f'$\omega^{category+1}_{1}$={slope:.2f}, $\omega^{category+1}_{2}$={intercept:.2f}, $R^2$={r_value**2:.2f}')

    plt.ylim(-15, 20)
    #plt.title(f'Relationship between cost and mu values at iteration {iteration}')
    plt.annotate(f'h(μ) = {h_x_plus_h_ys:.2f}', 
                     xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=15, bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'),
                     verticalalignment='top')
    plt.xlabel('$c$',fontsize=18)
    plt.ylabel('$\mu_{jkt}$',fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig(os.path.join('plots', f'plot_iteration_{iteration}.png'))
    
    plt.savefig(os.path.join('plots', f'plot_iteration_{iteration}.eps'))

    
    plt.close()


# Function to plot mu vs cost and regression line
def plot_mu_vs_cost_with_regression(mu, cost, iteration, h_x_plus_h_ys, config_data):
    y_values = [mu[j, k, t] for (j, k, t) in mu.keys()]
    x_values = [cost[t-1] for (j, k, t) in mu.keys()]
    
    # Perform linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x_values, y_values)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, label='mu values')
    line_x = np.linspace(min(x_values), max(x_values), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label=f'Regression line (R^2={r_value**2:.2f})')
    
    plt.ylim(-15, 20)
    # Title or separate annotation for slope and intercept
    plt.title(f'Relationship between cost and mu values at iteration {iteration}\n'
              f'Slope: {slope:.2f}, Intercept: {intercept:.2f}, R^2={r_value**2:.2f}')
    
    # Annotation for slope, intercept, and h_x + h_ys
    plt.annotate(f'h(μ) = h_x + h_ys: {h_x_plus_h_ys:.2f}', 
                     xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='yellow'),
                     verticalalignment='top')

    plt.xlabel('Cost')
    plt.ylabel('$\mu$',fontsize=16)
    plt.legend()
    plt.grid(True)
    
    # Save plot to 'plots' directory
    plt.savefig(os.path.join('plots', f'plot_iteration_{iteration}.png'))
    plt.savefig(os.path.join('plots', f'plot_iteration_{iteration}.eps'))
    plt.close()




def predict_mu(cost, slope=-.29, intercept=7.57):
    """
    Predicts the mu value based on the given cost using the linear regression model.

    Parameters:
    - cost (float): The cost value for which to predict mu.
    - slope (float): The slope of the regression line.
    - intercept (float): The intercept of the regression line.

    Returns:
    float: The predicted mu value, or zero if the prediction is negative.
    """
    predicted_mu = slope * cost + intercept
    return predicted_mu
    #return max(predicted_mu, 0)  # Return 0 if the prediction is negative

def generate_two_tuples(max_value):
    """
    Generate all possible two-tuples (i, j) such that i < j and both are in the range [0, max_value-1].

    :param max_value: The upper limit for the range of numbers (exclusive).
    :return: List of two-tuples.
    """
    return [(i, j) for i in range(max_value) for j in range(i + 1, max_value)]


def plot_combined_time_data(data1, data2, data3, title, xlabel, ylabel, legend1, legend2, legend3, filename, total_time_1, total_time_2, total_time_3):
    plt.figure()
    iterations = list(data2.keys())

    plt.plot(iterations, list(data1.values()), marker='o', markersize=4,color='blue', label=legend1)
    plt.plot(iterations, list(data2.values()), marker='x', color='red',markersize=4, label=legend2)
    plt.plot(iterations, list(data3.values()), marker='d', color='green',markersize=4, label=legend3)

    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Separate LaTeX and Python parts
    latex_part_1 = r"Total time $P^x$: "
    latex_part_2 = r", Total time $\tilde{P}^{ys}$: "
    latex_part_3 = "Total time: "
    
    # # Combine them with your Python variables
    updated_title = latex_part_1 + str(round(total_time_1, 2)) + \
                    latex_part_2 + str(round(total_time_2, 2)) + \
                    latex_part_3 + str(round(total_time_3, 2))

    plt.title(updated_title, fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(output_folder_path, filename)
    plt.savefig(filepath)
    plt.show()


def plot_combined_h_data_New(h_mu, h_x_mu, h_ys_mu, h_upp, title, xlabel, ylabel, filename, max_h_mu_value):
    plt.figure()
    initial_high = h_upp[1]  # Assuming the first value is the initial high value
    
    # Find the iteration where h_upp first changes from the initial high value
    change_iteration = 1#next((i for i, v in h_upp.items() if v != initial_high), len(h_upp))
    
    # Filter the iterations and values for plotting starting from the change_iteration
    filtered_iterations = [i for i in h_mu.keys() if i >= change_iteration]
    filtered_h_mu = [h_mu[i] for i in filtered_iterations]
    filtered_h_upp = [h_upp[i] for i in filtered_iterations if i in h_upp]
    
    # Update the plot to start from the iteration after the initial high value has changed
    plt.plot(filtered_iterations, filtered_h_mu, marker='o', color='green', markersize=4, label=r'$h(\mu)$')
    plt.plot(filtered_iterations, filtered_h_upp, marker='^', color='purple', markersize=4, label=r'$\bar{h}$')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Update the title and labels
    max_h_mu_value = round(max(filtered_h_mu), 2) if filtered_h_mu else 'N/A'
    min_h_upp_value = round(min(filtered_h_upp), 2) if filtered_h_upp else 'N/A'
    updated_title = f"h_low_best: {max_h_mu_value}, h_up_best: {min_h_upp_value}"
    plt.title(updated_title)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Save and show the plot
    filepath = os.path.join(output_folder_path, filename)
    plt.savefig(filepath)
    plt.show()

def plot_combined_h_data(h_mu, h_x_mu, h_ys_mu, h_upp, title, xlabel, ylabel, filename, max_h_mu_value):
    iterations = list(h_mu.keys())
    plt.figure()
    max_h_mu_value = round(max(h_mu.values()), 2)
    min_h_upp_value = round(min(h_upp.values()), 2)
    updated_title = f"h_low_best: {max_h_mu_value}, h_up_best: {min_h_upp_value}"
    # Update the title to include max h_mu value
    # Separate LaTeX and Python parts

    plt.plot(iterations, [h_mu[i] for i in iterations], marker='o', color='green', markersize=4, label=r'$h(\mu)$')
    #plt.plot(iterations, [h_x_mu[i] for i in iterations], marker='o', color='blue', label=r'$h_{x}(\mu)$')
    #plt.plot(iterations, [h_ys_mu[i] for i in iterations], marker='x', color='red', label=r'$h_{ys}(\mu)$')

    if h_upp:
       plt.plot(iterations, [h_upp[i] for i in iterations], marker='^', color='purple', markersize=4, label=r'$\bar{h}$')
       #pass
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.title(updated_title)
    plt.yscale('log')  # For setting the y-axis to logarithmic scale

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    filepath = os.path.join(output_folder_path, filename)
    plt.savefig(filepath)   
    plt.show()


def _merge_dicts_to_array(*dicts):
    # Check that all dictionaries have the same set of keys
    keys_list = [set(d.keys()) for d in dicts]
    if not all(keys == keys_list[0] for keys in keys_list):
        raise ValueError("All dictionaries must have the same keys")
    
    # Get sorted keys from the first dictionary
    sorted_keys = sorted(keys_list[0])
    
    # Merge dictionaries into a numpy array
    merged_array = np.array([[key] + [d[key] for d in dicts] for key in sorted_keys])
    return merged_array


def save_array_to_csv_with_params_new(data, base_filename, config_data, column_names, output_folder_path, vari):
    filename = f"{base_filename}_b{config_data.batteries}_p{config_data.n_ports}_g{config_data.gamma}_var_{vari}.csv"
    filepath = os.path.join(output_folder_path, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header, which includes the keys and the specified column names
        writer.writerow( column_names)  # ['Key'] +'Key' for the dictionary keys and then column names for each dictionary
        
        # Write the data rows from the numpy array
        for row in data:
            writer.writerow(row)


def save_dict_to_csv_with_params(data, base_filename, config_data, vari):
    filename = f"{base_filename}_b{config_data.batteries}_p{config_data.n_ports}_g{config_data.gamma}_var_{vari}.csv"
    filepath = os.path.join(output_folder_path, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Value'])  # Optional: Write header
        for key, value in data.items():
            writer.writerow([key, value])

class ConfigData:
    def __init__(self, batteries, n_ports, gamma, proportion, alpha, demand_window):
        """
       Initialize the configuration data for the system.
       :param batteries: Number of batteries in the system.
       :param n_ports: Number of charging ports available in the system.
       :param gamma: Maximum # of allowed switching
       :param proportion: A 1-D array of proportion of deliveries to be made in each time window and the last.
       :param alpha: Lower bound of the state-of-the-charge of remaining batteries.
       :param demand_window: Time period when the deliveries are to be made.
       """        
        self.batteries = batteries
        self.n_ports = n_ports
        self.gamma = gamma
        self.proportion = proportion
        self.ell = len(proportion)
        self.alpha = alpha
        self.demand_window = demand_window
        self.cost=hf.generate_electricity_prices(24, 60)
        p = hf.create_mixed_distribution_vector(self.batteries)
        self.p_dict={i+1: p[i] for i in range(len(p))} # time to charge the batteries
        self.B_ell = hf.distribute_B_based_on_p(range(1,self.batteries+1), self.p_dict, self.ell, self.proportion)
        self.validate()

    def validate(self):
        assert len(self.demand_window) == self.ell - 1, "Demand window does not match ell."
        # Add more validations as needed



class Heuristic:
    
    def __init__(self, config_data):
        self.config_data=config_data
        
    def generate_dictionaries(self):
        x_val = {}
        y = {}
        T = len(self.config_data.cost)
        n_ports = self.config_data.n_ports
        batteries = self.config_data.batteries
    
        # Adjusted for Python's 0-based indexing
        for j in range(1, batteries + 1):
            for k in range(1,n_ports+1):
                for t in range(1, T + 1):  # Time from 1 to T
                    # Adjust t-1 when accessing machine_states
                    x_val[j, k, t] = 1 if self.machine_states[k-1][t - 1] == j else 0
        
        _,_, y = create_reference_matrix_and_count_switches(x_val)
        # for k in range(n_ports):
        #     for t in range(2, T + 1):  # Starting from 2, as we need to access t-1 and t-2
        #         # Adjust for 0-based indexing
        #         y[k, t] = 1 if self.machine_states[k][t - 1] != self.machine_states[k][t - 2] and self.machine_states[k][t - 1] is not None else 0
    
        return x_val, y
    
    def calculate_cost(self):
        # Calculate the total cost based on the machine states and cost list
        total_cost = 0
        for t in range(len(self.config_data.cost)):
            # Count the number of machines used in time period t
            machines_used = sum(1 for machine in self.machine_states if machine[t] is not None)
            # Add the cost for this time period to the total cost
            total_cost += machines_used * self.config_data.cost[t]
        return total_cost
    
    def check_completion(self, batteries):
        # Check if all batteries are fully assigned
        for battery in batteries:
            if self.needs_assignment(battery, len(self.machine_states[0])):
                return False
        return True
    


    def assign_battery(self, j, current_time, switch_count):
        # Assign the battery to a machine while considering the switch count
        for i, machine in enumerate(self.machine_states):
            can_assign = True
            actual_switch_count = 0

            # Check if the battery can be assigned for its full duration from current_time
            for t in range(current_time, min(current_time + self.p_dict_new[j], len(machine))):
                if machine[t] is not None:
                    can_assign = False
                    break
                if t == current_time or (t > 0 and machine[t - 1] != j):
                    actual_switch_count += 1

            # Assign if possible
            if can_assign and actual_switch_count <= switch_count:
                for t in range(current_time, min(current_time + self.p_dict_new[j], len(machine))):
                    machine[t] = j
                return True

        return False
            
    def needs_assignment(self, j, current_time):
    # Check if the battery still needs to be assigned
        required_time = self.p_dict_new[j]
        assigned_time = sum(1 for states in self.machine_states for state in states if state ==j)
        return assigned_time < required_time    
            
    def greedy(self):
        self.p_dict_new={}
        for j in self.config_data.p_dict.keys():
            if j in self.config_data.B_ell[-1]:
                self.p_dict_new[j]= math.ceil(self.config_data.alpha*self.config_data.p_dict[j])
            else:
                self.p_dict_new[j] = self.config_data.p_dict[j]
    # Step 1: Initialization
        T = len(config_data.cost)
        # Map each battery to its group number (priority)
        battery_to_group = {j: i for i, group in enumerate(config_data.B_ell) for j in group}

        #sorted_batteries = sorted(range(1,config_data.batteries+1), key=lambda j: self.p_dict_new[j])  # Sort by shortest processing time
        
        sorted_batteries = sorted( range(1, config_data.batteries + 1), key=lambda j: (battery_to_group.get(j, float('inf')), self.p_dict_new[j]))
        
        self.machine_states = [[None] * T for _ in range(config_data.n_ports)]  # Track the battery assigned to each machine at each time
    
        # Step 2: Assignment Logic
        for t in range(T):
            switch_count = config_data.gamma
            for j in sorted_batteries:
                if self.needs_assignment(j, t):
                    if self.assign_battery(j, t, switch_count):
                        switch_count -= 1

                      
        # Steps 5 & 6: Cost Calculation and Completion Check
        total_cost = self.calculate_cost()
        all_assigned = self.check_completion(sorted_batteries)
        self.x_test, self.y_test=self.generate_dictionaries()
        if all_assigned:
            return total_cost, self.y_test
        else:
           return None, None


# ------ xxx -----------------------------------------------------


def find_list_k(config_data, x_erg, n_l):
    
    k_min = None
    k_max = None
    sum_min = float('inf')  # Initialize to infinity
    sum_max = float('-inf') # Initialize to negative infinity
    
    for k in range(1, config_data.n_ports+1):
        current_sum = 0
        for j in range(1, config_data.batteries+1):
            for t in range(1, n_l[j]+1):
                current_sum += config_data.cost[t-1]*x_erg[j,k,t]  # Using .get to handle missing keys, defaults to 0
        if current_sum < sum_min:
            sum_min = current_sum
            k_min = k
        if current_sum > sum_max:
            sum_max = current_sum
            k_max = k
    return k_min,k_max


def find_list_k_sorted(config_data, x_erg, n_l):

    k_sums = []  # List to hold tuples of (k, sum)

    # Iterate over all k values
    for k in range(1, config_data.n_ports+1):
        current_sum = 0  # Reset current sum for this k
        # Sum across all batteries and time periods
        for j in range(1, config_data.batteries+1):
            for t in range(1, n_l[j]+1):
                current_sum += config_data.cost[t-1] * x_erg[j, k, t]  # Calculate total cost

        k_sums.append((k, current_sum))  # Append the (k, sum) pair to the list

    # Sort the list of k values based on their sums, from largest to smallest
    sorted_k_sums = sorted(k_sums, key=lambda item: item[1], reverse=True)

    # Extract the sorted k values
    sorted_ks = [item[0] for item in sorted_k_sums]

    return sorted_ks, sorted_k_sums


def local_search_port(config_data,n_l,x_best, y_best,x_erg, coeff, curr_best, limit=5):
    
    # It rearranges y by defining a core problem for len(list_k) ports
    #list_k1,list_k2 = find_list_k(config_data, x_erg, n_l)
    list_k_sort, sums  = find_list_k_sorted(config_data, x_erg, n_l) #[list_k1, list_k2,5,6]
    list_k = list_k_sort[: math.ceil(.4*len(list_k_sort))]
    d = {j: sum(x_best[j,k,t] for t in range(1, n_l[j]+1) for k in range(1,config_data.n_ports+1) if k in list_k)  for j in range(1, config_data.batteries+1) }
    
    delta = {t: sum(y_best[k_p,t] for k_p in range(1, config_data.n_ports+1) if k_p not in list_k ) for t in range(1, len(config_data.cost))} # gamma-delta[k] is the maximum available switchings
    
    # assign all relevant jobs on the ports in d with switching allowance provided each time 
    tt=time.time()
    model = gp.Model()
    
    z = model.addVars(gp.tuplelist((j,k,t) for j in d.keys() for k in list_k for t in range(1, n_l[j]+1)  if d[j]!=0 ), vtype=GRB.BINARY)
    y = model.addVars( gp.tuplelist( (k,t) for k in list_k for t in range(1, len(config_data.cost)))   , vtype=GRB.BINARY,name="y")
    
    model.setObjective(gp.quicksum(config_data.cost[t-1] * z[j, k, t] for (j,k,t) in z.keys() ) + coeff*gp.quicksum(y[ k, t] for  (k, t) in y.keys()), GRB.MINIMIZE)
    
    model.addConstrs((gp.quicksum( z[j, k, t] for k in list_k for t in range(1, n_l[j]+1)  if (j,k,t) in z.keys()  ) >= d[j] for j in d.keys() ), "proc_time")
    
    model.addConstrs((gp.quicksum(z[j, k, t] for j in d.keys()  if (j,k,t) in z.keys() ) <= 1 for k in list_k for t in range(1, len(config_data.cost)+1)), "cap")
    
    # Battery assignment
    model.addConstrs((gp.quicksum(z[j, k, t] for k in list_k if (j,k,t) in z.keys() ) + sum(x_best[j,k,t] for k in range(1, config_data.n_ports+1) if k not in list_k) <= 1 for j in d.keys() for t in range(1, n_l[j]+1)), "job_assign")
    
    model.addConstrs( (y[k,t]  if config_data.gamma-delta[t] > 0 else 0     ) >= gp.quicksum(z[j,k,t+1] for j in d.keys() if (j,k,t+1) in z.keys() )- gp.quicksum(z[j,k,t] for j in d.keys() if (j,k,t) in z.keys()    ) for k in list_k for t in range(1, len(config_data.cost) ))
    
    
    model.addConstrs((y[k,t]  if config_data.gamma-delta[t] > 0 else 0     ) >= - (z[j,k,t+1]  if (j,k,t+1) in z.keys() else 0    )    + (z[j,k,t] if (j,k,t) in z.keys() else 0 )  for j in d.keys() for k in list_k for t in range(1, len(config_data.cost) )  )
    
    # Add constraints to limit number of switching
    model.addConstrs(gp.quicksum(y[k,t] for k in list_k if (k,t) in y.keys()) <= config_data.gamma-delta[t] for t in range(1, len(config_data.cost) )   )

    for j in d.keys():
        for k in list_k:
            for t in range(1, n_l[j]+1):
                if (j,k,t) in z.keys(): 
                   z[j,k,t].start=x_best[j,k,t]
        
    model.setParam('TimeLimit', limit)
    end_time = time.time() - tt
    if end_time >50:
        return None, None, None
        
    model.optimize()
    
    if model.solCount==0:
        return None, None, None
    else:
        x_best_new = {(j,k,t):(z[j,k,t].x if (j,k,t) in z.keys() else x_best[j,k,t] ) for (j,k,t) in x_best.keys()}
        y_best_new = {(k,t): (y[k,t].x if (k,t) in y.keys() else y_best[k,t] ) for (k,t) in y_best.keys()}
        #_,_, y_actual = create_reference_matrix_and_count_switches(x_best_new)
        h_upp_news = sum(config_data.cost[t-1] * x_best_new[j, k, t] for (j,k,t) in x_best_new.keys() ) + coeff*sum(y_best_new[ k, t] for  (k, t) in y_best_new.keys())
        return h_upp_news, x_best_new, y_best_new
    
    #
    
#--- xxx ---------------------------------------------------------


class Heuristic_New:
    
    def __init__(self, config_data, n_l={}, y_initial ={}):
        self.config_data=config_data
        self.vec = n_l
        self.y_initial = y_initial # can be empty 
        
    def generate_dictionaries(self):
        x_val = {}
        y = {}
    
        # Adjusted for Python's 0-based indexing
        T= range(1, len(config_data.cost)+1); N= range(1, config_data.n_ports+1); B = range(1, config_data.batteries+1)
        for j in B:
            for k in N:
                for t in T:
                  if t <= self.vec[j]:
                    # Adjust t-1 when accessing machine_states
                    x_val[j, k, t] = 1 if self.machine_states[k-1][t - 1] == j else 0
        
        _,_, y = create_reference_matrix_and_count_switches(x_val)
    
        return x_val, y
    
    def calculate_cost(self):
        # Calculate the total cost based on the machine states and cost list
        total_cost = 0
        for t in range(len(self.config_data.cost)):
            # Count the number of machines used in time period t
            machines_used = sum(1 for machine in self.machine_states if machine[t] is not None)
            # Add the cost for this time period to the total cost
            total_cost += machines_used * self.config_data.cost[t]
        return total_cost
    
    def check_completion(self, batteries):
        # Check if all batteries are fully assigned
        for battery in batteries:
            if self.needs_assignment(battery, len(self.machine_states[0])):
                return False
        return True
    


    def assign_battery(self, j, current_time, switch_count):
        # Assign the battery to a machine while considering the switch count
        for i, machine in enumerate(self.machine_states):
            can_assign = True
            actual_switch_count = 0

            # Check if the battery can be assigned for its full duration from current_time
            for t in range(current_time, min(current_time + self.p_dict_new[j], self.vec[j] )):
                if machine[t] is not None:
                    can_assign = False
                    break
                if t == current_time or (t > 0 and machine[t - 1] != j):
                    actual_switch_count += 1

            # Assign if possible
            if can_assign and actual_switch_count <= switch_count:
                for t in range(current_time, min(current_time + self.p_dict_new[j], self.vec[j])):
                    machine[t] = j
                return True

        return False
            
    def needs_assignment(self, j, current_time):
    # Check if the battery still needs to be assigned
        required_time = self.p_dict_new[j]
        assigned_time = sum(1 for states in self.machine_states for state in states if state ==j)
        return assigned_time < required_time    
            
    def greedy(self):
        self.p_dict_new={}
        for j in self.config_data.p_dict.keys():
            if j in self.config_data.B_ell[-1]:
                self.p_dict_new[j]= math.ceil(self.config_data.alpha*self.config_data.p_dict[j])
            else:
                self.p_dict_new[j] = self.config_data.p_dict[j]
    # Step 1: Initialization
        T = len(config_data.cost)
        # Map each battery to its group number (priority)
        battery_to_group = {j: i for i, group in enumerate(config_data.B_ell) for j in group}

        #sorted_batteries = sorted(range(1,config_data.batteries+1), key=lambda j: self.p_dict_new[j])  # Sort by shortest processing time
        
        sorted_batteries = sorted( range(1, config_data.batteries + 1), key=lambda j: (battery_to_group.get(j, float('inf')), self.p_dict_new[j]))
        
        self.machine_states = [[None] * T for _ in range(config_data.n_ports)]  # Track the battery assigned to each machine at each time
    
        # Step 2: Assignment Logic
        for t in range(T):
            switch_count = config_data.gamma
            for j in sorted_batteries:
                if self.needs_assignment(j, t):
                    if self.assign_battery(j, t, switch_count):
                        switch_count -= 1

                      
        # Steps 5 & 6: Cost Calculation and Completion Check
        total_cost = self.calculate_cost()
        all_assigned = self.check_completion(sorted_batteries)
        self.x_test, self.y_test=self.generate_dictionaries()
        if all_assigned:
            return total_cost, self.y_test
        else:
           return None, None


  
# ---------------------------------------------------------------

     

class Main_loop:  

    def __init__(self, method,x,s, mu,h_mu_current, config_data, tau=0.5):
        self.method=method
        self.x=x
        self.s=s
        self.tau=tau
        self.mu=mu
        self.h_mu_current = h_mu_current
        self.iter =1
        self.config_data = config_data
        self.min_cost = min(self.config_data.cost)
        
    

    def MDS_method(self):
        sub_k = {(j,k,t): (self.x[j,k,t]-self.s[j,k,t]) for (j,k,t) in self.mu.keys()} # subgradient
        
        dot_subk_subgrad_prev = sum(sub_k[key] * self.subgradient[key] for key in self.mu.keys())
        
        norm_subk=math.sqrt(sum(value**2 for value in sub_k.values()))
        
        norm_subgrad_prev= math.sqrt(sum(value**2 for value in self.subgradient.values()))
        
        alpha_k = max(0,-( (dot_subk_subgrad_prev)/(norm_subgrad_prev*norm_subk) ))
        
        
        
        zeta_k = 1/(2-alpha_k)
        
        T_k_MGT = max(0, ( (-zeta_k * dot_subk_subgrad_prev)/(norm_subgrad_prev**2) ) )
        
        T_k_ADS = norm_subk/norm_subgrad_prev
        
        T_k_MDS = (1-alpha_k)*T_k_MGT + alpha_k*T_k_ADS
        
        self.subgradient = copy.deepcopy(sub_k + T_k_MDS*self.subgradient) 
        
        
    def compute_dual_multiplier(self):
        self.mu = {key: self.mu[key] + (self.tau * self.subgradient[key]) for key in self.mu}
        #self.mu = {key: (self.mu[key] + (self.tau * self.subgradient[key]) ) if self.subgradient[key]>= 0 else self.mu[key]/2 for key in self.mu}
    def euclidean_norm(self):
        return math.sqrt(sum(value ** 2 for value in self.subgradient.values()))
   
    def compute_step_length(self,theta,h_upp):
        ggap = (h_upp-self.h_mu_current)
        #ggap = (2100-self.h_mu_current)
        
        self.tau = (theta*ggap)/((self.euclidean_norm())**2)
        #self.tau = theta*0.8
        assert self.tau >= 0
        
    def compute_subgradient(self):
        if self.method=='simple':
            self.subgradient = {(j,k,t): (self.x[j,k,t]-self.s[j,k,t]) for (j,k,t) in self.mu.keys()}
        else:
            if self.iter == 1:
                self.subgradient = {(j,k,t): (self.x[j,k,t]-self.s[j,k,t]) for (j,k,t) in self.mu.keys()}
                
            else: 
                self.MDS_method()


    
def initialize_config(batteries, n_ports, gamma):
    # Initialize and return the ConfigData object
    seed_value = 46
    np.random.seed(seed_value)
    random.seed(seed_value)
    return ConfigData(
        batteries=batteries,
        n_ports=n_ports,
        gamma=gamma,
        proportion=[0.50, 0.20, 0.30],
        alpha=0.8,  # Example value for alpha
        demand_window=[14, 19]
    )



def map_entries_to_row_index(arr):
    """
    Maps each entry in the array of lists to its row index.
    
    Parameters:
        arr (numpy.ndarray): An array of lists.
    
    Returns:
        dict: A dictionary where each key is an entry from the lists and its value is the row index.
    """
    entry_to_row_index = {}
    for row_index, lst in enumerate(arr):
        for entry in lst:
            # You might want to check if the entry already exists and handle it accordingly,
            # depending on whether entries can repeat across rows.
            entry_to_row_index[entry] = row_index
    return entry_to_row_index


"""
High-extended
slopes={0: -0.4637, 1: -0.485, 2: -0.42} , intercepts={0: 17.10, 1: 18.194, 2: 16.54} 
slopes={0: -0.349, 1: -0.46, 2: -0.421} , intercepts={0: 7.739, 1: 10.6379, 2: 10.018}

"""
def new_initialize_mu(config_data,vec, slopes = {0: -0.339, 1: -0.391, 2: -0.33}, intercepts={0: 9.2, 1: 10.866, 2: 9.79}): #  slopes = {0: -0.45 , 1: -0.45, 2: -0.35}, intercepts = {0: 11.22, 1: 11.30, 2: 9.83} slopes = {0: -0.455, 1: -0.4452, 2: -0.33900}, intercepts = {0: 10.92722, 1: 10.821, 2: 9.3494}  slopes={0: -0.319, 1: -0.381, 2: -0.318}, intercepts={0: 8.004, 1: 9.80, 2: 8.892} slopes={0: -0.244, 1: -0.384, 2: -0.277}, intercepts={0: 5.728, 1: 9.363, 2: 7.117}
    mu_init={}; bat_to_cat= map_entries_to_row_index(config_data.B_ell)
    for (j,k,t) in vec:
        cat = bat_to_cat[j]
        mu_init[j,k,t] = predict_mu(config_data.cost[t-1], slopes[cat], intercepts[cat])#0#((1-beta)*config_data.cost[t-1])-.01
    return mu_init



def update_c_bar(mu):
    c_bar = {(j,k,t): round((1-beta)*config_data.cost[t-1], 2) - mu[j,k,t] for (j,k,t) in mu.keys()}
    return c_bar



def s_k_rule(k, iters):
    t=iters+1
    first_term = (sum( (s+1)**k for s in range(t-2+1)))/(sum( (s+1)**k for s in range(t-1+1)))
    
    second_term = (t**k)/(sum( (s+1)**k for s in range(t-1+1)  ))
    
    return first_term, second_term

def find_incompatible_time_blocks(selected_time_block, all_time_blocks):
    """
    Finds and returns all time blocks that are incompatible with the selected time block
    according to the defined rules (overlapping or sharing an endpoint except at the boundaries).

    Args:
    selected_time_block (tuple): The selected time block (t1, t2).
    all_time_blocks (list): A list of all possible time blocks.

    Returns:
    list: A list of tuples representing incompatible time blocks.
    """
    # Extract the start and end times of the selected time block
    selected_start, selected_end = selected_time_block

    # Initialize an empty list to store incompatible time blocks
    incompatible_blocks = []

    # Iterate over all time blocks to find which are incompatible
    for block in all_time_blocks:
        block_start, block_end = block

        # Check for overlap or direct adjacency (excluding end-to-start adjacency that is allowed)
        if block_start < selected_end and block_end > selected_start:
            if block!= selected_time_block:
                incompatible_blocks.append(block)

    return incompatible_blocks  




def plot_two_algorithm_bounds(time_iter1, h_low1, h_upp1, time_iter2, h_low2, h_upp2):
    # Compute cumulative time for each algorithm
    cumulative_time1 = np.cumsum(time_iter1)
    cumulative_time2 = np.cumsum(time_iter2)
    
    # Prepare step-wise data for plotting
    def prepare_steps(cumulative_time, h_values):
        return np.repeat(cumulative_time, 2)[1:], np.repeat(h_values, 2)[:-1]
    
    cumulative_time1_steps, h_low1_steps = prepare_steps(cumulative_time1, h_low1)
    cumulative_time1_steps, h_upp1_steps = prepare_steps(cumulative_time1, h_upp1)
    cumulative_time2_steps, h_low2_steps = prepare_steps(cumulative_time2, h_low2)
    cumulative_time2_steps, h_upp2_steps = prepare_steps(cumulative_time2, h_upp2)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.step(cumulative_time1_steps, h_upp1_steps, label='$\\bar{h}$  with $P^{y-s}$', linewidth=2, where='post', color='blue')
    plt.step(cumulative_time1_steps, h_low1_steps, label='${h}$  with $P^{y-s}$', linestyle='--', where='post', color='blue')
    plt.step(cumulative_time2_steps, h_upp2_steps, label='$\\bar{h}$  with $\\tilde{P}^{y-s}$', linewidth=2, where='post', color='red')
    plt.step(cumulative_time2_steps, h_low2_steps, label='${h}$  with $\\tilde{P}^{y-s}$', linestyle='--', where='post', color='red')

    # Filling areas
    plt.fill_between(cumulative_time1_steps, h_low1_steps, h_upp1_steps, step='post', color='blue', alpha=0.1)
    plt.fill_between(cumulative_time2_steps, h_low2_steps, h_upp2_steps, step='post', color='red', alpha=0.1)
    # Labels and legend
    plt.xlabel('Cumulative Time', fontsize=18)
    plt.ylabel('Bounds', fontsize=18)
    plt.legend(loc='upper right',fontsize=20)
    #plt.title('Comparison of Step-wise Upper and Lower Bounds Over Time')
    plt.savefig("200_old_new.pdf", format='pdf')
    # Show the plot
    plt.show()
    
    # Compute primal-dual integral for each algorithm
    integral1 = trapz(h_upp1_steps - h_low1_steps, cumulative_time1_steps)
    integral2 = trapz(h_upp2_steps - h_low2_steps, cumulative_time2_steps)
    
    return integral1, integral2


def write_data_to_json(data, filename, dest_folder):
    """
    Writes a dictionary of various data types to a JSON file in the specified destination folder.

    Parameters:
        data (dict): A dictionary containing the data to be written.
        filename (str): The name of the file to create, without the extension.
        dest_folder (str): The folder where the JSON file will be saved.

    Returns:
        None
    """
    # Ensure the destination folder exists, create if it does not
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Full path to the output file
    file_path = os.path.join(dest_folder, f"{filename}.json")
    
    # Write the data to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)  # Use indent for pretty printing

    print(f"Data written to {file_path}")
    

# Use the function and print the primal-dual integrals for both algorithms
# integral1, integral2 = plot_two_algorithm_bounds(time_iter1, h_low1, h_upp1, time_iter2, h_low2, h_upp2)
# print(f'Primal-Dual Integral for Algorithm 1: {integral1}')
# print(f'Primal-Dual Integral for Algorithm 2: {integral2}')


#%%

os.makedirs('plots', exist_ok=True)

if __name__ == "__main__": 
    
#----------------------------------x
# ----| Initialize |---------------x 
# ---------------------------------x

  dest_for_input = "C:\\Users\\sunney\\OneDrive - Chalmers\\Desktop\\PostDoc_papers\\Paper_1\\Code\\output"
  data_transfers = False
  list_price_variation = [('base', 'base')]  #('low', 'extended')]
  subgradient_method = ["simple", "nsimple" ] # nsimple: MDS , reg: regression 
  regress =[True, False]  #  False
  heuristic = ["var-fix-binP", "var-fix"]    # var-fix: without binP 
  local_search = [True, False]          # no
  
  
  num_variations = generate_variations(subgradient_method,regress, heuristic, local_search)
  
  
#----------------------------------    
    
  threshold =.5
  combinations =[ [100,50,13], [150,75,19],[200,100,25], [250,125,32], [300,140,35], [350,140,35], [400,140,45]]# [100,50,13], [150,75,19],[200,100,25], [250,125,32], [300,140,35], [350,140,35], [400,140,45][ [100,50,13], [150,75,19],[200,100,25], [250,125,32], [300,140,35], [350,140,35], [400,140,45]]#[[150,75,19],[200,100,25], [250,125,32], [300,140,35], [350,140,35], [400,140,45] ] 
  beta=0.5
  #reg = True
  for price_variations in list_price_variation:
    for vari in num_variations.keys():
     if vari in [5,10,13]:  
        (sub,reg,heu,loc) = num_variations[vari]      
      
      
        method ='simple' if sub== 'simple' else 'nsimple' # nsimple is deflected and simple is the conventional subgradient
        learning = reg
        
        
        solve_YS_Using = 'New'#'New' 
        Form = 'F2'
        binP = True  if heu == 'var-fix-binP' else False
        reduction = 1
        total_iterations = 3000
        solve_px_direct= False      # use min-cost network flow
        Total_time_limit = 3600
        regress_plot=True
        ergodic=loc
        local_search= loc
     else:
         continue
     
        
     
      
      
     for l in combinations:  
        config_data = initialize_config(l[0], l[1], l[2])
        if price_variations != ('base', 'base'):
            config_data.cost[6] = config_data.cost[7]-1; config_data.cost[8] = config_data.cost[7]+1
            config_data.cost[18] = config_data.cost[19]-1; config_data.cost[20] = config_data.cost[20]+8 
            #config_data.cost = generate_hourly_prices(24, volatility=price_variations[0], peak_duration=price_variations[1], cost_original=None)[1]
        if data_transfers==True:
            store_input_data = f'input_data_file_{l[0]}_{l[1]}_{l[2]}'
            
            data_transfer = {'B_ell': [list(group) for group in config_data.B_ell] , 
                             'alpha':config_data.alpha,
                             'elec_cost':[round(i,2) for i in config_data.cost],
                             'n_l':[config_data.demand_window[0], config_data.demand_window[1], 24], 
                             'L':3,
                             'T':24,
                             'p': config_data.p_dict,
                             'w': (min(config_data.cost) / ((config_data.gamma * (len(config_data.cost) - 1)) + 1))   
                             }
            write_data_to_json(data_transfer, store_input_data, dest_for_input)
            
        if data_transfers == True:
            continue
        best_elect_cost = None
        coeff = (min(config_data.cost) / ((config_data.gamma * (len(config_data.cost) - 1)) + 1))
        T_blocks = generate_two_tuples(len(config_data.cost) + 1)
        
        
        vec=[(j,k,t) for l in range(config_data.ell) for j in config_data.B_ell[l] for k in range(1, config_data.n_ports+1) for t in range(1, config_data.demand_window[l]+1 if l<=config_data.ell-2 else len(config_data.cost)+1)]
    
        initial_val = new_initialize_mu(config_data, vec)
        mu = {(j,k,t):initial_val[j,k,t] for (j,k,t) in vec}
        c_bar = {(j,k,t): round((1-beta)*config_data.cost[t-1], 2) - mu[j,k,t] for (j,k,t) in mu.keys()}
        
        # to be retreieved
        h_mu ={};h_mu[1] = 0
        h_x_mu={};h_ys_mu={}; total_time = {}

        status_x={}; status_ys={}; time_x={}; time_ys={}
        n_l = {j: 14 if j in config_data.B_ell[0] else (19 if j in config_data.B_ell[1] else 24) for j in range(1, config_data.batteries+1)}
        theta =1.99 if learning == True else 1.0
        x_best=None; y_best= None
        
        
        heur_time_start = time.time()
        H=Heuristic_New(config_data, n_l)
        h_upp={};h_upp_time=[]; h_low=[]; time_iter = []
        val,y_init = H.greedy()
        h_upp[1] = round(val,3) + coeff*sum(y_init[ll] for ll in y_init.keys()) if val is not None else 30000
        
        if val is not None:
                  if binP == True:
                      h_upp_new, y_opt, elect_cost, x_sol_h = Full_Model_BinP(config_data, c_bar,  coeff, y_init, best_elect_cost)
                  else:
                      h_upp_new, y_opt, x_sol_h = Full_Model_x_y(config_data, c_bar,  coeff, y_init)
                  if (h_upp_new is not None   ):  #and h_upp_new < min(h_upp.values())
                      h_upp_new = round(h_upp_new,3)
    
                      h_upp[1]=round(h_upp_new,3)
                      y_best = copy.deepcopy(y_opt)
                      x_best = copy.deepcopy(x_sol_h)
                  else:
                    h_upp[1] = 30000 
    
        heur_time = time.time() - heur_time_start
        
    
    # -------------------| Timed |--------------
    
        start_time= time.time()
        if solve_px_direct==False:
            MN=MinCostFlowNetwork_ORTools_News(config_data, mu, beta=0.5)
    
        x, h_x, status_x_t  = model_Px(config_data, mu, beta=beta) if solve_px_direct ==True else MN.solve_problem(mu)
        end_time= time.time()
        time_x[1]= end_time - start_time
        
        status_x[1]=status_x_t
        
        
        if solve_YS_Using=='New':
            h_ys, s, y, time_all_iter  = solve_ys(config_data,c_bar,coeff, T_blocks, formulation=Form)
        else:
            h_ys, s, y, time_all_iter = solveYS_original(config_data,c_bar)
        
        time_ys[1]= time_all_iter
        
        last_h=0
        iters=0
        if regress_plot:
            new_plot_mu_vs_cost_with_regression(mu, config_data.cost, iters, h_x+h_ys, config_data)
        
        countss=0
        #y_best= y_opt#{}
        #x_best=None
        
        total_time[1] = max(time_x[1], time_ys[1])
        x_erg =copy.deepcopy(x)
        h_mu[1] = h_x+h_ys
        for iters in range(1, total_iterations+1):
            
           start_time_initial = time.time() 
            
           if sum(total_time.values())>Total_time_limit:
                if binP == False:
                    h_upp_new, y_opt,x_sol_h = Full_Model_x_y(config_data, c_bar,  coeff, y_best)
                else:
                   if y_best is None:
                       break
                   h_upp_new, y_opt, elect_cost,x_sol_h = Full_Model_BinP(config_data, c_bar,  coeff, y_best, best_elect_cost)
                if (h_upp_new is not None and (h_upp_new < min(h_upp.values())   )  ):  #
                    h_upp_new = round(h_upp_new,3)
     
                    h_upp[iters]=round(h_upp_new,3)
                    y_best = copy.deepcopy(y_opt)
                    x_best= copy.deepcopy(x_sol_h)
                time_to_add =   time.time() - start_time_initial
                total_time[iters] = total_time[iters]+ time_to_add
                break
           if theta > 1e-2: 

            if h_upp[iters] is not None and abs(min(h_upp.values())-max(h_mu.values())) <=1:
                break
            Update = Main_loop(method, x, s, mu, h_x+h_ys, config_data)
            Update.compute_subgradient()
            Update.compute_step_length(theta, (h_upp[iters] if h_upp[iters] is not None else 30000  ))
            Update.compute_dual_multiplier()
            mu= Update.mu
            
            #current_subgrad= copy.deepcopy(Update.subgradient)
            if learning == True:
                if iters%reduction==0:
                    
                    y_values = [mu[j, k, t] for (j, k, t) in mu.keys()]
                    x_values = [config_data.cost[t-1] for (j, k, t) in mu.keys()]
                    #slope, intercept, _, _, _ = stats.linregress(x_values,y_values )
                    slopes, intercepts = get_regression_parameter(mu, config_data)
                    initial_val = new_initialize_mu(config_data,vec, slopes, intercepts)
                    mu = {(j,k,t):(initial_val[j,k,t]) for (j,k,t) in vec  }
                    
                    
                    
                    
            
            if iters%10==0:
                new_h = max(h_mu.values())
                if new_h<= last_h:
                    theta=0.95*theta
                else:
                    last_h=new_h
            h_upp[iters+1]=round(h_upp[iters],3) if h_upp[iters] is not None else h_upp[iters]
            c_bar = update_c_bar(mu)
            if solve_YS_Using=='New':
                h_ys, s, y, time_all_iter  = solve_ys(config_data,c_bar,coeff, T_blocks, formulation=Form,remove={}, y_start=y)
            else:
                h_ys, s, y, time_all_iter = solveYS_original(config_data,c_bar)
            
            
            time_ys[iters+1]=time_all_iter
        
            # solve the updated subproblem
            start_time= time.time()
            x,h_x, status_x_t = model_Px(config_data, mu) if solve_px_direct ==True else MN.solve_problem(mu)
            end_time= time.time()
            time_x[iters+1]=end_time-start_time
            status_x[iters+1]=status_x_t
            if regress_plot:
                new_plot_mu_vs_cost_with_regression(mu, config_data.cost, iters, h_x+h_ys, config_data)
                
            if iters%20==0 or (h_upp[iters]>=30000 and iters>=5 ):
                if y!=y_best:
                   if binP == False:
                       h_upp_new, y_opt, x_sol_h = Full_Model_x_y(config_data, c_bar,  coeff, y)
                   else:
                      h_upp_new, y_opt, elect_cost,x_sol_h = Full_Model_BinP(config_data, c_bar,  coeff, y, best_elect_cost)
                   if (h_upp_new is not None and h_upp_new < min(h_upp.values())  ):  #
                       h_upp_new = round(h_upp_new,3)
    
                       h_upp[iters+1]=round(h_upp_new,3)
                       y_best = copy.deepcopy(y_opt)
                       x_best= copy.deepcopy(x_sol_h)
            end_local=0
            if iters%1==0 and ergodic==True:
                    first_term, second_term = s_k_rule(5, iters)
                    for (j,k,t) in x_erg.keys():
                        x_erg[j,k,t] = x_erg[j,k,t]*first_term + x[j,k,t]*second_term
                    if iters%10==0 and x_best is not None:
                       limit = 40 if iters%20 ==0 else 50 
                       local_time = time.time()
                       h_upp_news, x_best_new, y_best_new = local_search_port(config_data,n_l,x_best, y_best,x_erg, coeff, min(h_upp.values()), limit)
                       end_local = time.time()-local_time
                       
                       if (h_upp_news is not None and h_upp_news < min(h_upp.values())  ):  #
                           h_upp_news = round(h_upp_news,3)
        
                           h_upp[iters+1]=round(h_upp_news,3)
                           y_best = copy.deepcopy(y_best_new)
                           x_best= copy.deepcopy(x_best_new)
     
                       
            sorted_two = sorted([time_x[iters+1], time_ys[iters+1], end_local   ])           
            final_time = time.time()-start_time_initial - sorted_two[0]-sorted_two[1]
            total_time[iters+1] = final_time 
            h_mu[iters+1] =h_x+h_ys
            h_x_mu[iters+1] = h_x
            h_ys_mu[iters+1] =h_ys 
            # if x_best is not None:
            #     print('-------------- here')
            #     break
            h_upp_time.append(min(h_upp.values())); h_low.append(max(h_mu.values())); time_iter.append(final_time)    
            
        
        # Calculate total time and maximum h_mu value
        total_time_x = sum(time_x.values())
        total_time_ys = sum(time_ys.values())
        total_time_sum = sum(total_time.values())
        max_h_mu = max(h_mu.values())
    
    
        total_h_upp = min(h_upp.values()) if h_upp else None
    
        base_filename = f"output_{l[0]}_{l[1]}_{l[2]}_{price_variations[0]}_{price_variations[1]}_"
        save_dict_to_csv_with_params(h_mu, base_filename + '_h_mu', config_data, vari)
        save_dict_to_csv_with_params(h_x_mu, base_filename + '_h_x_mu', config_data, vari)
        save_dict_to_csv_with_params(h_ys_mu, base_filename + '_h_ys_mu', config_data, vari)
        save_dict_to_csv_with_params(h_upp, base_filename + '_h_upp', config_data, vari)
        save_dict_to_csv_with_params(time_x, base_filename + '_time_x', config_data, vari)
        save_dict_to_csv_with_params(time_ys, base_filename + '_time_ys', config_data, vari)
        save_dict_to_csv_with_params(total_time, base_filename + '_time_total', config_data, vari)
        data_merged = _merge_dicts_to_array(h_upp,h_mu, total_time) ; column_names = ['Iteration', 'h_up', 'h_low', 'time']
        save_array_to_csv_with_params_new(data_merged, base_filename + '_overall_', config_data, column_names, output_folder_path, vari)
        
        
        
        
        # Plotting for each combination 
        plot_combined_time_data(time_x, time_ys,total_time, 'Time for Subproblems over Iterations', 'Iteration', 'Time (seconds)', 'Subproblem X', 'Subproblem YS','Total', base_filename + f'_combined_time_plot_{price_variations[0]}_{price_variations[1]}.png', total_time_x, total_time_ys, total_time_sum)
        #plot_combined_h_data(h_mu, h_x_mu, h_ys_mu, h_upp, 'h Values over Iterations', 'Iteration', 'Value', base_filename + '_combined_h_plot.png', max_h_mu)
        plot_combined_h_data_New(h_mu, h_x_mu, h_ys_mu, h_upp, 'h Values over Iterations', 'Iteration', 'Value', base_filename + f'_combined_h_plot__{price_variations[0]}_{price_variations[1]}.png', max_h_mu)
        
        plot_combined_time_data(time_x, time_ys,total_time, 'Time for Subproblems over Iterations', 'Iteration', 'Time (seconds)', 'Subproblem X', 'Subproblem YS','Total', base_filename + f'_combined_time_plot_{price_variations[0]}_{price_variations[1]}.eps', total_time_x, total_time_ys, total_time_sum)
        plot_combined_h_data_New(h_mu, h_x_mu, h_ys_mu, h_upp, 'h Values over Iterations', 'Iteration', 'Value', base_filename + f'_combined_h_plot__{price_variations[0]}_{price_variations[1]}.eps', max_h_mu)
    


