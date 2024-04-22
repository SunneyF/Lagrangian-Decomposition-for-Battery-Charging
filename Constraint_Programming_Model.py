# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:04:38 2024

@author: Sunney

This module uses the OR-Tools CP-SAT solver to optimize battery charging schedules across multiple ports,
taking into account electricity costs and operational constraints.
"""

import math
import time
import numpy as np
import random
import helper_function as hf
from ortools.sat.python import cp_model


class ConfigData:
    def __init__(self, batteries, n_ports, gamma, proportion, alpha, demand_window):
        """
        Initialize the configuration data for the system.

        :param batteries: Number of batteries in the system.
        :param n_ports: Number of charging ports available.
        :param gamma: Maximum number of allowed switchovers.
        :param proportion: Proportion of deliveries to be made in each time window and the last.
        :param alpha: Lower bound of the state-of-charge of remaining batteries.
        :param demand_window: Time period when deliveries are to be made.
        """
        self.batteries = batteries
        self.n_ports = n_ports
        self.gamma = gamma
        self.proportion = proportion
        self.ell = len(proportion)
        self.alpha = alpha
        self.demand_window = demand_window
        self.cost = hf.generate_electricity_prices(24, 60)
        p = hf.create_mixed_distribution_vector(self.batteries)
        self.p_dict = {i + 1: p[i] for i in range(len(p))}  # Time to charge the batteries
        self.B_ell = hf.distribute_B_based_on_p(
            range(1, self.batteries + 1), self.p_dict, self.ell, self.proportion
        )
        self.validate()

    def validate(self):
        """Validate the configuration data."""
        assert len(self.demand_window) == self.ell - 1, "Demand window does not match ell."


def initialize_config(batteries, n_ports, gamma):
    """
    Initialize and return the configuration data object.
    
    :param batteries: Number of batteries.
    :param n_ports: Number of ports.
    :param gamma: Maximum allowed switchovers.
    :return: ConfigData instance.
    """
    seed_value = 46
    np.random.seed(seed_value)
    random.seed(seed_value)
    return ConfigData(
        batteries=batteries,
        n_ports=n_ports,
        gamma=gamma,
        proportion=[0.50, 0.20, 0.30],
        alpha=0.8,
        demand_window=[14, 19]
    )


def main(config_data):
    """
    Main function to setup and solve the CP-SAT model.

    :param config_data: Configuration data.
    :return: Solution information.
    """
    loading_start = time.time()
    vec = [
        (j, k, t) for l in range(config_data.ell)
        for j in config_data.B_ell[l]
        for k in range(1, config_data.n_ports + 1)
        for t in range(1, config_data.demand_window[l] + 1 if l <= config_data.ell - 2 else len(config_data.cost) + 1)
    ]
    
    model = cp_model.CpModel()

    # Parameters
    num_ports = config_data.n_ports
    num_periods = len(config_data.cost)
    num_batteries = config_data.batteries
    gamma = config_data.gamma  # Maximum allowed switchovers
    coeff = (min(config_data.cost) / ((gamma * (len(config_data.cost) - 1)) + 1))

    # Decision Variables
    x = {}
    y = {}
    for j in range(1, num_batteries + 1):
        for k in range(1, num_ports + 1):
            for t in range(1, num_periods + 1):
                if (j, k, t) in vec:
                    x[(j, k, t)] = model.NewBoolVar(f'x[{j},{k},{t}]')
    
    for k in range(1, num_ports + 1):
        for t in range(1, num_periods):
            y[(k, t)] = model.NewBoolVar(f'y[{k},{t}]')

    # Constraints
    for l in range(config_data.ell - 1):
        for j in config_data.B_ell[l]:
            model.Add(
                sum(x[(j, k, t)] for k in range(1, num_ports + 1) for t in range(1, num_periods + 1) if t <= config_data.demand_window[l] if (j, k, t) in x.keys()) == config_data.p_dict[j]
            )
    
    for j in config_data.B_ell[config_data.ell - 1]:
        model.Add(
            sum(x[(j, k, t)] for k in range(1, num_ports + 1) for t in range(1, num_periods + 1) if (j, k, t) in x.keys()) >= math.ceil(config_data.alpha * config_data.p_dict[j])
        )

    for k in range(1, num_ports + 1):
        for t in range(1, num_periods + 1):
            model.Add(
                sum(x[(j, k, t)] for j in range(1, num_batteries + 1) if (j, k, t) in x.keys()) <= 1
            )

    for t in range(1, num_periods):
        model.Add(
            sum(y[(k, t)] for k in range(1, num_ports + 1)) <= gamma
        )

    # Objective
    total_cost = sum(config_data.cost[t-1] * x[(j, k, t)] for (j, k, t) in x.keys())
    switch_cost = sum(coeff * y[(k, t)] for (k, t) in y.keys())
    model.Minimize(total_cost + switch_cost)

    # Solve the model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3600.0
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Optimal or feasible solution found:')
        for j in range(1, num_batteries + 1):
            for k in range(1, num_ports + 1):
                for t in range(1, num_periods + 1):
                    if (j, k, t) in x.keys() and solver.Value(x[(j, k, t)]):
                        print(f'Battery {j} is assigned to port {k} at time {t}.')
        return {key: solver.Value(x[key]) for key in x}, solver.ObjectiveValue(), solver.BestObjectiveBound()
    else:
        print('No solution found.')
        return None, None, None


if __name__ == '__main__':
    config_data = initialize_config(150, 75, 19)
    x_sol, obj_val, obj_bnd = main(config_data)
