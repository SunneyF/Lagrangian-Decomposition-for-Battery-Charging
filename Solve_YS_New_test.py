# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:56:41 2024

@author: sunney

"""

import gurobipy as gp
from gurobipy import GRB
import time


def calculate_cost_for_time_blocks(config_data, c_bar, T_blocks):
    """
    Calculate the cost for each time block (t1, t2) such that for each time block,
    we choose the least value over j from c_bar. Remove any time block with non-negative cost
    and record the job j for which the cost was negative.

    :param J: Number of jobs
    :param T: Number of time periods
    :param c_bar: Dictionary containing costs c_bar[j, t] for job j and time t
    :return: Two dictionaries - one for the costs of each time block and another for the corresponding job j
    """
    
    cost_time_block = {}
    job_for_time_block = {}
    for k in range(1, config_data.n_ports+1):  
        for t1, t2 in T_blocks:
            min_cost = float('inf')
            min_cost_job = None
            for j in range(1, config_data.batteries+1):
               if (j,k,t2) in c_bar.keys(): 
                cost = sum(c_bar[j, k,t] for t in range(t1 + 1, t2 + 1) )
                if cost < min_cost:
                    min_cost = cost
                    min_cost_job = j
    
            if min_cost < 0:
                cost_time_block[(k,t1, t2)] = min_cost
                job_for_time_block[(k,t1, t2)] = min_cost_job

    return cost_time_block, job_for_time_block



def filter_time_blocks(cost_time_block, job_for_time_block, config_data):
    """
    Filters out overlapping time blocks with the same job assignments for a given k,
    if a broader encompassing time block with the same job exists.
    
    :param cost_time_block: Dictionary with costs for each time block and machine.
    :param job_for_time_block: Dictionary with job assignments for each time block and machine.
    :return: Filtered versions of cost_time_block and job_for_time_block dictionaries.
    """
    # Initialize containers for items to remove
    to_remove = set()

    # Check for each k separately
    for k in range(1, config_data.n_ports + 1):
        # Find all time blocks for this k
        time_blocks_for_k = [(t1, t2) for (k_, t1, t2) in job_for_time_block.keys() if k_ == k]
        
        # Sort time blocks to compare them sequentially
        time_blocks_for_k.sort(key=lambda x: x[0])
        
        for i, (t1, t2) in enumerate(time_blocks_for_k):
            for j, (t3, t5) in enumerate(time_blocks_for_k[i+1:], start=i+1):
                # Check if there is a direct continuation and the same job assignment
                if t2 == t3 and job_for_time_block[(k, t1, t2)] == job_for_time_block[(k, t3, t5)]:
                    # Check if the encompassing block exists
                    if (k, t1, t5) in job_for_time_block:
                        if job_for_time_block[(k, t1, t2)] == job_for_time_block[(k, t1, t5)]:
                            # Mark smaller blocks for removal
                            to_remove.add((k, t1, t2))
                            to_remove.add((k, t3, t5))
    
    # Remove marked items
    for item in to_remove:
        if item in cost_time_block:
            del cost_time_block[item]
        if item in job_for_time_block:
            del job_for_time_block[item]

    return cost_time_block, job_for_time_block

def extract_s(lambda_var_opt, job_time_block, c_bar):
    s={}
    for (k,t1,t2) in lambda_var_opt.keys():
        if lambda_var_opt[k,t1,t2]==1:
           for t in range(t1+1, t2+1):  
               s[job_time_block[k,t1,t2], k, t]=1 
    for (j,k,t) in c_bar.keys():
        if (j,k,t) not in s.keys():
            s[j,k,t]=0
    return s



def solve_ys(config_data,c_bar,coeff, T_blocks, formulation='F2',remove={}, y_start={}):
    start_time = time.time()
    
    cost_time_block, job_time_block = calculate_cost_for_time_blocks(config_data, c_bar, T_blocks)
    
    model=gp.Model()
    lambda_var= model.addVars(gp.tuplelist( (k,t1, t2) for (k,t1, t2) in cost_time_block.keys()  ), vtype=GRB.BINARY) #vtype=GRB.BINARY
    y = model.addVars(gp.tuplelist( (k,t) for k in range(1, config_data.n_ports+1) for t in range(1, len(config_data.cost))), vtype=GRB.BINARY)
    
    model.setObjective(gp.quicksum(cost_time_block[k,t1,t2]*lambda_var[k,t1, t2] for (k,t1, t2) in lambda_var.keys() ) + coeff *(gp.quicksum(y[k,t]  for (k,t) in y.keys())) )
    if formulation=='Old':
        model.addConstrs(lambda_var[k,t1,t2] + lambda_var[k,t1, t] <= 1  for (k,t1, t2) in lambda_var.keys() for t in range(t1+1,t2) if (t2-t1 >1 and t1!=t and (k,t1,t) in lambda_var.keys()  )  )
        model.addConstrs(lambda_var[k,t1,t2] + lambda_var[k,t, t2] <= 1  for (k,t1, t2) in lambda_var.keys() for t in range(t1+1,t2+1) if (t2-t1 >1 and t2!=t and (k,t,t2) in lambda_var.keys() ))
    elif formulation =='F1':
        model.addConstrs(gp.quicksum(lambda_var[k,t_p, t_pp] for t_p in range(t1+1, t2-1+1) if (k,t_p, t_pp) in lambda_var.keys()  ) <= (1-lambda_var[k,t1,t2])*(t2-t1-1)  for (k,t1, t2) in lambda_var.keys() for t_pp in range(t1+1, t2) if (t2-t1)!=1   )
    elif formulation =='F2':
        model.addConstrs(lambda_var[k,t1,t2] + lambda_var[k,t1_p, t2_p] <= 1 for (k,t1,t2) in lambda_var.keys() for (ll,t1_p, t2_p) in lambda_var.keys() if (t1_p <t2 and t2_p>t1 and (t1, t2)!=(t1_p, t2_p) and ll==k)  )

    # Existing model setup
    # ...
    
    # Additional constraints to prevent overlapping time blocks
    if formulation!='F2' or formulation!='F3' : 
        for (k,t1, t2) in lambda_var.keys():
            model.addConstrs(lambda_var[k,t1, t2] + (lambda_var[k,t1_prime, t2_prime] )   <= 1 
                             for (zz,t1_prime, t2_prime) in lambda_var.keys() 
                             if (t1 < t2_prime <= t2 or t1 <= t1_prime < t2) and (t1_prime, t2_prime) != (t1, t2) and zz==k)
    
    
    for k in range(1, config_data.n_ports+1):
     for t1 in range(1, len(config_data.cost)  ):
        model.addConstr(y[k,t1]>= sum(lambda_var[k,t1,t] for t in range(1,len(config_data.cost)+1) if (t>t1 and (k,t1,t) in lambda_var.keys()  ) ))
    
    for k in range(1, config_data.n_ports+1):
     for t2 in range(1, len(config_data.cost)):
        model.addConstr(y[k,t2]>= sum(lambda_var[k,t,t2] for t in range(0, len(config_data.cost)) if (t<t2 and (k,t,t2) in lambda_var.keys())      ))
    
    for t in range(1, len(config_data.cost)):
        model.addConstr(sum(y[k,t]  for k in range(1, config_data.n_ports+1)   ) <= config_data.gamma  )
    
    
    
    model.setParam('TimeLimit',100)
    model.setParam('MIPGap', 0.009)
    if len(y_start)!=0:
      for ll in y.keys():  
        y[k,t].start= y_start[k,t]
    model.optimize()
    y_opt ={(k,t): y[k,t].x for (k,t) in y.keys()}
    lambda_var_opt = {ll: lambda_var[ll].x for ll in lambda_var.keys()}
    s_opt = extract_s(lambda_var_opt, job_time_block, c_bar)
    
    if model.Status == GRB.Status.OPTIMAL or model.MIPGap < 0.01:
        optimal_val = model.ObjVal
    else:
        optimal_val = model.ObjBound
    end_time = time.time()
    elapsed_time = end_time-start_time
    return optimal_val, s_opt, y_opt, elapsed_time






