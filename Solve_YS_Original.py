# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:53:40 2024

@author: sunney
"""
import gurobipy as gp
from gurobipy import GRB
import time
def solveYS_original(config_data,c_bar):
    start_time = time.time()

    model_ys = gp.Model("Pys")
        # Add decision variables
    s = model_ys.addVars(gp.tuplelist((j,k,t) for (j,k,t) in c_bar.keys()), vtype=GRB.BINARY)
    y = model_ys.addVars(gp.tuplelist((k,t) for k in range(1, config_data.n_ports+1)  for t in range(1, len(config_data.cost)))  , vtype=GRB.BINARY, name="y")
        # Objective: Minimize total electricity cost
          #self.coeff = (min(self.config_data.cost)/((self.config_data.gamma*(len(self.config_data.cost)-1))+1))
    coeff = 0.02
    
    model_ys.setObjective(gp.quicksum( c_bar[j,k,t]*s[j,k,t] for (j,k,t) in s.keys()) + coeff*gp.quicksum(y[k,t] for (k,t) in y.keys()))
    model_ys.addConstrs(y[k,t] >=  gp.quicksum( (s[j,k,t+1] if (j,k,t+1) in s.keys() else 0) for j in range(1, config_data.batteries+1) if ((j,k,t) in s.keys()) ) - gp.quicksum(s[j,k,t] for j in range(1, config_data.batteries+1)  if (j,k,t) in s.keys()   ) for k in range(1, config_data.n_ports+1) for t in range(1, len(config_data.cost)))
    model_ys.addConstrs(y[k,t] >= s[j,k,t]  - (s[j,k,t+1] if (j,k,t+1) in s.keys() else 0) for j in range(1, config_data.batteries+1) for k in range(1, config_data.n_ports+1) for t in range(1, len(config_data.cost)) if ((j,k,t) in s.keys() ))
    #6
    model_ys.addConstrs(gp.quicksum(y[k,t] for k in range(1,config_data.n_ports+1)) <= config_data.gamma for t in range(1, len(config_data.cost)) )
    
    model_ys.addConstrs(gp.quicksum(s[j,k,t] for j in range(1, config_data.batteries+1) if (j,k,t) in c_bar.keys()) <= 1 for k in range(1, config_data.n_ports+1) for t in range(1, len(config_data.cost)+1  ))
    
    model_ys.setParam('TimeLimit',100)
    model_ys.setParam('MIPGap', 0.009) 
    model_ys.optimize()
    
    y_opt ={(k,t): y[k,t].x for (k,t) in y.keys()}
    s_opt = {ll: s[ll].x for ll in s.keys()}
    
    if model_ys.Status == GRB.Status.OPTIMAL or model_ys.MIPGap < 0.01:
        optimal_val = model_ys.ObjVal
    else:
        optimal_val = model_ys.ObjBound
    end_time = time.time()
    elapsed_time = end_time-start_time
    return optimal_val, s_opt, y_opt, elapsed_time

    