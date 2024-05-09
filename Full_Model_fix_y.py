# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:41:36 2024

@author: sunney
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math 
#import networkx as nx
from ortools.graph.python import min_cost_flow
import pandas as pd
import time

def _create_time_blocks(y, T):
    Time_block = {}
    
    # Identify unique 'k' values
    unique_ks = set(k for k, t in y.keys())
    
    # Process each 'k'
    for k in unique_ks:
        # Find all 't' for which y[k, t] = 1
        active_ts = sorted(t for k2, t in y.keys() if k2 == k and y[k2, t] == 1)
        
        # Initialize the time blocks list, starting with 0
        time_blocks = [(0,)]  # Start time of the first block
        
        # Create blocks based on active time periods
        for t in active_ts:
            # If there is a gap between blocks, close the last block and start a new one
            if time_blocks[-1][-1] != t:  # There's a gap between the current end and new start
                time_blocks[-1] = (time_blocks[-1][0], t)  # Close the last block
                time_blocks.append((t,))  # Start a new block
                
        # Finalize the last block, default or based on last active time
        if len(time_blocks[-1]) == 1:  # If the last block is not closed
            time_blocks[-1] = (time_blocks[-1][0], T)  # T + 1 assuming T is the last time period
        else:
            # If there were no active periods, set default block
            if len(active_ts) == 0:
                time_blocks = [(0, T )]  # Default block if no active periods
        
        # Assign the computed time blocks to the current 'k'
        Time_block[k] = time_blocks
    
    return Time_block


def _calculate_cost_for_time_blocks(config_data, T_blocks):
    """
    Calculate the cost for each time block (k,t1, t2) such that for each time block,

    :param config_data: Object containing input data
    :param T_blocks: Dictionary with keys k \in N and eack k is a list of time blocks of two-tuples
    Output: cost of each time block for each k
    """
    
    cost_time_block = {}
    for k in T_blocks.keys():  
        for (t1, t2) in T_blocks[k]:
             cost_time_block[k, (t1, t2)] = sum(config_data.cost[t-1] for t in range(t1 + 1, t2 + 1) )
    return cost_time_block





def create_reference_matrix_and_count_switches(x_jkt):
    # Determine the size of the matrix
    max_k = max(key[1] for key in x_jkt)
    max_t = max(key[2] for key in x_jkt)
    
    # Initialize the matrix of zeros
    matrix = np.zeros((max_k, max_t), dtype=int)
    
    # Fill in the matrix
    for (j, k, t), value in x_jkt.items():
        if value == 1:
            matrix[k-1, t-1] = j  # Adjust for 0-based indexing
    
    # Initialize dictionaries for switching counts and switching occurrences
    switching_counts = {}
    switching_occurrences = {}

    # Count switchings and record occurrences
    for k in range(max_k):
        for t in range(max_t - 1):  # Exclude the last column for switching count
            # Count switching
            if matrix[k, t] != matrix[k, t + 1]:
                switching_counts[t + 1] = switching_counts.get(t + 1, 0) + 1
                switching_occurrences[(k + 1, t + 1)] = 1  # Record switching occurrence as 1
            else:
                switching_occurrences[(k + 1, t + 1)] = 0  # Record no switching as 0
    
    return matrix, switching_counts, switching_occurrences



def Full_Model_x_y(config_data, c_bar,  coeff, y_copy):
    
    if y_copy is None:
        return None, None, None
    tt=time.time()
    model = gp.Model("battery_charging_scheduling")
    N= range(1, config_data.n_ports+1); T= range(1, len(config_data.cost)+1 ); B= range(1, config_data.batteries+1 )
    # Add decision variables
    x = model.addVars(gp.tuplelist( (j,k,t) for (j,k,t) in c_bar.keys()) , vtype=GRB.BINARY, name="x")
    y = model.addVars( gp.tuplelist( (k,t) for k in N for t in T if t!= len(config_data.cost) )   , vtype=GRB.BINARY,name="y")
    
    #------------------------xxxxxxxxxxx
    # Objective: Minimize total electricity and switching cost
    model.setObjective(gp.quicksum(config_data.cost[t-1] * x[j, k, t] for (j,k,t) in x.keys() ) + coeff*gp.quicksum(y[ k, t] for  (k, t) in y.keys()), GRB.MINIMIZE)
    
    #model.setObjective(gp.quicksum((.5*config_data.cost[t-1]+mu[j,k,t]) *x[j, k, t] for (j,k,t) in x.keys()) , GRB.MINIMIZE)
    # Processing time requirement
    model.addConstrs((gp.quicksum( x[j, k, t] for k in N for t in T if t <= config_data.demand_window[l]   if (j,k,t) in x.keys()     ) >= config_data.p_dict[j] for l in range(config_data.ell-1) for j in config_data.B_ell[l]  ), "proc_time")
    model.addConstrs((gp.quicksum( x[j, k, t] for k in N for t in T   if (j,k,t) in x.keys()  ) >= config_data.alpha*config_data.p_dict[j]  for j in config_data.B_ell[config_data.ell-1]  ), "proc_time_stock")

    # Battery capacity
    model.addConstrs((gp.quicksum(x[j, k, t] for j in B  if (j,k,t) in x.keys()    ) <= 1 for k in N for t in T), "cap")

    # Battery assignment
    model.addConstrs((gp.quicksum(x[j, k, t] for k in N if (j,k,t) in x.keys()     ) <= 1 for j in config_data.B_ell[config_data.ell-1] for t in T), "job_assign")
    model.addConstrs((gp.quicksum(x[j, k, t] for k in N  if (j,k,t) in x.keys()     ) <= 1 for l in range(config_data.ell-1) for j in config_data.B_ell[l] for t in T if  t <= config_data.demand_window[l]), "job_assign")
    

    # Reallocation variable constraint
    model.addConstrs(y[k,t] >= gp.quicksum(x[j,k,t+1] for j in B if (j,k,t+1) in x.keys() )- gp.quicksum(x[j,k,t] for j in B if (j,k,t) in x.keys()    ) for k in N for t in T if t!= len(T))
    #model.addConstrs(y[k,t] >= (x[j,k,t+1]  if (j,k,t+1) in x.keys() else 0    )    - (x[j,k,t] if (j,k,t) in x.keys() else 0 )  for j in B for k in N for t in T if t!= len(T) )

    model.addConstrs(y[k,t] >= - (x[j,k,t+1]  if (j,k,t+1) in x.keys() else 0    )    + (x[j,k,t] if (j,k,t) in x.keys() else 0 )  for j in B for k in N for t in T if t!= len(T) )
    
    # Add constraints to limit number of switching
    model.addConstrs(gp.quicksum(y[k,t] for k in N if (k,t) in y.keys()) <= config_data.gamma for t in T)
    
    # for ll in x.keys():
    #   if x_sol_new[ll]>.3:  
    #     model.addConstr(x[ll]==1)
    #   else:
    #       model.addConstr(x[ll]==0)
    for ll in y.keys():
        if y_copy[ll]==1: 
          model.addConstr(y[ll]==1)
        else:
            model.addConstr(y[ll]==0)
    #        #if t not in [2,3,8,17,18,19]: 
    #         model.addConstr(y[ll]==0)
    
    end_time = time.time()-tt
    
    if end_time > 100:
        return None,None, None
    
    model.setParam('TimeLimit',50)
    model.setParam('OutputFlag', 0)
    #model.setParam('MipGap', 0.001)

    model.optimize()
    status = model.status
    
    if status == 2 or model.SolCount >0:
        x_sol ={ll: x[ll].x for ll in x.keys()}
        _,_, y_actual = create_reference_matrix_and_count_switches(x_sol)
        y_opt = y_actual
        objective = sum(config_data.cost[t-1] * x[j, k, t].x for (j,k,t) in x.keys()                                                                                                                                        ) + coeff*sum(y_opt[ k, t] for  (k, t) in y.keys())
        return objective, y_opt, x_sol
    else:
        return None, None,None
    
    

def Full_Model_BinP(config_data, c_bar, coeff, y_copy, elec_best=None):
    """
    Solve a bin-packing problem where items are (k,t1,t2) and each needs to be assigned to at most one of the bin i.e. battery j \in B. Each bin has a so-called capacity of p_j. Although we need an equality here
    
    """
    tt = time.time()
    dd ={t: False if (sum(y_copy[k,t] for k in range(1, config_data.n_ports+1)) >config_data.gamma) else True for t in range(1, len(config_data.cost))}
    if all(dd.values())!=True:
        return None, None, None, None
    T= len(config_data.cost)

    T_blocks= _create_time_blocks(y_copy, T)

    cost_time_blocks = _calculate_cost_for_time_blocks(config_data, T_blocks)
    model = gp.Model("bin-packing problem")
    z= model.addVars(gp.tuplelist(  (k,(t1,t2), j) for (k,(t1,t2)) in cost_time_blocks.keys() for j in range(1, config_data.batteries+1) if (j,k,t2) in c_bar.keys() ), vtype = GRB.BINARY    )
    
    
    
    
    model.setObjective(gp.quicksum(cost_time_blocks[k,(t1,t2)]*z[k,(t1,t2),j] for (k,(t1,t2),j) in z.keys()  ) , GRB.MINIMIZE ) #+ coeff*sum(y_copy[k,t] for (k,t) in y_copy.keys())
    
    
    # if elec_best!= None:
    #     pass
    #     model.addConstr(gp.quicksum(cost_time_blocks[k,(t1,t2)]*z[k,(t1,t2),j] for (k,(t1,t2),j) in z.keys()  ) <= elec_best-.1)
    
    
    # Eack (k,t1,t2) to be assigned to at most one j \in B
    model.addConstrs(gp.quicksum(z[k,(t1,t2),j] for j in range(1, config_data.batteries+1) if (k,(t1,t2),j) in z.keys()) <= 1 for (k,(t1,t2)) in cost_time_blocks.keys() )
    
    # Each j must receive p_j 
    model.addConstrs(gp.quicksum((t2-t1)*z[k,(t1,t2),j] for (k,(t1,t2)) in cost_time_blocks.keys() if (k,(t1,t2),j) in z.keys()) == math.ceil( (config_data.alpha if j in config_data.B_ell[config_data.ell-1] else 1)*config_data.p_dict[j]) for j in range(1, config_data.batteries+1) )
    
    # Each t and j can be assigned to only one k add lazy call backs for time periods with high values
    model.addConstrs(gp.quicksum( z[k,(t1,t2),aa] for (k,(t1,t2),aa) in z.keys() if aa==j and t in range(t1+1,t2+1)  ) <= 1 for l in range(config_data.ell) for j in config_data.B_ell[l] for t in range(1, T+1) if (j,1,t) in c_bar.keys()  ) #

    end_time = time.time()-tt
    
    if end_time > 100:
        return None,None,None, None
    #model.setParam('MipGap', 0.001)
    model.setParam('TimeLimit', 50)
    model.setParam('OutputFlag', 0)

    model.optimize()
    status= model.status
    if status == 2 or model.SolCount >0:
        z_copy = {ll: z[ll].x for ll in z.keys()}
        x_sol_new={}
        for (k,(t1,t2),j) in z.keys():
            for t in range(t1+1, t2+1):
               if (j,k,t) in c_bar.keys(): 
                x_sol_new[j,k,t] = (1 if abs(z_copy[k,(t1,t2),j]-1) <=.01 else 0)
        for (j,k,t) in c_bar.keys():
            if (j,k,t) not in x_sol_new.keys():
                x_sol_new[j,k,t]=0

        _,_, y_actual_new = create_reference_matrix_and_count_switches(x_sol_new)
        elect_cost = sum(config_data.cost[t-1] * x_sol_new[j, k, t] for (j,k,t) in x_sol_new.keys() ) 
        objective = elect_cost + coeff*sum(y_actual_new[ k, t] for  (k, t) in y_actual_new.keys())
        return objective, y_actual_new, elect_cost, x_sol_new
    else:
        return None, None, None, None

    
#- Full model for P_x

def model_Px(config_data, mu,  beta=0.5):
    
    model = gp.Model("battery_charging_scheduling")
    N= range(1, config_data.n_ports+1); T= range(1, len(config_data.cost)+1 ); B= range(1, config_data.batteries+1 )
    # Add decision variables
    x = model.addVars(gp.tuplelist( (j,k,t) for (j,k,t) in mu.keys()) , vtype=GRB.BINARY, name="x")
    
    #------------------------xxxxxxxxxxx
    # Objective: Minimize total electricity and switching cost
    model.setObjective(gp.quicksum((beta*config_data.cost[t-1]+mu[j,k,t]) *x[j, k, t] for (j,k,t) in x.keys()) , GRB.MINIMIZE)
    # Processing time requirement
    model.addConstrs((gp.quicksum( x[j, k, t] for k in N for t in T if t <= config_data.demand_window[l]   if (j,k,t) in x.keys()     ) == config_data.p_dict[j] for l in range(config_data.ell-1) for j in config_data.B_ell[l]  ), "proc_time")
    model.addConstrs((gp.quicksum( x[j, k, t] for k in N for t in T   if (j,k,t) in x.keys()  ) == math.ceil(config_data.alpha*config_data.p_dict[j])  for j in config_data.B_ell[config_data.ell-1]  ), "proc_time_stock")

    # Battery capacity
    model.addConstrs((gp.quicksum(x[j, k, t] for j in B  if (j,k,t) in x.keys()    ) <= 1 for k in N for t in T), "cap")

    # Battery assignment
    model.addConstrs((gp.quicksum(x[j, k, t] for k in N if (j,k,t) in x.keys()     ) <= 1 for j in config_data.B_ell[config_data.ell-1] for t in T), "job_assign")
    model.addConstrs((gp.quicksum(x[j, k, t] for k in N  if (j,k,t) in x.keys()     ) <= 1 for l in range(config_data.ell-1) for j in config_data.B_ell[l] for t in T if  t <= config_data.demand_window[l]), "job_assign")
    
    model.setParam('TimeLimit',50)
    model.setParam('OutputFlag', 0)
    
    # for (j,k,t) in xx.keys():
    #     model.addConstr( x[j,k,t]==xx[j,k,t])
    
    model.optimize()
    status = model.status
    
    if status == 2 or model.SolCount >0:
        x_copy = {ll: x[ll].x for ll in x.keys()}
        objective = model.ObjVal
        return x_copy,objective, 'Optimal'
    else:
        return None, None, 'Infeasible'



class MinCostFlowNetwork_ORTools:
    def __init__(self, config_data, initial_mu, beta=0.5):
        self.beta = beta
        self.config_data = config_data
        self.N = range(1, config_data.n_ports + 1)
        self.T = range(1, len(config_data.cost) + 1)
        self.B = range(1, config_data.batteries + 1)
        self.mu = initial_mu  # Initialize mu values
        self.node_ids = {}  # Maps (node_type, j, k, t) to OR-Tools node IDs
        column_names = ['node_id', 'node_type', 'j', 'k', 't']
        self.inv_map=pd.DataFrame(columns=column_names)
        self.start_nodes, self.supplies = self._start_nodes()
        self.starting_nodes, self.ending_nodes, self.capacities, self.unit_costs, self.marked_indices= self._end_nodes()

    def _start_nodes(self):
        # Create a unique node identifier based on node type and indices
        node_counter=0
        start_nodes=np.array([])
        supplies = np.array([])
        # source nodes
        for j in self.config_data.p_dict.keys():
            start_nodes = np.append(start_nodes,node_counter)
            self.node_ids[node_counter] = ('source', j,None, None)
            self.inv_map = self.inv_map._append({'node_id': node_counter,'node_type': 'source','j':j,'k':None,'t':None}, ignore_index = True)
            supplies = np.append(supplies,math.ceil(self.config_data.p_dict[j] * (self.config_data.alpha if j in self.config_data.B_ell[self.config_data.ell-1] else 1)) )
            node_counter+=1
            
        # j-t nodes
        for j in self.config_data.p_dict.keys():
            for t in self.T:
               if (j,1,t) in self.mu.keys(): 
                start_nodes = np.append(start_nodes, node_counter)
                self.node_ids[node_counter] = ('j-t', j,None,t)
                self.inv_map = self.inv_map._append({'node_id':  node_counter,'node_type':'j-t', 'j'  : j,'k': None, 't':  t}, ignore_index = True)
                supplies = np.append(supplies,0)
                node_counter+=1
        # k-t nodes
        for k in self.N:
            for t in self.T:
                start_nodes = np.append(start_nodes, node_counter)
                self.node_ids[node_counter] = ('k-t', None,k,t)
                self.inv_map = self.inv_map._append({ 'node_id': node_counter, 'node_type':  'k-t', 'j': None, 'k':  k, 't': t}, ignore_index=True)
                supplies = np.append(supplies,0)

                node_counter+=1

        # sink node
        start_nodes = np.append(start_nodes, node_counter)
        self.node_ids[node_counter] =('sink',None,None,None)
        self.inv_map = self.inv_map._append({'node_id': node_counter,'node_type':'sink', 'j':  None,'k':  None,'t': None}, ignore_index = True)
        supplies = np.append(supplies,-sum( ( math.ceil(self.config_data.p_dict[j]*self.config_data.alpha) if j in self.config_data.B_ell[self.config_data.ell-1] else self.config_data.p_dict[j]  )for j in self.config_data.p_dict.keys()))

        node_counter+=1
        return start_nodes, supplies
        
    def _end_nodes(self):
        starting_nodes = np.array([])
        ending_nodes = np.array([])
        capacities = np.array([])
        unit_costs = np.array([])
        marked_indices ={} # index of starting_nodes and end_nodes where a specific (j,k,t) exists
        for start in self.start_nodes:
            
            # source -> j-t nodes
            if self.node_ids[start][0]=='source':
                j_curr= self.node_ids[start][1]
                filtered_map = self.inv_map[ (self.inv_map['node_type']== 'j-t') & (self.inv_map['j']== j_curr) ]
                for _,row in filtered_map.iterrows():
                    starting_nodes = np.append(starting_nodes,start)
                    ending_nodes = np.append(ending_nodes,row['node_id'])
                    capacities = np.append(capacities, 1)
                    unit_costs = np.append(unit_costs,0)
            
            # j-t nodes -> k-t nodes        
            if self.node_ids[start][0]=='j-t':
                t_curr= self.node_ids[start][3]
                filtered_map = self.inv_map[ (self.inv_map['node_type']== 'k-t') & (self.inv_map['t']== t_curr) ]
                for _,row in filtered_map.iterrows():
                    starting_nodes = np.append(starting_nodes,start)
                    ending_nodes=np.append(ending_nodes,row['node_id'])
                    capacities = np.append(capacities, 1)
                    unit_costs = np.append(unit_costs,0)
                    marked_indices[self.node_ids[start][1], row['k'], t_curr] = len(unit_costs)-1
            
            #  k-t nodes -> sink
            if self.node_ids[start][0]=='k-t':
                  starting_nodes = np.append(starting_nodes,start)
                  ending_nodes=np.append(ending_nodes,len(self.node_ids)-1)
                  capacities = np.append(capacities, 1)
                  unit_costs = np.append(unit_costs,0)
          
        return starting_nodes, ending_nodes,capacities, unit_costs, marked_indices

    def update_costs(self, new_mu):
            
             for (j,k,t) in self.marked_indices.keys():
                 self.unit_costs[self.marked_indices[j,k,t]] = 1000*((self.beta * self.config_data.cost[t - 1]) + new_mu[j,k,t])


    def solve_problem(self, new_mu):
        
        smcf = min_cost_flow.SimpleMinCostFlow()
        self.update_costs(new_mu)
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost( self.starting_nodes, self.ending_nodes, self.capacities, self.unit_costs )
        smcf.set_nodes_supplies(np.arange(0, len(self.supplies)), self.supplies)
        status = smcf.solve()
        
        if status != smcf.OPTIMAL:
           print("There was an issue with the min cost flow input.")
           print(f"Status: {status}")
           exit(1)
           return None, None,'Infeasible'
           
        else:
            opt_cost = smcf.optimal_cost()/1000
            print(f"Minimum cost: {opt_cost}")
            print("")
            print(" Arc    Flow / Capacity Cost")
            solution_flows = smcf.flows(all_arcs)
            costs = solution_flows * self.unit_costs
            x_sol={}
            for arc, flow, cost in zip(all_arcs, solution_flows, costs):
                if self.node_ids[smcf.tail(arc)][0]=='j-t':
                    x_sol[self.node_ids[smcf.tail(arc)][1], self.node_ids[smcf.head(arc)][2], self.node_ids[smcf.tail(arc)][3]] = flow
        obj = sum((self.beta*self.config_data.cost[t-1]+new_mu[j,k,t]) *x_sol[j, k, t] for (j,k,t) in x_sol.keys())        
        return x_sol, obj, 'Optimal'


class MinCostFlowNetwork_ORTools_News:
    def __init__(self, config_data, initial_mu, beta=0.5):
        self.beta = beta
        self.config_data = config_data
        self.N = range(1, config_data.n_ports + 1)
        self.T = range(1, len(config_data.cost) + 1)
        self.B = range(1, config_data.batteries + 1)
        self.mu = initial_mu  # Initialize mu values
        self.node_ids = {}  # Maps (node_type, j, k, t) to OR-Tools node IDs
        column_names = ['node_id', 'node_type', 'j', 'k', 't']
        self.inv_map={}
        for name in column_names:
            self.inv_map[name] = []
        self.start_nodes, self.supplies = self._start_nodes()
        self.starting_nodes, self.ending_nodes, self.capacities, self.unit_costs, self.marked_indices= self._end_nodes()

    def _start_nodes(self):
        # Create a unique node identifier based on node type and indices
        node_counter=0
        start_nodes=np.array([])
        supplies = np.array([])
        # source nodes
        for j in self.config_data.p_dict.keys():
            start_nodes = np.append(start_nodes,node_counter)
            self.node_ids[node_counter] = ('source', j,None, None)
            
            self.inv_map['node_id'].append(node_counter)
            self.inv_map['node_type'].append('source')
            self.inv_map['j'].append(j)
            self.inv_map['k'].append(None)
            self.inv_map['t'].append(None)
            supplies = np.append(supplies,math.ceil(self.config_data.p_dict[j] * (self.config_data.alpha if j in self.config_data.B_ell[self.config_data.ell-1] else 1)) )
            node_counter+=1
            
        # j-t nodes
        for j in self.config_data.p_dict.keys():
            for t in self.T:
               if (j,1,t) in self.mu.keys(): 
                start_nodes = np.append(start_nodes, node_counter)
                self.node_ids[node_counter] = ('j-t', j,None,t)
                
                self.inv_map['node_id'].append(node_counter)
                self.inv_map['node_type'].append('j-t')
                self.inv_map['j'].append(j)
                self.inv_map['k'].append(None)
                self.inv_map['t'].append(t)
                supplies = np.append(supplies,0)
                node_counter+=1
        
        # k-t nodes
        for k in self.N:
            for t in self.T:
                start_nodes = np.append(start_nodes, node_counter)
                self.node_ids[node_counter] = ('k-t', None,k,t)
                
                self.inv_map['node_id'].append(node_counter)
                self.inv_map['node_type'].append('k-t')
                self.inv_map['j'].append(None)
                self.inv_map['k'].append(k)
                self.inv_map['t'].append(t)
                supplies = np.append(supplies,0)

                node_counter+=1

        # sink node
        start_nodes = np.append(start_nodes, node_counter)
        self.node_ids[node_counter] =('sink',None,None,None)
        
        self.inv_map['node_id'].append(node_counter)
        self.inv_map['node_type'].append('sink')
        self.inv_map['j'].append(None)
        self.inv_map['k'].append(None)
        self.inv_map['t'].append(None)

        
        supplies = np.append(supplies,-sum( ( math.ceil(self.config_data.p_dict[j]*self.config_data.alpha) if j in self.config_data.B_ell[self.config_data.ell-1] else self.config_data.p_dict[j]  )for j in self.config_data.p_dict.keys()))

        node_counter+=1
        return start_nodes, supplies
        
    def _end_nodes(self):
        starting_nodes = np.array([])
        ending_nodes = np.array([])
        capacities = np.array([])
        unit_costs = np.array([])
        marked_indices ={} # index of starting_nodes and end_nodes where a specific (j,k,t) exists
        for start in self.start_nodes:
            
            # source -> j-t nodes
            if self.node_ids[start][0]=='source':
                j_curr= self.node_ids[start][1]
                
                indices = [index for index, (type, value) in enumerate(zip(self.inv_map['node_type'], self.inv_map['j'])) if type == 'j-t' and value == j_curr]

                for idx in indices:
                    starting_nodes = np.append(starting_nodes,start)
                    ending_nodes = np.append(ending_nodes,self.inv_map['node_id'][idx])
                    capacities = np.append(capacities,1)
                    unit_costs = np.append(unit_costs,0)

            
            # j-t nodes -> k-t nodes
                       
            if self.node_ids[start][0]=='j-t':
                t_curr= self.node_ids[start][3]
                
                mask = (np.array(self.inv_map['node_type']) == 'k-t') & (np.array(self.inv_map['t']) == t_curr)

                # Extract 'node_id' values at the filtered indices
                selected_node_ids = np.array(self.inv_map['node_id'])[mask]
                
                # Update 'starting_nodes' and 'ending_nodes' by repeating 'start' and stacking selected 'node_ids'
                starting_nodes = np.append(starting_nodes, np.full(selected_node_ids.shape, start))
                ending_nodes = np.append(ending_nodes, selected_node_ids)
                
                # Update 'capacities' and 'unit_costs' by appending ones and zeros, respectively
                new_entries_count = selected_node_ids.size
                capacities = np.append(capacities, np.ones(new_entries_count))
                unit_costs = np.append(unit_costs, np.zeros(new_entries_count))
                
                # Update 'marked_indices' for the filtered positions
                # This step might still require iteration if 'self.node_ids' and 'self.inv_map' relations are complex,
                # but if possible, we'll perform a vectorized operation.
                # Assuming 'self.node_ids[start][1]' is a scalar and can be used directly:
                
                if new_entries_count > 0:  # Check if there are any new entries to avoid shape mismatch
                    # This line assumes your marked_indices is a 3D numpy array and self.node_ids[start][1] is applicable for all
                    k_values = np.array(self.inv_map['k'])[mask]
                    j_values = np.full(new_entries_count, self.node_ids[start][1])  # Make it an array of repeated values
                    indices_to_mark = (j_values, k_values, np.full(new_entries_count, t_curr))  # Now all elements are iterables

                    #indices_to_mark = (self.node_ids[start][1], k_values, np.full(new_entries_count, t_curr))
                    # If 'marked_indices' is a NumPy array, we could vectorize this, otherwise, individual assignment might be needed
                    for idx, (j, k, t) in enumerate(zip(*indices_to_mark)):
                        marked_indices[(j, k, t)] = len(unit_costs) - new_entries_count + idx
            
            #  k-t nodes -> sink
            if self.node_ids[start][0]=='k-t':
                  starting_nodes = np.append(starting_nodes,start)
                  ending_nodes=np.append(ending_nodes,len(self.node_ids)-1)
                  capacities = np.append(capacities, 1)
                  unit_costs = np.append(unit_costs,0)
          
        return starting_nodes, ending_nodes,capacities, unit_costs, marked_indices

    def update_costs(self, new_mu):
            
             for (j,k,t) in self.marked_indices.keys():
                 self.unit_costs[self.marked_indices[j,k,t]] = 1000*((self.beta * self.config_data.cost[t - 1]) + new_mu[j,k,t])


    def solve_problem(self, new_mu):
        
        smcf = min_cost_flow.SimpleMinCostFlow()
        self.update_costs(new_mu)
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost( self.starting_nodes, self.ending_nodes, self.capacities, self.unit_costs )
        smcf.set_nodes_supplies(np.arange(0, len(self.supplies)), self.supplies)
        status = smcf.solve()
        
        if status != smcf.OPTIMAL:
           print("There was an issue with the min cost flow input.")
           print(f"Status: {status}")
           exit(1)
           return None, None,'Infeasible'
           
        else:
            opt_cost = smcf.optimal_cost()/1000
            print(f"Minimum cost: {opt_cost}")
            print("")
            print(" Arc    Flow / Capacity Cost")
            solution_flows = smcf.flows(all_arcs)
            costs = solution_flows * self.unit_costs
            x_sol={}
            for arc, flow, cost in zip(all_arcs, solution_flows, costs):
                if self.node_ids[smcf.tail(arc)][0]=='j-t':
                    x_sol[self.node_ids[smcf.tail(arc)][1], self.node_ids[smcf.head(arc)][2], self.node_ids[smcf.tail(arc)][3]] = flow
        obj = sum((self.beta*self.config_data.cost[t-1]+new_mu[j,k,t]) *x_sol[j, k, t] for (j,k,t) in x_sol.keys())        
        return x_sol, obj, 'Optimal'


