import random
import numpy as np
import math

def distribute_B_based_on_p(B, p_new, ell, proportion):
    """
    Distributes elements from B into ell vectors based on given proportions and sorted according to p.

    Args:
    B (list): The list to distribute.
    p (list): Corresponding values for sorting.
    ell (int): Number of vectors to distribute into.
    proportions (list): List of proportions for each vector.

    Returns:
    numpy.array: A numpy array where each row corresponds to a distribution for a specific ell.
    """
    p= p_new.values()
    proportions = np.array(proportion)*100
    if len(B) != len(p):
        raise ValueError("Length of B and p must be the same")

    # Combine B and p and sort based on p
    combined = sorted(zip(B, p), key=lambda x: x[1])

    # Separate B after sorting
    B_sorted = [item[0] for item in combined]

    # Rest of the function remains the same as before
    if sum(proportions) != 100:
        raise ValueError("Proportions must sum to 100")

    distribution_counts = [int(len(B) * (prop / 100)) for prop in proportions]
    distributions = []
    start = 0
    for count in distribution_counts:
        end = start + count
        distributions.append(B_sorted[start:end])
        start = end

    remaining_elements = B_sorted[end:]
    for i, element in enumerate(remaining_elements):
        distributions[i % ell].append(element)

    distribution_array = np.array(distributions, dtype=object)

    return distribution_array



def generate_electricity_prices(Total_hours,uniform_duration):
    prices = []
    for hour in range(Total_hours):
        for minute in range(0, 60, uniform_duration):
            if 7 <= hour < 9 or 18 <= hour < 20:  # Peak hours
                price = random.uniform(20, 30)
            else:
                price = random.uniform(5, 15)
            prices.append(price)
    return prices


def create_mixed_distribution_vector(length=2400):
    """
    Distributes the batteries among 3 time windows
    
    """

    
    dist_params = [
            {'percent': 0.4, 'mean': 8, 'std': .5},
            {'percent': 0.3, 'mean': 6, 'std': .5},
            {'percent': 0.3, 'mean': 7, 'std': .5}
        ]
    
    # Initialize the vector
    vector = np.zeros(length)
    
    # Calculate the number of elements from each distribution
    counts = {i: int(params['percent'] * length) for i, params in enumerate(dist_params)}
    
    # Adjust the last count in case of rounding issues
    counts[len(dist_params) - 1] = length - sum(counts.values())
    
    # Fill the vector with data from each distribution
    start = 0
    for i, params in enumerate(dist_params):
        count = counts[i]
        vector[start:start+count] = np.round(np.random.normal(params['mean'], params['std'], count),2)
        start += count
    
    # Shuffle the vector to ensure the values from different distributions are mixed
    np.random.shuffle(vector)
    int_vector =[max(1, int(num)) for num in vector]
    return int_vector


# feasibilty check

def check_constraints_1(x_test, config_data):
    
    for j in config_data.B_ell[-1]:
        if sum(x_test[j,k,t] for k in range(1,config_data.n_ports+1) for t in range(1, len(config_data.cost)+1) if (j,k,t) in x_test.keys())!=math.ceil(config_data.alpha*config_data.p_dict[j]):
            print(j)
        else:
            print('---------------------------')
            print(sum(x_test[j,k,t] for k in range(1,config_data.n_ports+1) for t in range(1, len(config_data.cost)+1) if (j,k,t) in x_test.keys()))
            print('battery={}'.format(j))
            print(math.ceil(config_data.alpha*config_data.p_dict[j]))
            print('-----------------------------')
            

def check_constraints_2(x_test, config_data):
  for l in range(config_data.ell-1):  
    for j in config_data.B_ell[l]:
        if sum(x_test[j,k,t] for k in range(1,config_data.n_ports+1) for t in range(1, len(config_data.cost)+1) if ( (j,k,t) in x_test.keys() and t <= config_data.demand_window[l]))!=math.ceil(config_data.p_dict[j]):
            print(j)
        else:
            print('---------------------------')
            print(sum(x_test[j,k,t] for k in range(1,config_data.n_ports+1) for t in range(1, len(config_data.cost)+1) if ( (j,k,t) in x_test.keys() and t <= config_data.demand_window[l])))
            print('battery={}'.format(j))
            print(math.ceil(config_data.p_dict[j]))
            print('-----------------------------')
            
def check_constraint_3(x_test, config_data):
    for k in range(1, config_data.n_ports+1):
        for t in range(1, len(config_data.cost)  +1 ):
           if sum(x_test[j,k,t] for j in range(1,config_data.batteries+1) if (j,k,t) in x_test.keys()) > 1:
               print(k,t); break
           else:
               print('-----------------------------------')
               print(sum(x_test[j,k,t] for j in range(1,config_data.batteries+1) if (j,k,t) in x_test.keys()))
               print('------------------------------------------')
               

def check_constraint_4a(x_test, config_data):
    for l in range(config_data.ell-1):
        for j in config_data.B_ell[l]:
            for t in range(1, config_data.demand_window[l]+1):
               if sum(x_test[j,k,t] for k in range(1, config_data.n_ports+1) if ((j,k,t) in x_test.keys() and t <= config_data.demand_window[l])) >1:
                   print(j,t); break
               else:
                   print('-----------------------')
                   print(sum(x_test[j,k,t] for k in range(1, config_data.n_ports+1) if ((j,k,t) in x_test.keys() and t <= config_data.demand_window[l])))
                   print('-----------------------')