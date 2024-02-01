import random, concurrent.futures, scipy.optimize
import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import Counter
from collections.abc import Iterable
import powerlaw

def find_jams(evolution, add_lifespans = False):
    def dfs(i, j, cluster):
        """
        Recursive function to find the size and lifespan of a jam in the evolution of a CA. 
        It updates the cluster dictionary with the size and lifespan of the jam.
        """
        if 0 <= i < len(evolution) and -len(evolution[0]) < j < len(evolution[0]) and evolution[i][j]:
            evolution[i][j] = False
            cluster['size'] += 1
            cluster['lifespan'] = max(cluster['lifespan'], i)
            
            # Explore neighbors
            dfs(i + 1, j, cluster)
            dfs(i - 1, j, cluster)
            dfs(i, j + 1, cluster)
            dfs(i, j - 1, cluster)
    
    """
    Function to find jams in the evolution of a CA. It returns a list of the sizes of all the jams found in the evolution of the CA.
    If add_lifespans is True, it also returns a list of the lifespans of all the jams found in the evolution of the CA.
    
    Parameters:
    - evolution (list): The evolution of the CA.
    - add_lifespans (bool): Whether or not to return the lifespans of the jams.
    
    Returns:
    - jams['size'] (list): A list of the sizes of all the jams found in the evolution of the CA.
    - jams['lifespan'] (list): A list of the lifespans of all the jams found in the evolution of the CA. Only returned if add_lifespans is True.
    """

    evolution = [row[:] for row in evolution]  # Make a copy of the evolution, so we don't change the original matrix
    
    jams = {'size': [], 'lifespan': []}

    for i in range(len(evolution)):
        for j in range(len(evolution[0])):
            if evolution[i][j]:
                jam = {'size': 0, 'lifespan': 0}
                dfs(i, j, jam)
                if jam['size'] > 1:
                    jam['lifespan'] = max(1, jam['lifespan'] - i + 1)
                    jams['size'].append(jam['size'])
                    jams['lifespan'].append(jam['lifespan'])

    if add_lifespans:
        return jams['size'], jams['lifespan']
    return jams['size']

def test_new_jam_finder():
    """
    Function to test the new jam finder function. It prints the sizes and lifespans of the jams found in the evolution of the CA.
    This specific example is chosen to include a jam that flows through the border of the CA.
    """
    # Example usage:
    evolution = [
        [False, False, True, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, False, True]
    ]

    jam_sizes, jam_lifespans = find_jams(evolution, add_lifespans=True)

    print("Cluster Sizes:", jam_sizes)
    print("Cluster Lifespans:", jam_lifespans)
    
    # plot the car evolution
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size as needed
    cpl.plot(np.array(evolution), colormap='GnBu')
    
    # Example usage:
    evolution = [
        [False, True, False, True, False],
        [True, True, False, True, False],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [False, False, False, True, True]
    ]

    jam_sizes, jam_lifespans = find_jams(evolution, add_lifespans=True)

    print("Cluster Sizes:", jam_sizes)
    print("Cluster Lifespans:", jam_lifespans)
    
    # plot the car evolution
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the size as needed
    cpl.plot(np.array(evolution), colormap='GnBu')

    return None

def triangulize_evolution(evolution):
    """
    Function to remove the cells that are not part of the triangle in the evolution of a CA. This is done by setting the value of the cell to 0 if it is not part of the triangle.
    This ensures that all cars can reach all jams.
    """
    L = len(evolution[0])
    for y in range(len(evolution)):
        for x in range(len(evolution[y])):
            if y > x or y > L-x:
                evolution[y][x] = 0
    return evolution

# def run_model(p, L, T, n_repetitions = 100):
#     """
#     Function to run the CA rule 184 model for a given p, L and T. It returns the lifespans and jam sizes of all the jams found in the evolution of the CA.
    
#     Parameters:
#     - p (float): The probability of a cell being 1 in the initial state.
#     - L (int): The length of the CA.
#     - T (int): The number of timesteps.
#     - n_repetitions (int): The number of times the model should be run.
    
#     Returns:
#     - lifespan_counter (Counter): A counter with the lifespans of all the jams found in the evolutions of the CA.
#     - jam_counter (Counter): A counter with the sizes of all the jams found in the evolutions of the CA.
#     """

#     def gen_initial_state_bernoulli(L, p):
#         """
#         Generate a random initial state for the cellular automaton. Bernoulli distribution.

#         Returns:
#         list: Initial state.
#         """
#         assert p >= 0 and p <= 1, "p must be between 0 and 1"
#         assert L > 0, "L must be greater than 0"

#         initial_state = np.array([[np.random.choice([0, 1], p=[1 - p, p]) for i in range(L)]])
#         return initial_state
    
#     def initial_to_random_walk(initial_state):
#         # Plot the random walk that is the initial state, go up for 1, down for 0
#         L = len(initial_state[0])
#         random_walk = [0] * L
#         for i in range(L):
#             if initial_state[0][i] == 1:
#                 random_walk[i] = random_walk[i-1] + 1
#             else:
#                 random_walk[i] = random_walk[i-1] - 1
#         return random_walk
    
#     def jam_lifespans(initial_random_walk):
#         # Find the lifespans of the jams based on the random walk of the initial state (see initial_to_random_walk). For every value in the random walk, find the next occurrence of that value and calculate the lifespan based on this return time.
        
#         lifespans = []

#         for i in range(len(initial_random_walk) - 1):
#             value = initial_random_walk[i]

#             # If the next value is higher, we can't calculate the lifespan of a jam from here.
#             if value < initial_random_walk[i + 1]:
#                 continue
            
#             # Find the first next occurrence of the value
#             try:
#                 next_occurrence = initial_random_walk[i + 1:].index(value) + i + 1
#                 assert initial_random_walk[next_occurrence] == value, "The next occurrence is not the same as the value we are looking for"
            
#             except ValueError:
#                 continue
            
            
#             lifespan = (next_occurrence - i)/2
            
#             if lifespan > 1:
#                 lifespans.append(lifespan)
                
#         return lifespans
    
#     assert isinstance(p, float) and 0 <= p <= 1, "The given density p should be a float between 0 and 1"
#     assert isinstance(L, int) and L > 0, "The length L should be a positive integer"
#     assert isinstance(T, int) and T > 0, "The time T should be a positive integer"
#     assert isinstance(n_repetitions, int) and n_repetitions > 0, "The amount of times the model is ran n_repitions should be an integer larger than 0"

#     total_lifespans = []
#     total_jam_sizes = []
#     for _ in range(n_repetitions):
#         initial_state = gen_initial_state_bernoulli(L, p)
#         random_walk = initial_to_random_walk(initial_state)
#         cellular_automaton = cpl.evolve(initial_state, timesteps=T, memoize=True, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule=184))
#         cellular_automaton = triangulize_evolution(cellular_automaton)

#         lifespans = jam_lifespans(random_walk)
#         total_lifespans += lifespans
        
#         jam_sizes = find_jams(cellular_automaton)
#         total_jam_sizes += jam_sizes
    
#     lifespan_counter = Counter(total_lifespans)
#     jam_counter = Counter(total_jam_sizes)
#     return lifespan_counter, jam_counter

            

#changed run model function introducing stochasticity/dynamics Influx=Outflux 
def run_model_stochastic(p, L, T, n_repetitions=100, v_max=5, p_slowdown=0.1, triangular=False, return_evolutions=False, 
                         dynamic_model=False, neighbourhood_size=1, entry_chance=0.5, exit_chance=0.5):
    """
    Function to run the NaSch model for a given p, L, T. It returns the lifespans and jam sizes of all the jams found in the evolution of the model.

    Parameters:
    - p (float): The probability of a cell being 1 in the initial state.
    - L (int): The length of the simulation.
    - T (int): The number of timesteps.
    - n_repetitions (int): The number of times the model should be run.
    - v_max (int): Maximum speed of vehicles.
    - p_slowdown (float): Probability of slowing down.
    - triangular (bool): Whether or not to triangulize the evolution.
    - return_evolutions (bool): Whether or not to return the evolution of the model.
    - dynamic_model (bool): Whether or not to use the dynamic model.
    - neighbourhood_size (int): The size of the neighbourhood for the dynamic model. Number of cars in front of cell that influences entry or exit probability.
    - entry_chance (float): The probability of a car entering in a completely empty neighbourhood (scales down with fuller neighbourhoods).
    - exit_chance (float): The probability of a car exiting in a completely full neighbourhood (scales down with emptier neighbourhoods).

    Returns:
    - lifespan_counter (Counter): A counter with the lifespans of all the jams found in the evolutions of the model.
    - jam_counter (Counter): A counter with the sizes of all the jams found in the evolutions of the model.
    - all_evolutions (list): A list containing the evolution of the model for every repetition. Only returned if return_evolutions is True.
    """

    
    def initial_state_nasch(L, p, v_max):
        """
        Function to generate an initial state for the NaSch model. It returns the initial state of the CA.

        Parameters:
        - L (int): The length of the CA.
        - p (float): The probability of a cell being 1 in the initial state.
        - v_max (int): Maximum speed of vehicles.

        Returns:
        - initial_state (list): The initial state of the CA. It contains a tuple for every cell, with the first element indicating whether the cell is occupied and the second element indicating the speed of the vehicle.
        """
        initial_state = [(False, 0) if random.random() > p else (True, random.randint(1, v_max)) for _ in range(L)]
        return initial_state

    def nasch_step(current_state, v_max, p_slowdown, dynamic_model=False, neighbourhood_size=1, entry_chance=0.5, exit_chance=0.5):
        """
        Function to perform a single timestep of the NaSch model. It returns the next state of the CA. Start with the potential dynamics of the model.
        Then perform the acceleration, slowing down, randomization and movement steps.

        Parameters:
        - current_state (list): The current state of the CA. It contains a tuple for every cell, with the first element indicating whether the cell is occupied and the second element indicating the speed of the vehicle.
        - v_max (int): Maximum speed of vehicles.
        - p_slowdown (float): Probability of slowing down.
        - dynamic_model (bool): Whether or not to use the dynamic model.
        - neighbourhood_size (int): The size of the neighbourhood for the dynamic model. Number of cars in front of cell that influences entry or exit probability.
        - entry_chance (float): The probability of a car entering in a completely empty neighbourhood (scales down with fuller neighbourhoods).
        - exit_chance (float): The probability of a car exiting in a completely full neighbourhood (scales down with emptier neighbourhoods).

        Returns:
        - next_state (list): The next state of the CA. Same format as current_state.
        """

        # Start with the potential dynamics of the model (influx and outflux):
        if dynamic_model:
            # Loop over cells from left to right, that way newly changed cells don't influence the next cells
            for i, (car_present, velocity) in enumerate(current_state):
                neighbourhood_density = np.mean([current_state[(i + j) % len(current_state)][0] for j in range(neighbourhood_size)])
                
                # If the cell is empty, there is a chance that a car enters, car has random speed between 1 and v_max
                if not car_present and random.random() < entry_chance * (1 - neighbourhood_density):
                    # print(f"Car appeared at {i}, had probability {entry_chance * (1 - neighbourhood_density)}")
                    current_state[i] = (True, random.randint(1, v_max))
                    
                # If the cell is occupied, there is a chance that a car exits
                elif car_present and random.random() < exit_chance * neighbourhood_density:
                    # print(f"Car disappeared at {i}, had probability {exit_chance * neighbourhood_density}")
                    current_state[i] = (False, 0)

        # Acceleration: Increase the speed of each vehicle by 1, up to the maximum speed
        current_state = [(x[0], min(x[1] + 1, v_max)) for x in current_state]

        # Slowing down: Check the distance to the next vehicle, and reduce the speed to that distance if it is smaller than the current speed
        for i, (car_present, velocity) in enumerate(current_state):
            if not car_present:
                continue

            distance_to_next_car = 0
            for j in range(i + 1, len(current_state) + 1 + v_max):
                # If we reach the end of the road, we check for cars at the beginning of the road
                if j >= len(current_state):
                    # Check for cars at the beginning of the road
                    if current_state[j - len(current_state)][0]:
                        break
                # If we find a car, we stop counting
                elif current_state[j][0]:
                    break
            
                distance_to_next_car += 1
            
            # Reduce the speed to the distance to the next car if it is smaller than the current speed
            current_state[i] = (current_state[i][0], min(distance_to_next_car, current_state[i][1]))
        
        # Randomization: Reduce the speed of each vehicle by 1 with probability p_slowdown
        current_state = [(x[0], max(x[1] - 1, 0) if random.random() < p_slowdown else x[1]) for x in current_state]

        # Movement: Move each vehicle forward by its speed
        next_state = [(False, 0)] * len(current_state)
        for i, (car_present, velocity) in enumerate(current_state):
            # If there is no car in the cell, we don't have to do anything
            if not car_present:
                continue
            
            # If the car is at the end of the road, it re-enters at the beginning
            if i + velocity >= len(current_state):
                next_state[i + velocity - len(current_state)] = (True, velocity)
            else:
                next_state[i + velocity] = (True, velocity)

        return current_state, next_state

    assert isinstance(p, float) and 0 <= p <= 1, "The given density p should be a float between 0 and 1"
    assert isinstance(L, int) and L > 0, "The length L should be a positive integer"
    assert isinstance(T, int) and T > 0, "The time T should be a positive integer"
    assert isinstance(v_max, int) and v_max > 0, "The maximum speed v_max should be an integer larger than 0"
    assert isinstance(p_slowdown, float) and 0 <= p_slowdown <= 1,  "The given slowing down probability p_slowdown should be a float between 0 and 1"
    assert isinstance(neighbourhood_size, int) and neighbourhood_size > 0, "The neighborhood size neighborhood_size should be an integer larger than 0"
    assert isinstance(entry_chance, float) and 0 <= entry_chance <= 1, "The given chance of entry entry_chance should be a float between 0 and 1"
    assert isinstance(exit_chance, float) and 0 <= exit_chance <= 1, "The given chance of entry entry_chance should be a float between 0 and 1"


    total_lifespans = []
    total_jam_sizes = []

    all_evolutions = []

    for i in range(n_repetitions):
        # Create initial state
        initial_state = initial_state_nasch(L, p, v_max)

        # Run the model
        evolution = [initial_state]
        for t in range(T):
            # print()
            # print(f"Timestep {t}")
            current, next = nasch_step(evolution[-1], v_max, p_slowdown, dynamic_model=dynamic_model, 
                                       neighbourhood_size=neighbourhood_size, entry_chance=entry_chance, 
                                       exit_chance=exit_chance)
            evolution[-1] = current
            evolution.append(next)
        
        if return_evolutions:
            all_evolutions.append(evolution)

        location_states = np.array([[cell[0] for cell in state] for state in evolution])

        if triangular:
            location_states = triangulize_evolution(location_states)

        jam_sizes, lifespans = find_jams(location_states, add_lifespans=True)
        
        total_jam_sizes += jam_sizes
        total_lifespans += lifespans

    lifespan_counter = Counter(total_lifespans)
    jam_counter = Counter(total_jam_sizes)

    if return_evolutions:
        return lifespan_counter, jam_counter, all_evolutions

    return lifespan_counter, jam_counter


def calculate_flow_nasch(evolution):
    """
    Function to calculate the total flow of the NaSch model. The flow is defined as the 
    amount of car movement in total. This is calculated by summing the speeds of cars over every timestep.
    """
    total_flow = 0
    
    # Check all timesteps but the last one, because we assume that no movement happens in the last timestep
    for t in range(len(evolution[:-1])):
        for i in range(len(evolution[t])):
            speed = evolution[t][i][1]
            car_present = evolution[t][i][0]
            if car_present:
                total_flow += speed
    
    return total_flow


def calculate_delay_nasch(evolution, v_max):
    """
    Function to calculate the total delay of the NaSch model. The flow is defined as the
    amount of difference between v_max and exact speed in total. This is calculated by summing the total delay of cars.
    """
    total_delay = 0
    car_num=0

    # Check all timesteps but the last one, because we assume that no movement happens in the last timestep
    for t in range(len(evolution[:-1])):
        for i in range(len(evolution[t])):
            speed = evolution[t][i][1]
            car_present = evolution[t][i][0]
            if car_present:
                elem_delay = max(0, v_max - speed)
                car_num += 1
                total_delay += elem_delay

    if car_num==0:
        car_num=1
        mean_delay=total_delay/car_num
        return mean_delay

    else:
        mean_delay = total_delay/car_num


    return mean_delay

def critical_density_wrapper(args):
    """
    Function to run the model with a single argument, so it can be used with concurrent.futures.ProcessPoolExecutor's map function.

    Parameters:
    - args (tuple): A tuple containing the arguments for the run_model_stochastic function. In this order (!): p_values, L, T, n, v_max, p_slowdown
    """
    p_values, L, T, n, v_max, p_slowdown = args

    total_flows_per_density = dict()
    average_flows_per_density = dict()

    for density in p_values:

        _, _, evolutions = run_model_stochastic(density, L, T, n, v_max=v_max, p_slowdown = p_slowdown, return_evolutions=True)
        

        total_flows = [calculate_flow_nasch(evolution) for evolution in evolutions]
        average_flows = [flow / float(T) for flow in total_flows]
        
        total_flows_per_density[density] = total_flows
        average_flows_per_density[density] = average_flows

    # Find critical point: the density for which the total flow average is maximum
    mean_total_flow_per_density = {density: np.mean(flows) for density, flows in total_flows_per_density.items()}
    critical_density = max(mean_total_flow_per_density, key=mean_total_flow_per_density.get)

    return critical_density


def find_critical_dataframe_nasch(p_slowdown_values, v_max_values, p_values, L, T, n, repetitions=1):
    """
    Function to find critical densities for combinations of p_slowdown and v_max values in the NaSch model. 
    It returns a dataframe with the critical densities for every combination of p_slowdown and v_max.

    Parameters:
    - p_slowdown_values (list): The list of p_slowdown values to run the model for.
    - v_max_values (list): The list of v_max values to run the model for.
    - p_values (list): The list of density values to run the model for.
    - L (int): The simulated road length.
    - T (int): The number of timesteps.
    - n (int): The number of repetitions.

    Returns:
    - critical_densities (pandas.DataFrame): A dataframe with the critical densities for every combination of p_slowdown and v_max.
    """
    assert isinstance(p_slowdown_values, Iterable),  "The list of p_slowdown values to run the model for p_values should be a list"
    assert isinstance(v_max_values, Iterable), "The list of v_max values to run the model for p_values should be a list"
    assert isinstance(p_values, Iterable), "The list of density values to run the model for p_values should be a list"
    assert isinstance(L, int) and L > 0, "The length L should be a positive integer"
    assert isinstance(T, int) and T > 0, "The time T should be a positive integer"
    assert isinstance(n, int) and n > 0, "The number of repititions n should be an integer larger than 0"

    # Create a list with inputs for the critical_density_wrapper function
    args = []
    for v_max in v_max_values:
        for p_slowdown in p_slowdown_values:
            args.append((p_values, L, T, n, v_max, p_slowdown))

    # Run the model for every combination of p_slowdown and v_max
    outputs = []
    for i in range(repetitions):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            output = list(tqdm(executor.map(critical_density_wrapper, args), total=len(args), desc=f"Finding critical densities (Repetition {i + 1} of {repetitions})"))
            outputs.append(output)
        
    # Create a dataframe with the results
    v_maxes = []
    p_slowdowns = []
    critical_densities = []

    for i, args in enumerate(args):
        v_maxes.append(args[4])
        p_slowdowns.append(args[5])
        critical_densities.append([output[i] for output in outputs])
    
    # Create DataFrame
    output_df = pd.DataFrame({
        'p_slowdown': p_slowdowns,
        'v_max': v_maxes,
        'critical_density': critical_densities
    })

    return output_df

# def visualize_jam_counter(jam_counter, fit_line = False):
#     plt.figure(figsize=(12,6))
#     plt.title(f'Jam Sizes')
#     plt.loglog(range(1, int(max(jam_counter.keys()))), [jam_counter[i] for i in range(1, int(max(jam_counter.keys())))], 'o')
#     plt.xlabel('Jam size')
#     plt.ylabel('Number of jams')

#     if fit_line:
#         lists = sorted(jam_counter.items())
#         x, y = zip(*lists)
#         x_log, y_log = np.log(x), np.log(y)

#         # Fit linear function to log-log data   
#         def func(x, a, b):
#             return -a * x + b
        
#         def power_law_func(x, a, b):
#             return a * x ** -b

#         popt, _ = scipy.optimize.curve_fit(func, x_log, y_log, p0=[1, 0.5])
#         a_optimal, b_optimal = popt

#         # plot the fitted line
#         plt.plot(x, power_law_func(x, np.exp(b_optimal), a_optimal), label=f'Linear fit on log-log data: y = {np.exp(b_optimal):.2f}x^-{a_optimal:.2f}')

#         # Fit power-law function
#         popt, _ = scipy.optimize.curve_fit(power_law_func, x, y, p0=[1, 0.5])
#         a_optimal, b_optimal = popt

#         # plot the fitted line
#         plt.plot(x, power_law_func(x, a_optimal, b_optimal), label=f'Power-law fit: y = {a_optimal:.2f}x^-{b_optimal:.2f}')

#     plt.legend()
#     plt.show()

def find_density(L, p, n, v_max, p_slowdown, dynamic_model, neighbourhood_size, entry_chance, exit_chance):
    """
    Short function to find how the density evolves over time. 
    
    Parameters:
    - L (int): The length of the simulated road.
    - p (float): The probability of a cell being 1 in the initial state.
    - n (int): The number of repetitions.
    - v_max (int): Maximum speed of vehicles.
    - p_slowdown (float): Probability of slowing down.
    - dynamic_model (bool): Whether or not to use the dynamic model.
    - neighbourhood_size (int): The size of the neighbourhood for the dynamic model. Number of cars in front of cell that influences entry or exit probability.
    - entry_chance (float): The probability of a car entering in a completely empty neighbourhood (scales down with fuller neighbourhoods).
    - exit_chance (float): The probability of a car exiting in a completely full neighbourhood (scales down with emptier neighbourhoods).
    
    Returns:
    - density_evolution (list): List containing the density evolution of the model for each timestep.
    """
    T = int(L)

    #find evolution
    evolutions = run_model_stochastic(p=p, L=L, T=T, n_repetitions=n, v_max=v_max, p_slowdown=p_slowdown, 
                                        return_evolutions=True, dynamic_model=dynamic_model, neighbourhood_size=neighbourhood_size,
                                        entry_chance=entry_chance, exit_chance=exit_chance)[2]



    evolution = evolutions[0]
    location_evolution = [[x[0] for x in line] for line in evolution]                   #List containing the evolution of the model for each timestep.
    density_evolution = [np.sum(line) / len(line) for line in location_evolution]#      #List containing the density evolution of the model for each timestep.
    
    return density_evolution

def analyze_powerlaw_distribution(data):
    """
    Function that analysis the distribution of given data with the powerlaw package.
    Makes use of the Kolmogorov-Smirnov test which returns R-log likelihood (ks_stat) and p-value (ks_p_value). 
    p-value/ks_p_value > 0.05 means that the data is likely to be drawn from the same distribution.
    R/ks_stat <0  means the data is drawn from a exponential distribution
    R/ks_stat >0 means the data is drawn from a power law distribution 


    Input: data (array-like)
    Output: string with distribution type

    """
    #Make data in array format
    data = np.asarray(data)

    #Fit to a power-law distribution
    fit = powerlaw.Fit(data, discrete=True)

    #Goodness of fit using the Kolmogorov-Smirnov test for power-law vs. exponential
    distr1 = "power_law"
    distr2 = "exponential"
    ks_stat, ks_p_value = fit.distribution_compare(distr1, distr2, normalized_ratio=True)
    exponent = fit.power_law.alpha
    

    # Analyze the fit and return the result
    if ks_stat > 0:
        # R log likelihood was positive, so data prefers power-law distribution
        result = f"Data prefers {distr1} over {distr2} (p-value: {ks_p_value:.4f})"

    else:
        # R log likelihood was negative, so data prefers exponential distribution
        result = f"Data prefers {distr2} over {distr1} (p-value: {ks_p_value:.4f})"

    return result, exponent

# A power-law distribution for a given initial density
def analyze_critical_exponent_for_density(p, L, T, n_repetitions, v_max, p_slowdown, triangular, dynamic_model, neighbourhood_size, entry_chance, exit_chance):
    """Function that saves the critical exponent for a initial density."""

    # Nasch model
    lifespan_counter, jam_counter = run_model_stochastic(p, L, T, n_repetitions, v_max, p_slowdown, triangular, return_evolutions=False, dynamic_model=dynamic_model, neighbourhood_size=neighbourhood_size, entry_chance=entry_chance, exit_chance=exit_chance)
    
    # Previous defined analyze_powerlaw_distribution function 
    result, exponent = analyze_powerlaw_distribution(list(jam_counter.elements()))
    
    return p, exponent

def density_evolution_nasch (evolution):
    """
    Function to calculate the density evolution of the NaSch model. The density is defined as the
    amount of cars in total. This is calculated by summing the total amount of cars.
    Input: 
    - evolution (list): List containing the evolution of the model for each timestep.
    Output:
    - density_evolution (list): List containing the density evolution of the model for each timestep.
    """
    location_evolution = [[x[0] for x in line] for line in evolution]
    density_evolution = [np.sum(line) / len(line) for line in location_evolution]
    return density_evolution
