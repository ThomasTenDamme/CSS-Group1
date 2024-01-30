import random, concurrent.futures, scipy.optimize
import numpy as np
import cellpylib as cpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from collections import Counter

class CA:
    """
    Cellular Automaton (CA) class for simulating one-dimensional cellular automata.

    Attributes:
    - L (int): Length of the automaton.
    - r (int): Radius of the neighborhood.
    - k (int): Number of possible cell states.
    - dec_rule (int): Decimal representation of the rule.
    - t_end (int): Number of time steps.
    - s_q (str): Quiescent state for lambda parameter calculation.
    - initial_state (list): Initial state of the automaton.

    Methods:
    - gen_CA(L, r, k, dec_rule, t_end, s_q, initial_state=False):
        Generate the evolution of the cellular automaton together with the value for lambda.

    - calculate_transient_length():
        Calculate the transient length of the cellular automaton.

    Usage:
    # Create an instance of the CA class
    ca_instance = CA()

    # Generate the evolution of the automaton
    result, evolution = ca_instance.gen_CA(L, r, k, dec_rule, t_end, s_q, initial_state=)

    # Calculate the transient length
    transient_length = ca_instance.calculate_transient_length()
    """

    def __init__(self):
        """
        Initialize the Cellular Automaton (CA) instance with default values.
        """
        self.L = 0
        self.r = 0
        self.k = 0
        self.dec_rule = 0
        self.t_end = 0
        self.s_q = "0"
        self.initial_state = False
        self.evolution = None

    def gen_CA(self, L, r, k, dec_rule, t_end, s_q, initial_state=False):
        """
        Generate the evolution of the cellular automaton.

        Returns:
        Tuple: lambda_parameter (float), evolution (numpy.ndarray)
        """
        self.L = L
        self.r = r
        self.k = k
        self.dec_rule = dec_rule
        self.t_end = t_end
        self.s_q = s_q
        self.initial_state = initial_state
        self.evolution = np.zeros((t_end + 1, L))

        if initial_state == False:
            self.initial_state = [(random.randint(0, k - 1)) for i in range(L)]

        if k > 2:
            bin_rule_short = np.base_repr(dec_rule, base=k)[2:]
        else:
            bin_rule_short = bin(dec_rule)[2:]

        num_zeroes = (k ** (2 * r + 1)) - len(bin_rule_short)
        bin_rule = "0" * num_zeroes + bin_rule_short

        old_state = self.initial_state

        for t in range(t_end + 1):
            self.evolution[t] = old_state
            new_state = [0] * L

            for i in range(L):
                previous_values = []
                for j in range(2 * r + 1):
                    previous_values.append(np.take(old_state, i - r + j, mode='wrap'))

                string_previous_values = ''.join(str(x) for x in previous_values)
                rule_previous_values = int(string_previous_values, k)
                new_value = bin_rule[-(rule_previous_values + 1)]

                new_state[i] = int(new_value)

            old_state = new_state
        
        lambda_parameter = 1 - ((k ** (2 * r + 1) - bin_rule_short.count(s_q)) / (k ** (2 * r + 1)))

        return lambda_parameter, self.evolution

    def calculate_transient_length(self):
        """
        Calculate the transient length of the cellular automaton.

        Returns:
        int or str: Transient length or 'no transient found'.
        """
        transient_length = 'no transient found'

        for i in range(1, len(self.evolution)):
            for j in range(i):
                if np.array_equal(self.evolution[i], self.evolution[j]):
                    return i

        return transient_length
    
    def gen_initial_state_bernoulli(self, L, p):
        """
        Generate a random initial state for the cellular automaton. Bernoulli distribution.

        Returns:
        list: Initial state.
        """
        assert p >= 0 and p <= 1, "p must be between 0 and 1"
        assert L > 0, "L must be greater than 0"

        self.L = L
        self.initial_state = np.array([[np.random.choice([0, 1], p=[1 - p, p]) for i in range(self.L)]])
        return self.initial_state


def find_jams(evolution, add_lifespans = False):
    """
    A function to find the jams in the evolution of a CA. It checks every row for groupings of 1's that are bigger than 1.
    Then it checks if it can find the same group in the next row, if so, it adds the length of the group to the size of the jam.
    The resulting list contains the starting and ending index of the last row that contained the jams, as well as the size of the jam.

    Parameters:
    - evolution (list): The evolution of the CA.
    - add_lifespans (bool): Whether or not to add the lifespans of the jams to the result.

    Returns:
    - result (list): A list containing the starting and ending index of the last row that contained the jams, as well as the size of the jam. If add_lifespans is True, it also contains the lifespans of the jams.
    """

    def row_jams(row):
        result = []
        start = None

        for i, value in enumerate(row):
            if value == 1:
                if start is None:
                    start = i
            elif value == 0 and start is not None:
                if i - start >= 2:  # Check if the group has at least two 1's
                    result.append([start, i - 1])
                start = None

        # Check if the last group extends to the end of the list
        if start is not None and len(row) - start >= 2:
            result.append([start, len(row) - 1])

        return result

    previous_jams = [[x, x[1] - x[0] + 1] for x in row_jams(evolution[0])]
    lifespans = [1] * len(previous_jams)

    for row in evolution[1:]:
        current_jams = row_jams(row)

        # The rightmost cell of every previous jam should now be one cell to the left in the current jam
        for i, jam in enumerate(previous_jams):
            right_cell = jam[0][1]
            
            # find the jam in the current jams that has the rightmost cell one cell to the left
            for current_jam in current_jams:
                if current_jam[1] == right_cell - 1:
                    previous_jams[i][1] += current_jam[1] - current_jam[0] + 1
                    previous_jams[i][0] = current_jam
                    lifespans[i] += 1
                    break
    
    if add_lifespans:
        return previous_jams, lifespans

    return previous_jams

def initial_to_random_walk(initial_state):
    # Plot the random walk that is the initial state, go up for 1, down for 0
    L = len(initial_state[0])
    random_walk = [0] * L
    for i in range(L):
        if initial_state[0][i] == 1:
            random_walk[i] = random_walk[i-1] + 1
        else:
            random_walk[i] = random_walk[i-1] - 1
    return random_walk

def jam_lifespans(initial_random_walk):
    # Find the lifespans of the jams based on the random walk of the initial state (see initial_to_random_walk). For every value in the random walk, find the next occurrence of that value and calculate the lifespan based on this return time.
    
    lifespans = []

    for i in range(len(initial_random_walk) - 1):
        value = initial_random_walk[i]

        # If the next value is higher, we can't calculate the lifespan of a jam from here.
        if value < initial_random_walk[i + 1]:
            continue
        
        # Find the first next occurrence of the value
        try:
            next_occurrence = initial_random_walk[i + 1:].index(value) + i + 1
            assert initial_random_walk[next_occurrence] == value, "The next occurrence is not the same as the value we are looking for"
        
        except ValueError:
            continue
        
        
        lifespan = (next_occurrence - i)/2
        
        if lifespan > 1:
            lifespans.append(lifespan)
            
    return lifespans

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

def run_model(p, L, T, n_repetitions = 100):
    """
    Function to run the CA rule 184 model for a given p, L and T. It returns the lifespans and jam sizes of all the jams found in the evolution of the CA.
    
    Parameters:
    - p (float): The probability of a cell being 1 in the initial state.
    - L (int): The length of the CA.
    - T (int): The number of timesteps.
    - n_repetitions (int): The number of times the model should be run.
    
    Returns:
    - lifespan_counter (Counter): A counter with the lifespans of all the jams found in the evolutions of the CA.
    - jam_counter (Counter): A counter with the sizes of all the jams found in the evolutions of the CA.
    """
    ca_184 = CA()
    total_lifespans = []
    total_jam_sizes = []

    for _ in range(n_repetitions):
        initial_state = ca_184.gen_initial_state_bernoulli(L, p)
        random_walk = initial_to_random_walk(initial_state)
        cellular_automaton = cpl.evolve(initial_state, timesteps=T, memoize=True, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule=184))
        cellular_automaton = triangulize_evolution(cellular_automaton)

        lifespans = jam_lifespans(random_walk)
        total_lifespans += lifespans
        
        jams = find_jams(cellular_automaton)
        jam_sizes = [jam[1] for jam in jams]
        total_jam_sizes += jam_sizes
    
    lifespan_counter = Counter(total_lifespans)
    jam_counter = Counter(total_jam_sizes)
    return lifespan_counter, jam_counter

def run_model_wrapper(args):
    # Function to run the model with a single argument, so it can be used with concurrent.futures.ProcessPoolExecutor's map function.
    return run_model(args[0], args[1], args[2], args[3])

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

def nasch_step(current_state, v_max, p_slowdown):
    """
    Function to perform a single timestep of the NaSch model. It returns the next state of the CA.

    Parameters:
    - current_state (list): The current state of the CA. It contains a tuple for every cell, with the first element indicating whether the cell is occupied and the second element indicating the speed of the vehicle.
    - v_max (int): Maximum speed of vehicles.
    - p_slowdown (float): Probability of slowing down.

    Returns:
    - next_state (list): The next state of the CA. Same format as current_state.
    """

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
            

#changed run model function introducing stochasticity/dynamics Influx=Outflux 
def run_model_stochastic(p, L, T, n_repetitions=100, v_max=5, p_slowdown=0.1, triangular=False, visualize=False, return_evolutions=False):
    """
    Function to run the NaSch model for a given p, L, T. It returns the lifespans and jam sizes of all the jams found in the evolution of the model.

    Parameters:
    - p (float): The probability of a cell being 1 in the initial state.
    - L (int): The length of the simulation.
    - T (int): The number of timesteps.
    - n_repetitions (int): The number of times the model should be run.
    - v_max (int): Maximum speed of vehicles.
    - p_slowdown (float): Probability of slowing down.

    Returns:
    - lifespan_counter (Counter): A counter with the lifespans of all the jams found in the evolutions of the model.
    - jam_counter (Counter): A counter with the sizes of all the jams found in the evolutions of the model.
    - all_evolutions (list): A list containing the evolution of the model for every repetition. Only returned if return_evolutions is True.
    """
    total_lifespans = []
    total_jam_sizes = []

    all_evolutions = []

    for i in range(n_repetitions):
        # Create initial state
        initial_state = initial_state_nasch(L, p, v_max)

        # Run the model
        evolution = [initial_state]
        for t in range(T):
            current, next = nasch_step(evolution[-1], v_max, p_slowdown)
            evolution[-1] = current
            evolution.append(next)
        
        if return_evolutions:
            all_evolutions.append(evolution)

        location_states = np.array([[cell[0] for cell in state] for state in evolution])

        if triangular:
            location_states = triangulize_evolution(location_states)

        jams, lifespans = find_jams(location_states, add_lifespans=True)
        
        jam_sizes = [jam[1] for jam in jams]
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
    amount of car movement in total. This is calculated by summing the total movement of cars.
    """
    total_flow = 0
    
    # Check all timesteps but the last one, because we assume that no movement happens in the last timestep
    for t in range(len(evolution[:-1])):
        for i in range(len(evolution[t])):
            speed = evolution[t][i][1]
            car_present = evolution[t][i][0]
            if car_present:
                # assert that there's a car in the cell i + speed in the next timestep
                if i + speed < len(evolution[t]):
                    assert evolution[t + 1][i + speed][0], f"There should be a car in the next timestep (error at t = {t}, i = {i}, speed = {speed})"
                
                total_flow += speed
    
    return total_flow

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


def find_critical_dataframe_nasch(p_slowdown_values, v_max_values, p_values, L, T, n):
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

    # Create a list with inputs for the critical_density_wrapper function
    args = []
    for v_max in v_max_values:
        for p_slowdown in p_slowdown_values:
            args.append((p_values, L, T, n, v_max, p_slowdown))

    # Run the model for every combination of p_slowdown and v_max
    with concurrent.futures.ProcessPoolExecutor() as executor:
        output = list(tqdm(executor.map(critical_density_wrapper, args), total=len(args), desc=f"Finding critical densities, {len(v_max_values)} v_max values and {len(p_slowdown_values)} p_slowdown values"))
    
    # Create a dataframe with the results
    v_maxes = []
    p_slowdowns = []
    critical_densities = []

    for output, args in zip(output, args):
        v_maxes.append(args[4])
        p_slowdowns.append(args[5])
        critical_densities.append(output)
    
    # Create DataFrame
    output_df = pd.DataFrame({
        'p_slowdown': p_slowdowns,
        'v_max': v_maxes,
        'critical_density': critical_densities
    })

    return output_df

def visualize_jam_counter(jam_counter, fit_line = False):
    plt.figure(figsize=(12,6))
    plt.title(f'Jam Sizes')
    plt.loglog(range(1, int(max(jam_counter.keys()))), [jam_counter[i] for i in range(1, int(max(jam_counter.keys())))], 'o')
    plt.xlabel('Jam size')
    plt.ylabel('Number of jams')

    if fit_line:
        lists = sorted(jam_counter.items())
        x, y = zip(*lists)
        x_log, y_log = np.log(x), np.log(y)

        # Fit linear function to log-log data   
        def func(x, a, b):
            return -a * x + b
        
        def power_law_func(x, a, b):
            return a * x ** -b

        popt, _ = scipy.optimize.curve_fit(func, x_log, y_log, p0=[1, 0.5])
        a_optimal, b_optimal = popt

        # plot the fitted line
        plt.plot(x, power_law_func(x, np.exp(b_optimal), a_optimal), label=f'Linear fit on log-log data: y = {np.exp(b_optimal):.2f}x^-{a_optimal:.2f}')

        # Fit power-law function
        popt, _ = scipy.optimize.curve_fit(power_law_func, x, y, p0=[1, 0.5])
        a_optimal, b_optimal = popt

        # plot the fitted line
        plt.plot(x, power_law_func(x, a_optimal, b_optimal), label=f'Power-law fit: y = {a_optimal:.2f}x^-{b_optimal:.2f}')

    plt.legend()
    plt.show()


def run_model_for_densities(p_values, L, T, p_repetitions, n_repetitions, concurrently = True):
    """
    Function to run the model for a given list of p values, L and T. It fits a power law to the data and returns p_repetitions fitted exponents for every p value.

    Parameters:
    - p_values (list): The list of p values to run the model for.
    - L (int): The length of the CA.
    - T (int): The number of timesteps.
    - p_repetitions (int): The number of times the model should be run for each p value.
    - n_repetitions (int): The number of times the model should be run to estimate the exponent once.
    - concurrently (bool): Whether or not to run the model concurrently.

    Returns:
    TODO
    """

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = tqdm_parallel_map(executor, assigned_material_job, job_ids)

    results_dict = {}

    for p in p_values:
        print(f"Running model for p = {p}")
        results_dict[p] = []

        args = [(p, L, T, n_repetitions)] * p_repetitions

        if not concurrently:
            raise NotImplementedError("Consecutive version not implemented yet")

        if concurrently:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                output = list(tqdm(executor.map(run_model_wrapper, args), total=len(args)))

                # Now for every output, we have two counters. From the jam size counter we want to fit an exponent
                for out in output:
                    jam_counter = out[1]
                    
                    

                    lists = sorted(jam_counter.items())
                    x, y = zip(*lists)
                    x_log, y_log = np.log(x), np.log(y)

                    # Fit linear function to log-log data   
                    def func(x, a, b):
                        return -a * x + b
                    
                    test_index = int(len(x_log) * 0.8)

                    popt, _ = scipy.optimize.curve_fit(func, x_log[:test_index], y_log[:test_index], p0=[1, 0.5])
                    a_optimal, _ = popt

                    results_dict[p].append(a_optimal)

                    # visualize_jam_counter(jam_counter, fit_line = True)

                    # visualize_jam_counter(jam_counter, fit_line = False)

                    # lists = sorted(jam_counter.items())
                    # x, y = zip(*lists)

                    # # Fit power-law function
                    # def func(x, a, b):
                    #     return a * x ** -b

                    # popt, _ = scipy.optimize.curve_fit(func, x, y, p0=[1, 0.5])
                    # _, b_optimal = popt

                    # results_dict[p].append(b_optimal)
        
        print(f"Results for p = {p}: {results_dict[p]}")
    
    # Visualize exponents for every p value
    plt.figure(figsize=(12,6))
    plt.title(f'Exponent for every p value')
    plt.plot(p_values, results_dict.values(), 'o')
    plt.xlabel('p')
    plt.ylabel('Exponent')

    # # Fit line through p_values and exponents 
    # def func(x, a, b):
    #     return a * x + b
    
    # popt, _ = scipy.optimize.curve_fit(func, x, y, p0=[1, 0.5])
    # a_optimal, b_optimal = popt

    # plt.plot(x, func(x, a_optimal, b_optimal), label=f'Linear fit: y = {a_optimal:.2f}x + {b_optimal:.2f}')
    
    plt.show()

    return results_dict
        

        


