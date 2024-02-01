import numpy as np
import random
from collections import Counter

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

def find_jams(evolution, add_lifespans=False):
    """
    A function to find the jams in the evolution of a CA. It checks every row for groupings of 1's that are bigger than 1.
    Then it checks if it can find the same group in the next row, and if there is only one empty cell between them,
    it adds the length of the group to the size of the jam. The resulting list contains the starting and ending index
    of the last row that contained the jams, as well as the size of the jam. If add_lifespans is True, it also contains
    the lifespans of the jams.

    Parameters:
    - evolution (list): The evolution of the CA.
    - add_lifespans (bool): Whether or not to add the lifespans of the jams to the result.

    Returns:
    - result (list): A list containing the starting and ending index of the last row that contained the jams,
                    as well as the size of the jam. If add_lifespans is True, it also contains the lifespans of the jams.
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

    # Keep track of disappearing cars in the previous timestep
    disappearing_cars = [jam[0][1] for jam in previous_jams]

    for row in evolution[1:]:
        current_jams = row_jams(row)

        # The rightmost cell of every previous jam should now be one cell to the left in the current jam
        for i, jam in enumerate(previous_jams):
            right_cell = jam[0][1]

            # Find the jam in the current jams that has the rightmost cell one cell to the left
            for current_jam in current_jams:
                # Check if there is only one empty cell between them or if alternative_cell is in disappearing_cars
                if current_jam[1] == right_cell - 1 or right_cell in disappearing_cars:
                    previous_jams[i][1] += current_jam[1] - current_jam[0][0] + 1
                    previous_jams[i][0] = current_jam
                    lifespans[i] += 1

                    # Update disappearing_cars with the rightmost cell that disappeared
                    disappearing_cars[i] = jam[0][1]

                    break

    if add_lifespans:
        return previous_jams, lifespans

    return previous_jams



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

        jams, lifespans = find_jams(location_states, add_lifespans=True)
        
        jam_sizes = [jam[1] for jam in jams]
        total_jam_sizes += jam_sizes
        total_lifespans += lifespans

    lifespan_counter = Counter(total_lifespans)
    jam_counter = Counter(total_jam_sizes)

    if return_evolutions:
        return lifespan_counter, jam_counter, all_evolutions

    return lifespan_counter, jam_counter



