import numpy as np
import random

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


def find_elem_jams(evolution):
    """
    A function to find the elementary jams in the evolution of a CA. It checks every row for groupings of 1's that are bigger than 1.
    Then it checks if it can find the same group in the next row, if so, it adds the length of the group to the size of the jam.
    The resulting list contains the starting and ending index of the last row that contained the jams, as well as the size of the jam.
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
                    break
    
    return previous_jams

def initial_to_random_walk(initial_state):
    # Plot the random walk that is the initial state, go up for 1, down for 0
    L = len(initial_state[0])
    random_walk = [0] * L
    for i in range(1, L):
        if initial_state[0][i] == 1:
            random_walk[i] = random_walk[i-1] + 1
        else:
            random_walk[i] = random_walk[i-1] - 1
    return random_walk

def jam_lifespans(initial_random_walk):
    # Find the lifespans of the jams based on the random walk of the initial state (see initial_to_random_walk). For every value in the random walk, find the next occurrence of that value and calculate the lifespan based on this return time.
    
    lifespans = []

    for i in range(len(initial_random_walk)):
        value = initial_random_walk[i]

        # Find the first next occurrence of the value
        try:
            next_occurrence = initial_random_walk[i + 1:].index(value) + i + 1
            lifespans.append((next_occurrence - i)/2)
            
            assert initial_random_walk[next_occurrence] == value, "The next occurrence is not the same as the value we are looking for"

        except ValueError:
            pass


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

    
