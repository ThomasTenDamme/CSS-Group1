import numpy as np
import cellpylib as cpl
from collections import Counter
import matplotlib.pyplot as plt
import scipy.optimize
import imageio

class Model:
    """
    Class representing the cellular automaton (CA) model with rule 184 (the traffic rule).

    Attributes:
    - L (int): Length of the CA.
    - t_end (int): End time of evolution.
    - initial_state (bool): Initial state of the CA.
    - evolution (list): Evolution of the CA.

    Methods:
    - gen_initial_state_bernoulli(L, p): Generate initial state with a Bernoulli distribution.
    - find_jams(evolution): Find jams in the evolution of the CA.
    - initial_to_random_walk(initial_state): Convert initial state to a random walk.
    - jam_lifespans(initial_random_walk): Find lifespans of jams based on the random walk.
    - triangulize_evolution(evolution): Remove cells that are not part of the triangle in CA evolution.
    - run_model(p, L, T, n_repetitions): Run the CA model for a given set of parameters.
    """

    def __init__(self):
        """
        Initialize the cellular automaton model.
        """
        
        self.L = 0
        self.t_end = 0
        self.initial_state = False
        self.evolution = None

    def gen_initial_state_bernoulli(self, L, p):
        """
        Generate initial state with a Bernoulli distribution.

        Parameters:
        - L (int): Length of the CA.
        - p (float): Probability of a cell being occupied in the initial state.

        Returns:
        - ndarray: Initial state of the CA.
        """

        assert p >= 0 and p <= 1, "p must be between 0 and 1"
        assert L > 0, "L must be greater than 0"

        self.L = L
        self.initial_state = np.array([[np.random.choice([0, 1], p=[1 - p, p]) for i in range(self.L)]])
        return self.initial_state

    def find_jams(self, evolution):
        """
        Find jams in the evolution of a CA. It checks every row for groupings of 1's that are bigger than 1.
        Then it checks if it can find the same group in the next row, if so, it adds the length of the group to the size of the jam.
        The resulting list contains the starting and ending index of the last row that contained the jams, as well as the size of the jam.

        Parameters:
        - evolution (list): Evolution of the CA.

        Returns:
        - list: Starting and ending index of the last row that contained jams, as well as the size of the jam.
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

    def initial_to_random_walk(self, initial_state):
        """
        Convert the initial state to a random walk.

        Parameters:
        - initial_state (ndarray): Initial state of the CA.

        Returns:
        - list: Random walk representing the initial state.
        """

        L = len(initial_state[0])
        random_walk = [0] * L
        for i in range(L):
            if initial_state[0][i] == 1:
                random_walk[i] = random_walk[i-1] + 1
            else:
                random_walk[i] = random_walk[i-1] - 1
        return random_walk
    
    def jam_lifespans(self, initial_random_walk):
        """
        Find the lifespans of jams based on the random walk of the initial state.

        Parameters:
        - initial_random_walk (list): Random walk representing the initial state.

        Returns:
        - list: Lifespans of jams.
        """

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

    def triangulize_evolution(self, evolution):
        """
        Remove the cells that are not part of the triangle in the evolution of a CA.

        Parameters:
        - evolution (list): Evolution of the CA.

        Returns:
        - list: Evolution with cells not part of the triangle set to 0.
        """

        L = len(evolution[0])
        for y in range(len(evolution)):
            for x in range(len(evolution[y])):
                if y > x or y > L-x:
                    evolution[y][x] = 0
        return evolution

    def run_model(self, p, L, T, n_repetitions = 100):
        """
        Run the model for a given p, L, and T. It returns the lifespans and jam sizes of all the jams found in the evolution of the CA.

        Parameters:
        - p (float): The probability of a cell being 1 in the initial state.
        - L (int): The length of the CA.
        - T (int): The number of timesteps.
        - n_repetitions (int): The number of times the model should be run.

        Returns:
        - Counter: Lifespans of all the jams found in the evolutions of the CA.
        - Counter: Sizes of all the jams found in the evolutions of the CA.
        """

        total_lifespans = []
        total_jam_sizes = []

        for i in range(n_repetitions):
            initial_state = self.gen_initial_state_bernoulli(L, p)
            random_walk = self.initial_to_random_walk(initial_state)
            cellular_automaton = cpl.evolve(initial_state, timesteps=T, memoize=True, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule=184))
            cellular_automaton = self.triangulize_evolution(cellular_automaton)

            lifespans = self.jam_lifespans(random_walk)
            total_lifespans += lifespans
            
            jams = self.find_elem_jams(cellular_automaton)
            jam_sizes = [jam[1] for jam in jams]
            total_jam_sizes += jam_sizes
        
        lifespan_counter = Counter(total_lifespans)
        jam_counter = Counter(total_jam_sizes)
        return lifespan_counter, jam_counter

class Results:
    """
    Class for analyzing and visualizing results.

    Attributes:
    - None

    Methods:
    - func(x, a, b): Function for fitting.
    - single_fit(x, y): Fit a single function to the data.
    - double_fit(x, y): Fit two functions to the data.
    - find_intersection(x_data, a1, b1, a2, b2): Find the intersection of two fitted functions.
    - plot_fit(x, y, a1, b1, a2, b2): Plot the fitted functions.
    - find_critical_size(density_list, n_simulations, L, T, n_repetitions): Find critical size based on simulations.
    """

    def func(self, x, a, b):
        """
        Function for fitting.

        Parameters:
        - x (list): Independent variable.
        - a (float): Coefficient.
        - b (float): Exponent.

        Returns:
        - list: Result of the function.
        """
        return [a * x_i ** -b for x_i in x]
    
    def func_linear(self, x, a, b):
        return [a - b * x_i for x_i in x]

    def single_fit(self, x, y):
        """
        Fit a single function to the data.

        Parameters:
        - x (list): Independent variable.
        - y (list): Dependent variable.

        Returns:
        - tuple: Coefficients of the fit.
        """
        popt1, pcov1 = scipy.optimize.curve_fit(self.func, x, y, p0=[1, 0.5])
        a1, b1 = popt1

        return a1, b1
            
    def single_fit_linear(self, x, y):
        popt1, pcov1 = scipy.optimize.curve_fit(self.func_linear, x, y, p0=[1, 0.5])
        a1, b1 = popt1

        return a1, b1

class Animate:
    def plot_fit_saver(x, y, a1, b1, a2, b2, index):
        plt.clf()
        plt.loglog(x, y, 'o', markersize=2)
        plt.xlim(0, 1.1*np.max(x))
        plt.ylim(0, 1.1*np.max(y))
        plt.loglog(x, func(x, a1, b1))
        plt.loglog(x, func(x, a2, b2))
        plt.savefig(f'Animation/fig{index}.png')

    filenames = [f'Animation/fig{index}.png' for index in range(int(len(x_data)*0.2), int(len(x_data)*0.8))]
    print(filenames)

    with imageio.get_writer('Animation/critical_size.gif', mode='I') as writer:      #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)