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
