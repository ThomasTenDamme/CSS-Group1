
def double_fit(self, x, y):
    """
    Fit two functions to the data.

    Parameters:
    - x (list): Independent variable.
    - y (list): Dependent variable.

    Returns:
    - tuple: Coefficients and index of the optimal fit.
    """
    highest_b2 = 0

    for index in range(int(len(x) * 0.2), int(len(x) * 0.8)):
        popt1, pcov1 = scipy.optimize.curve_fit(self.func, x[0:index], y[0:index], p0=[1, 0.5], maxfev=5000)
        a1, b1 = popt1

        popt2, pcov2 = scipy.optimize.curve_fit(self.func, x[index:], y[index:], p0=[1, 0.5], maxfev=5000)
        a2, b2 = popt2

        if b2 > highest_b2:
            highest_b2 = b2
            best_a1, best_b1, best_a2, best_b2 = a1, b1, a2, b2
            optimal_index = index

    return best_a1, best_b1, best_a2, best_b2, optimal_index

def find_intersection(self, x_data, a1, b1, a2, b2):
    """
    Find the intersection of two fitted functions.

    Parameters:
    - x_data (list): Data points.
    - a1 (float): Coefficient of the first function.
    - b1 (float): Exponent of the first function.
    - a2 (float): Coefficient of the second function.
    - b2 (float): Exponent of the second function.

    Returns:
    - float/None: Intersection point or None if it could not be found
    """
    x = np.linspace(1, np.max(x_data), np.max(x_data))
    y1 = self.func(x, a1, b1)
    y2 = self.func(x, a2, b2)
    for index in range(len(x)):
        if y1[index] >= y2[index]:
            return x[index]

    return None

def plot_fit(self, x, y, a1, b1, a2, b2):
    """
    Plot the fitted functions.

    Parameters:
    - x (list): Independent variable.
    - y (list): Dependent variable.
    - a1 (float): Coefficient of the first function.
    - b1 (float): Exponent of the first function.
    - a2 (float): Coefficient of the second function.
    - b2 (float): Exponent of the second function.

    Returns:
    - None
    """
    plt.clf()
    plt.loglog(x, y, 'o', markersize=2)
    plt.xlim(0, 1.1 * np.max(x))
    plt.ylim(0, 1.1 * np.max(y))
    plt.loglog(x, self.func(x, a1, b1))
    plt.loglog(x, self.func(x, a2, b2))
    plt.show()

    return None

def find_critical_size(self, density_list, n_simulations, L, T, n_repetitions):
    """
    Find critical sizes of jams for different densities in the cellular automaton model.

    Parameters:
    - density_list (list): List of densities to simulate.
    - n_simulations (int): Number of simulations for each density.
    - L (int): Length of the cellular automaton.
    - T (int): Number of time steps for each simulation.
    - n_repetitions (int): Number of times the model should be run.

    Returns:
    dict: A dictionary mapping densities to lists of critical sizes for each simulation.
    """
    
    critical_size_dict = {}
    for p in density_list:
        for n in range(n_simulations):
            print(p, n)
            lifespan_counter, final_jams_counted = Model.run_model(p, L, T, n_repetitions)
            lists = sorted(final_jams_counted.items())  # sorted by key, return a list of tuples

            x_data, y_data = zip(*lists)
            a_optimal1, b_optimal1, a_optimal2, b_optimal2, optimal_index = self.double_fit(x_data, y_data)

            critical_size = self.find_intersection(x_data, a_optimal1, b_optimal1, a_optimal2, b_optimal2)

            if critical_size is None:
                if p in critical_size_dict:
                    del critical_size_dict[p]
                break

            else:
                if p not in critical_size_dict:
                    critical_size_dict[p] = [critical_size]
                else:
                    critical_size_dict[p] += [critical_size]

    return critical_size_dict
