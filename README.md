
# Self-Organized Criticality in a One Lane Traffic Flow Model / Group 1

For the course Complex System Simulation from the Master's Computational Science (joint degree) at the University of Amsterdam and Vrije Universiteit Amsterdam our group explored the properties which would implicate self organized criticality of a 1D dynamic implementation of the Nasch Model.


## Git-fame work distribution table
| Author    |   loc |   coms |   fils |  distribution   |
|:----------|------:|-------:|-------:|:----------------|
| Jenna     | 50443 |     27 |     17 | 73.5/22.0/32.1  |
| Lingfeng  | 14052 |     15 |     13 | 20.5/12.2/24.5  |
| Caro      |  2338 |     41 |     13 | 3.4/33.3/24.5   |
| Thomas    |  1817 |     40 |     10 | 2.6/32.5/18.9   |


## Assert usage
Our modules.py file contains 5.7% assert statements (19 assert statements / 335 lines of code)

## Installation
Use the following command to install dependencies:

```python
pip install -r requirements.txt
```

## Usage
Have the modules.py file inside the directory you are working on, then in any py or ipynb file you can use:

```python
import modules
```

To import the module with its functions.

## Usage Example
To run a basic simulation, execute the following command after importing modules:

```python
run_model_stochastic(p=0.5, L=100, T=50, n_repetitions=100, v_max=5, p_slowdown=0.1, triangular=False, return_evolutions=False, 
                         dynamic_model=True, neighbourhood_size=1, entry_chance=0.5, exit_chance=0.5)
```


## Functions

- **find_jams(evolution, add_lifespans=False):**
  Identifies and returns the sizes and lifespans of traffic jams in the given CA evolution.

- **triangulize_evolution(evolution):**
  Removes cells that are not part of the triangle in the evolution of a CA. This ensures that all cars can reach all jams.

- **run_model(p, L, T, n_repetitions=100):**
  Runs the CA rule 184 model for a given initial density, road length, and time steps. Returns the lifespans and jam sizes of all the jams found in the evolution of the CA.

- **run_model_stochastic(p, L, T, n_repetitions=100, v_max=5, p_slowdown=0.1, triangular=False, return_evolutions=False, dynamic_model=False, neighbourhood_size=1, entry_chance=0.5, exit_chance=0.5):**
  Runs the NaSch model for a given set of parameters. Returns the lifespans and jam sizes of all the jams found in the evolution of the model. Can also return the evolution of the model for each repetition.

- **calculate_flow_nasch(evolution):**
  Calculates the total flow of the NaSch model, defined as the sum of speeds of all vehicles over every timestep.

- **calculate_delay_nasch(evolution, v_max):**
  Calculates the average delay of the NaSch model, defined as the average of the differences between v_max and the exact speed of each vehicle.

- **critical_density_wrapper(args):**
  Runs the model with a single argument, suitable for concurrent execution, to find the critical density.

- **find_critical_dataframe_nasch(p_slowdown_values, v_max_values, p_values, L, T, n, repetitions=1):**
  Finds critical densities for combinations of p_slowdown and v_max values in the NaSch model. Returns a dataframe with the critical densities for every combination of p_slowdown and v_max.

- **visualize_jam_counter(jam_counter, fit_line=False):**
  Plots a log-log graph of jam sizes and the number of jams. Optionally fits a line to the data.

- **find_density(L, p, n, v_max, p_slowdown, dynamic_model, neighbourhood_size, entry_chance, exit_chance):**
  Finds the density evolution of the NaSch model for a given set of parameters.

- **analyze_powerlaw_distribution(data):**
  Analyzes the distribution of given data using the powerlaw package. Returns a string with the distribution type.

- **analyze_critical_exponent_for_density(p, L, T, n_repetitions, v_max, p_slowdown, triangular, dynamic_model, neighbourhood_size, entry_chance, exit_chance):**
  Saves the critical exponent for a given initial density.

- **density_evolution_nasch(evolution):**
  Calculates the density evolution of the NaSch model, defined as the sum of cars over every timestep.

## Authors

- Thomas ten Damme
- Caro Kluin
- Lingfeng Li
- Jenna de Vries

## License

This project is licensed under the [MIT] - see the [LICENSE](LICENSE) file for details.


