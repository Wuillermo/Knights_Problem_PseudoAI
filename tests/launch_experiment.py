
import KnightsProblem
import numpy
from tests.timer_utils import timer


@timer
def search_horse_byb(initial_board):
    return KnightsProblem.search(initial_board, KnightsProblem.expand, KnightsProblem.cost, None, KnightsProblem.order_byb, KnightsProblem.is_solution)

@timer
def search_horse_astar(initial_board, heuristic):
    return KnightsProblem.search(initial_board, KnightsProblem.expand, KnightsProblem.cost, heuristic, KnightsProblem.order_astar, KnightsProblem.is_solution)


"""### Experiment Launcher"""

CONF = {'2x2': (2, 2),
        '3x3': (3, 3),
        '3x5': (3, 5),
        '5x5': (5, 5),
        '8x8': (8, 8),
        }

def measure_solution(board):
    # Returns the number of horses in the solution
    number_of_knights = numpy.sum(board == 1)

    return number_of_knights

def launch_experiment(configuration, heuristic=None):
    conf = CONF[configuration]
    print(f"Running {'A*' if heuristic else 'B&B'} with {configuration} board")
    if heuristic:
        sol = search_horse_astar(KnightsProblem.initial_state(*conf), heuristic)
    else:
        sol = search_horse_byb(KnightsProblem.initial_state(*conf))
    n_c = measure_solution(sol)
    print(f"Solution found: \n{sol}")
    print(f"Number of horses in solution: {n_c}")

    return sol, n_c