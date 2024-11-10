
""" ## Initial State """

### Initial state

import numpy as np

# 0 = Safe square
# 1 = Knight in place
# 2 = Attacked by knight

def initial_state(M, N):
    # Creates an empty board using 0s
    return np.zeros((M, N), dtype=int)


"""### State expansion """

# Possible movements of a knight
movimientos = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]

def expand(board):
    boards = [] # Create an empty list of boards
    M, N = board.shape # Create the M and N variables of the board to have easy access to them
    empty_positions = np.argwhere(board == 0) # Create a list of empty positions

    for i, j in empty_positions: # Loop through each empty position
      boards.append(place(board, i, j))

    # Creates a list of boards with all possible moves

    return boards # Return a list of boards

def copy_board(board):
  return np.copy(board)

def place(board, x, y):
  copied_board = copy_board(board)
  copied_board[x][y] = 1
  M, N = copied_board.shape
  # Add the attacked squares to the board
  for i_mov, j_mov in movimientos:
    ni, nj = x + i_mov, y + j_mov
    if 0 <= ni < M and 0 <= nj < N:
      copied_board[ni][nj] = 2

  return copied_board

# - A function that copies an entire board
# - A function that places a horse at a given position in i, j
# - A data structure with the possible movements for a horse


"""### Solution reached """

def is_solution(board):
    # Check if a board is a solution
    M, N = board.shape  # Dimensions of the board

    if M == 2 and N == 2:
      # A 2x2 board has always 4 knights as solution
      return True if np.sum(board == 1) == 4 else False

    # Other boards have this formula as max knights
    # np.ceil gives the reult of the operation rounded up (as a float)
    # astype makes it an integer
    return True if np.sum(board == 1) == np.ceil((M * N) / 2).astype(int) else False

    # Make all necessary checks to determine
    # if the board is a solution


""" ## Metrics """

### Cost function

costs_dic = {}

def cost(path): # path must contain MULTIPLE boards
    # Calculate the cost of a path
    # This should be possible: board = path[-1]
     board = path[-1]

     total_knights = np.sum(board == 1)
     attacked_squares = np.sum(board == 2)

    # maximising ocuppied squares while minimising attacked squares
     cost_value = attacked_squares - total_knights + 1

    # Calculate the cost of a complete path

     return cost_value

# - Remember that A* and B&B work by minimising cost.
# - Can we tackle this problem in another way? Maximising the occupied squares does NOT work...

"""### Heuristic(s) """

heuristic_cache = {}

# Valid heuristic
def heuristic_1(board):
    # Calculate a heuristic for a board here
     M, N = board.shape  # Dimensions of the board
     max_knights = 0

    # Calculate max knights like in is_solution
     if M == 2 and N == 2:
      max_knights = 4
     else:
      max_knights = np.ceil((M * N) / 2).astype(int)

    # Look if we have already calculated this heuristic
     board_key = np.packbits(board.flatten() != 0).tobytes()
     if board_key in heuristic_cache:
      return heuristic_cache[board_key]

     knights_on_board = np.sum(board == 1)
     heuristic_value = 3*(max_knights - knights_on_board)
     heuristic_cache[board_key] = heuristic_value


     return heuristic_value

# Other heuristics

def heuristic_2(board):
    # Calculate a heuristic for a board
     M, N = board.shape  # Dimensions of the board
     max_knights = 0

     if M == 2 and N == 2:
      max_knights = 4
     else:
      max_knights = np.ceil((M * N) / 2).astype(int)

     board_key = np.packbits(board.flatten() != 0).tobytes()
     if board_key in heuristic_cache:
      return heuristic_cache[board_key]

     safe_squares = np.sum(board == 0)
     attacked_squares = np.sum(board == 2)
     heuristic_value = 2 * safe_squares - attacked_squares
     heuristic_cache[board_key] = heuristic_value

    # Calculate a heuristic for a board here

     return heuristic_value


def heuristic_3(board):
    # Calculate a heuristic for a board
     M, N = board.shape  # Dimensions of the board
     max_knights = 0

     if M == 2 and N == 2:
      max_knights = 4
     else:
      max_knights = np.ceil((M * N) / 2).astype(int)

     board_key = np.packbits(board.flatten() != 0).tobytes()
     if board_key in heuristic_cache:
       return heuristic_cache[board_key]

     safe_squares = np.sum(board == 0)
     knights_on_board = np.sum(board == 1)
     heuristic_value = max_knights + safe_squares - knights_on_board
     heuristic_cache[board_key] = heuristic_value

    # Calculate a heuristic for a board here

     return heuristic_value
# - As with cost, the smaller the value of the heuristic the better, since it is intended to be minimised.


""" ## Search Algorithm """

### Prunning

# Check the symmetries to see if they are identical to remove unnecessary paths.
def generate_unique_transformations(board):
    #Generate a minimal set of unique transformations for symmetry checking.
    transformations = [
        board,
        np.rot90(board, 1),  # 90-degree rotation
        np.rot90(board, 2),  # 180-degree rotation
        np.rot90(board, 3),  # 270-degree rotation
        np.fliplr(board),    # Horizontal flip
        np.flipud(board)     # Vertical flip
    ]
    # Convert to hashable tuples and deduplicate
    unique_transforms = set(tuple(trans.flat) for trans in transformations)
    return unique_transforms

def prune(path_list):
    unique_paths = []
    seen_transformations = set()

    for path in path_list:
        board = path[-1]  # Extract board representation from the path

        board_transforms = generate_unique_transformations(board)

        # Check if any transformation of this board is already in seen_transformations
        if not any(transform in seen_transformations for transform in board_transforms):
          # If unique, add this path and its transformations to the sets
          unique_paths.append(path)
          seen_transformations.update(board_transforms)
    return unique_paths # Return a list of paths

    # If it detects that two paths lead to the same state,
    # we are only interested in the path with the lowest cost
    # Later we use pruning after ordering.


""" ### Ordering """

# *args and **kwargs are variable arguments, if the argument is not recognized it is stored in these variables.
# They are used here to ignore unnecessary arguments.

# Used to give you the board in a flat format making it a possible key for a dictionary
def _get_bitboard_key(board):
    return np.packbits(board.flatten() != 0).tobytes()

costs_dic = {}
costs_dic_astar = {}

def order_astar(old_paths, new_paths, c, h, *args, **kwargs):
    all_paths = old_paths + new_paths

    def get_cost_and_heuristic(path):
      board_key = _get_bitboard_key(path[-1])  # Convert the last board state to a bitboard key
      # Only calculate cost a heuristic if we havent seen this board yet
      if board_key in costs_dic_astar:
          return costs_dic_astar[board_key]
      cost = c(path)
      heuristic = h(path[-1])
      costs_dic_astar[board_key] = (cost, heuristic)
      return cost, heuristic

     # Precompute costs and heuristics
    evaluated_paths = [(path, *get_cost_and_heuristic(path)) for path in all_paths]
    # Sort the list of paths according to a heuristic and cost
    sorted_paths = sorted(evaluated_paths, key=lambda x: x[1] + x[2])  # using cost and heuristic directly from the evaluated list
    return prune([x[0] for x in sorted_paths])  # Return the list of paths sorted and pruned according to A*

def order_byb(old_paths, new_paths, c, *args, **kwargs):
  all_paths = old_paths + new_paths

  def get_cost(path):
      board_key = _get_bitboard_key(path[-1])  # Convert the last board state to a bitboard key
      # Only calculate cost if we havent seen this board yet
      if board_key in costs_dic:
          return costs_dic[board_key]
      cost = c(path)
      costs_dic[board_key] = (cost)
      return cost

  evaluated_paths = [(path, get_cost(path)) for path in all_paths]
  sorted_paths = sorted(evaluated_paths, key=lambda x: x[1])  # using cost directly from the evaluated list
  return prune([x[0] for x in sorted_paths])  # Return the list of paths sorted and pruned according to B&B


"""### Search Algorithm"""

def search(initial_board, expansion, cost, heuristic, ordering, solution):
    # Performs a search in the state space

    paths = [[initial_board]] # Create the list of paths
    solution_path = None # This is the solution state
    costs_dic = {} # initialize the costs dictionary for the optimization of the ordering function on byb
    costs_dic_astar = {} # initialize the costs dictionary for the optimization of the ordering function on astar
    heuristic_cache = {} # initialize the heuristics dictionary for the optimization of the ordering function on astar

    # 1 - As long as there are paths and no solution has been found
    while paths and solution_path is None:

      # 2 - Extract the first path (the one with best==lowest score after ordering)
      path = paths.pop(0)

      # 3 - Check to see if this is a solution state
      if solution(path[-1]):
        solution_path = path[-1]
        break

      # 4 - If it is not a solution, expand the path/ If it is a solution, stop and go to step 7
      expanded_boards = expansion(path[-1])

      # 5 - For each new expanded state, add it to the path, which generates a list of new paths
      if expanded_boards:
        new_paths = [path + [board] for board in expanded_boards]
        # 6 - Sort the new paths and old paths, and perform pruning. Return to step 1
        paths = ordering(paths, new_paths, cost, heuristic)

    # 7 - Return the path if it is a solution, otherwise return None
    return solution_path if solution_path is not None else None # Return only the solution, not the solution path


"""

### A*
"""

#it's not going to end, cant find the 8x8 in less than 5min

"""
## Conclusions

The comparison table between A* and B&B, add a critical assessment of the results, specifying the differences you find between both 
search algorithms, advantages of using one over the other, the effect of the problem configuration, etc.

| **Board** | **Algorithm** | **Time B&B** | **Time A*** | **Horses B&B** | **Horses A*** |
|:---------:|:-------------:|:------------:|:-----------:|:--------------:|:-------------:|
|    2x2    |  B&B and  A*  |     0.002    |    0.001    |        4       |       4       |
|    3x3    |  B&B and  A*  |     0.025    |    0.004    |        5       |       5       |
|    3x5    |  B&B and  A*  |     0.062    |    0.031    |        8       |       8       |
|    5x5    |  B&B and  A*  |    237.637   |    0.145    |       13       |       13      |
|    8x8    |  B&B and  A*  |     NONE     |     NONE    |      NONE      |      NONE     |


Here we have the two tables of B&B and A* combined in order to compare them.
Starting with the 2x2, we observe similar times and this continous to be like this for the 3x3 and the 3x5. The latter let us see that 
B&B sometimes could even takes less time than the A\* during tests, wich could be a result of the inefficient calculation of the heuristic 
for such a small problem.

The real difference occurs in the 5x5, here we notice the abysmal difference between the two algorithms, while B&B takes the staggering 
time of 237.637 seconds, A\* is able to do it in barely half a second, precisely in 0.584 seconds.

Concluding, A* is more optimal for higher numbers of rows and columns boards than B&B, although B&B is better for small ones. This is 
caused due to the heuristic function, which helps a lot in the process of seleccting the best path. Though it takes more time in the 
small one, on the bigger ones makes it exponentially faster since it wastes less time on the worst boards.
"""