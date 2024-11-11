
from tests.launch_experiment import launch_experiment
from KnightsProblem import heuristic_1

"""### Branch & Bound """

launch_experiment('2x2')
print()
launch_experiment('3x3')
print()
launch_experiment('3x5')
print()
launch_experiment('5x5')
print()
#launch_experiment('8x8')
print()

#it's not going to end, cant find the 8x8 in less than 5min

"""### A* """

launch_experiment('2x2', heuristic=heuristic_1)
print()
launch_experiment('3x3', heuristic=heuristic_1)
print()
launch_experiment('3x5', heuristic=heuristic_1)
print()
launch_experiment('5x5', heuristic=heuristic_1)
print()
#launch_experiment('8x8', heuristic=heuristic_1)
print()

#it's not going to end, cant find the 8x8 in less than 5min