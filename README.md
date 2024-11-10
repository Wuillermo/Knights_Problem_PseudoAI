# *The Knights Problem using AI*
## The problem

The aim is to solve a variant of the chess queens problem, using knights instead, to find out how many knights can be present on a chessboard 
without threatening each other. Any configuration of knights on the board is valid as long as they do not threaten each other, but you want 
to find the maximum number of knights. Below are several examples on a 3x3 chessboard:
```
Optimal and Valid   Valid   Not valid
K·K                 K·K     K··
·K·                 ···     ··K
K·K                 K··     ·K·
```
The aim of the algorithm is to find a valid configuration with as many horses as possible.

It is possible that the problem configuration is too large for some of the algorithms. As a rule of thumb, if the algorithm takes more than 5 
minutes to complete its execution, we can declare that the algorithm has not found a solution in a reasonable time (and we indicate this in 
the analysis of results).

* Various configurations are provided:
    * A \**2x2** board,
    * A **3x3** board,
    * A **3x5** board,
    * A **5x5** board,
    * A **8x8** board.
* Two algorithms are to be applied:
    * Branch & Bound: We want to obtain an optimal solution, (maximum number of horses)
    * A-Star: It's provided at least one admissible heuristic for finding an optimal solution. In this report, the admissibility of the 
      heuristic must be justified and demonstrated.

* The use of external libraries is not allowed except for numpy and pandas.