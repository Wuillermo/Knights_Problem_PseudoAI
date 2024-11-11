# *The Knights Problem using Branch & Bound and A\**

Disclaimer: This code is a college group practical assignment that has been addapted to apply and practice some python proyect structures and
knowledges, and also with the aim of optimizing its functionality.

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
The aim of the algorithm is to find a valid configuration with as many horses as possible. In this solve it always tries to find the maximum
amount of knights possible.

It is possible that the problem configuration is too large for some of the algorithms. As a rule of thumb, if the algorithm takes more than 5 
minutes to complete its execution, we can declare that the algorithm has not found a solution in a reasonable time (and we indicate this in 
the analysis of results).

* Various configurations are provided:
    * A **2x2** board,
    * A **3x3** board,
    * A **3x5** board,
    * A **5x5** board,
    * A **8x8** board.
* Two algorithms are applied:
    * Branch & Bound
    * A-Star: It's provided at least one admissible heuristic for finding an optimal solution.

* The use of external libraries is not allowed except for numpy and pandas.

## Admisibility of the heuristic

We have several heuristics, each of them trying a different aproach.

* First = 3*(max_knights - knights_on_board)

* Second = 2 * safe_squares - attacked_squares

* Third = max_knights + safe_squares - knights_on_board

The second and third ones were the ones tried first but they both gave negative values so they had to be discarded.

The first one is always grater than 1 and it is multiplied by 3, with the intention to exaggerate the heuristic value itself when its bad, 
making better boards have a much better (lower in this case since a heuristic is always minimised) score.

**Regarding addmisibility**

We can safely assume that this heuristic is admissible, since it only accounts for the remaining knights to be placed and does not 
overestimate other complex calculations the algorithm might take. Also the problem being solved is simple enought to make such asumptions 
since it's result is always determisintic regardless of the board size.

## Results

**B&B**

| Board | Algorithm | Time    | Horses |
|-------|-----------|---------|--------|
| 2x2   | B&B       | 0.001   | 4      |
| 3x3   | B&B       | 0.010   | 5      |
| 3x5   | B&B       | 0.023   | 8      |
| 5x5   | B&B       | 124.1   | 13     |
| 8x8   | B&B       | >5min   | >5min  |

The cuantity of horses doesnt need any analysis since it's determined by the board so if we find a solution we know its going to have 
the max amount of knights.

Three of the tables, the 2x2, 3x3 and 3x5 have similar times. This is before 5x5 comes into action, this iteration suddenly increases 
the time substantially. This show how exponential the growth of ths algorithm is.
We also observe that the 8x8 is not done by any means in a resonable time.

The B&B algorithm is a great one for small simple problems or ones that have a very clear path (or some small amout of paths) with lower 
costs. When it comes to bigger or more generic problems it is very slow a costly and not really recomended.

**A\***

| Board | Algorithm | Time  | Horses |
|-------|-----------|-------|--------|
| 2x2   | A*        | 0.001 | 4      |
| 3x3   | A*        | 0.002 | 5      |
| 3x5   | A*        | 0.012 | 8      |
| 5x5   | A*        | 0.059 | 13     |
| 8x8   | A*        | >5min | >5min  |

Taking a first look at the table, we see that we have pretty good times for the algorithm.
The last one that this algorithm is able to accomplish is the 5x5 taking only 0.145 seconds, so we can conclude that this algorithm 
is satisfactorily fast and even if it still escalates very quickly looking at the long time 8x8 takes, we can see that the algorithm 
greatly improves from the B&B thanks to the inclusion of the heuristic value.

The A* algorithm is a much better and a bit more efficient one over B&B but it still struggles over complex or generic problems.
The efficiency of A* is greatly dependant on the precision of the heuristic value and really suffers over a bad aproximation of said value,
and adds the problem of proving it's addmisibility.

**Conclusions**

| **Board** | **Algorithm** | **Time B&B** | **Time A*** |
|:---------:|:-------------:|:------------:|:-----------:|
|    2x2    |  B&B and  A*  |     0.001    |    0.001    |
|    3x3    |  B&B and  A*  |     0.010    |    0.002    |
|    3x5    |  B&B and  A*  |     0.023    |    0.012    |
|    5x5    |  B&B and  A*  |     124.1    |    0.059    |
|    8x8    |  B&B and  A*  |     NONE     |     NONE    |

Here we have the two tables of B&B and A* combined in order to compare them.
As seen before A* is always a better option over B&B thanks to the addition of the heuristic value but it can be harder to implement in 
more complex settings where the calculation of a heuristic is not as simple.

**Authors**

This project is an addaptation and improvement from a practical assignment at the Universidad Politecnica de Madrid on 2024 and made by 4 people 
including myself:
*   Wuillermo (Guillermo Vergara)
*   [Jose Belchí](https://github.com/josebelchi)
*   [Alex](https://github.com/acorcobado)
*   Marco