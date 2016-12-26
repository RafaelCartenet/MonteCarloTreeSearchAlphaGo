## MonteCarloTreeSearchAlphaGo

# Files
mcts.py is a simple implementation of monte carlo tree search

mctsgo.py is a skeleton for montecarlo tree search related to the implementation given. It has to be in the future implemented within the strategy.py


# Details

I did the skeleton but some functions have to be completed. I did it according to the tutorials i read online and the lectures. I don't understand everything so that's why it's still containing a lot of blanks.

The structure i used for this is :

class State:
represents a board, and a list of edges.

class Edge: 
represents an action. The original state is not contained in the Edge class. The class contains the state after the action though. It's for me the most logical way to represent it but it could be subject to change.

class MCTree:
represents the tree. Only contains the starting Node and the montecarlo tree search functions.
