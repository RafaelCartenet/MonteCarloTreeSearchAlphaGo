import numpy as np

# State class
# contains the board of the current State and a list of edges that lead to the
# next states.
class State(object):
    def __init__(self, board):
        # Board
        self.board = board

        # Actions
        self.edges = [] # List of Edge
        self.nbedges = 0

    # Chooses the best action according to Q value of each edge. + bonus u
    # PUCT algorithm cf lect22.
    def choose_action(self):
        PUCTs = np.zeros(self.nbedges)
        for i in range(self.nbedges):
            PUCTs[i] = self.edges[i].PUCT() # PUCT(s,a) = Q(s,a) + u(s,a). cf p10 lect22.
        return edges[np.argmax(PUCTs)]

    # Choose the best action according to Q value (Greedy)
    def choose_greedy_action(self):
        Qs = np.zeros(self.nbedges)
        for i in range(self.nbedges):
            Qs[i] = self.edges[i].Q # Q(s,a)
        return edges[np.argmax(Qs)]

    # bonus u(s,a) depends on s so needs to be updated from s.
    def update_u_edges(self):
        sumNr = 0
        for edge in self.edges:
            sumNr += edge.Nr
        for edge in self.edges:
            edge.update_u(sumNr)

    def is_leaf(self):
        return self.nbedges == 0


    # -------------- #
    # MCTS FUNCTIONS
    # -------------- #

    def expand(self):
    # expand a new node if the Ntrh (threshold is over nthr (40 in AlphaGo))
    # expansion is explained in lecture 22 page 13. Not really clear ...
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        return None

    def evaluate(self):
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #

    def rollout(self): # Rollout policy in order to determine a reward 1 0 or -1
    # according to the result of the game. policy = plays random move until the end.
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        return None # 1 0 or -1

    def valuenetwork(self): # uses the VAL function that uses the Value network
    # to predict win value.
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        return None # return result related to value network


# Edge class : Edge from state S with action A
class Edge:
    # GLOBAL VARIABLES
    self.lbda = 0.5
    self.cpuct = 5
    self.nthr = 40 # visit threshold

    def __init__(self, board):
        self.state = State(board) # Board
        self.player = None # Player performing action

        self.priorP = None # Prior probability for that edge

        # VALUE NETWORK
        self.Nv = 0     # Number of leaf evaluations using the value network
        self.Wv = 0     # Accumulation of values estimates using the value network

        # ROLLOUT POLICY
        self.Nr = 0     # Number of leaf evaluations using the rollout policy
        self.Wr = 0     # Accumulation of rewards using the rollout policy

        # Q VALUE
        self.Q = 0      # Combined estimate of the action value function using the
                        # value network and the rollout policy

        # UCT
        self.u = 0      # bonus u(P) for edge selection

        # Rough Prediction
        self.P = 0      # Can be probability from :
                        # - Tree policy         P_tau(a|s)
                        # - SL policy network   P_delta(a|s) <- AlphaGo
                        # - RL policy network   P_rau(a|s)

    def update_Q(self):
        qV = self.Wv / self.Nv
        qR = self.Wr / self.Nr
        self.Q = (1 - self.lbda) * qV + self.lbda * qR

    def update_u(self, sumNr):
        num = np.sqrt(sumNr)
        denom = 1 + self.Nr
        self.u = self.cpuct * self.P * num / denom

    def PUCT(self):
        return self.Q + self.u


    # -------------- #
    # MCTS FUNCTIONS
    # -------------- #

    # evaluate this action
    # evaluation takes in consideration two things :
    # - Value Network Estimate : v
    # - Rollout policy (random play until end of the game) with reward : r
    def evaluate(self):
        self.Nv += 1
        self.Nr += 1

        # ------ TO COMPLETE ------- #
        self.Nv += self.valuenetwork() # ESTIMATION THANKS TO VALUE NETWORK.
        self.Nr += self.rollout() # ESTIMATION THANKS TO ROLLOUT POLICY
        # -------------------------- #


class MCTree:
    def __init__(self, board):
        self.starting_Node = State(board)

    def selection(self, path = [], node = None):
        if node == None:
            node = self.starting_Node
        if not node.is_leaf():
            nextt = node.choose_action() # CHOOSE NEXT EDGE ACCORDING TO POLICY
            path.append(nextt.state) # Adds the state to the path.
            return self.selection(path, nextt.state)
        return path, node

    # Adds an aditionnal node from the selected node
    def expansion(self, node):
        return node.expand()

    # Simulates the end of the game and computes the result
    def simulation(self, node):
        return node.evaluate()

    # change the played / won value for every node in the path, according to the result
    def backprop(self, path, result, node = None):
        #if node == None: node = self.starting_Node
        #if not node.is_leaf():

        # I didn't really understand how they do it.
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #


    def MCTS(self):
        # Selects a leaf and returns the path from the root to node.
        path, node = self.selection()

        # Creates an aditional node from this leaf
        node = self.expansion(node)

        # Add the created node in the path
        path.append(0)

        # Simulate the game and returns the result.
        result = self.simulation(node)

        # Backpropagate the result of the simulation in the three through the path
        self.backprop(path, result)
