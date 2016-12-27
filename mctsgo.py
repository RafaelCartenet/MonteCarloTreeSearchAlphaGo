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
    # add the edges of the often (more than a threshold value) visited node to the search tree
        for i in edges:
            if i.Nr > i.nthr:
                # add the edges of the state the edge points to, by this we have effectively added the node 
                # i.state  = state to add to the search tree, edge to that state exists already
                i.state.edges.append(Edge() ) #append the edges of the newly added node to enlarge search tree
                # TODO where do we get the edges from, update_u_edges???

        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # ------DONE??-------------- #
        # --break after first edge?- #
        # -------------------------- #
        return None

    def evaluate(self, starting_Node):
    # I guess this function is about sampling some possible actions, a finite number of them,
    # and evaluate them using rollout and value net. (That should use the function edge.evaluate())
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #

    def rollout(self, starting_Node): # Rollout policy in order to determine a reward 1 0 or -1
    # according to the result of the game. policy = plays random move until the end.
    # use structure from go_train_value to get the fully random game evaluation
    #game = game1()
        r_all = np.ones((n_train)) # random moves for all games
            [d1, w1, wp1, d2, w2, wp2] = game.play_games([], [],\
                    r_all, [], [], r_all, n_train, nargout = 6)
        #   w_black: nb1*1, 0: tie, 1: black wins, 2: white wins
        #   wp_black: win probabilities for black
        #   d_white: 4-d matrix of size nx*ny*3*nb2 containing all moves by white
        #   w_white: nb2*1, 0: tie, 1: black wins, 2: white wins
        #   wp_white: win probabilities for white
        # ------ TO COMPLETE ------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        # -------------------------- #
        return None # 1 0 or -1

    def valuenetwork(self): # uses the VAL function that uses the Value network
    # to predict win value.
    # SHOULD BE CALLED ONLY ONCE PER NODE (if not saving the result might be good)
    # use the value function of strategy.py ?
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

    # executes the steps needed in the backprop algorithm
    # do we reinitialise the values of Nv and Wv -> no, the leaf Node will automatically be initialised by the EVALUATE step
    def backprop_update(self,nodePrevious):
        
        #no parallel processing -> commented out
        #zt = 1 # random value chosen, since no parallel processing in order
        #self.Nr = self.Nr + 1 # virtual loss applied to discourage evaluation by parallel threads
        #self.Wr = self.Wr + zt # virtual loss applied to discourage evaluation by parallel threads
        
        self.Nv = self.Nv + 1
        if nodePrevious.isLeaf(): # we only have to evaluate the winning probabilities for the new leaf node, the rest is propagated
            self.Wv = self.Wv + nodePrevious.valuenetwork() # add the value network estimate of the new leaf Node
        else:
            self.Wv = self.Wv + nodePrevious.Wv
            
        self.update_Q()

    def getBackpropVal(self):
        return [Q,Nv,Wv]
        


    # evaluate this action
    # evaluation takes in consideration two things :
    # - Value Network Estimate : v
    # - Rollout policy (random play until end of the game) with reward : r
    def evaluate(self):
        self.Nv += 1 
        self.Nr += 1 
        # ------ TO COMPLETE ------- #
        self.Wv += self.valuenetwork() # ESTIMATION FROM THE VALUE NETWORK (current Node is the starting point)
        self.Wr += self.rollout() # ESTIMATION FROM THE ROLLOUT POLICY (current Node is the starting point)
        # -------------------------- #
        
        #TODO update the result of the evaluation for the updating of the tree in the backprop
        # -> did the change to Wv and Wr do this?



class MCTree:
    def __init__(self, board):
        self.starting_Node = State(board)# starting_Node is the tree head 
        
        # path holds all the information needed for the backpropagation
        # add availability of setting the values calculated in backprop (needs to be able to pass the values up the tree!!!!)
        
        
        # path only contains only the selected path in one selection step -> board state at the end of the path end
    # select one path from rootNode until a leaf node of the newly constructed monteCarloTree
    def selection(self, path = [], node = None):
        if node == None:
            node = self.starting_Node
        if not node.is_leaf():
            nextt = node.choose_action() # CHOOSE NEXT EDGE ACCORDING TO POLICY
            path.append(nextt.state) # Adds the state to the path.
            return self.selection(path, nextt.state)
        return path, node

    # Adds an aditional node C from the selected node
    def expansion(self, node):
        # add Node to MCTree if a certain number of visits to an !EDGE! is reached
        result = node.expand()
        
        return

    # Simulates from the Node C which was !EXPANDED! until the end of the game and computes the result
    def simulation(self, node):
        return node.evaluate()

    # change the played / won value for every node in the path from C to the root, according to the result
    def backprop(self, path):
        previous = len(path) - 1
        for i in reversed(range(len(path))):
            # values in the leaf node already initialised 
            path[i].backprop_update(path[previous])
            previous = i


    def MCTS(self):
        # Selects a leaf and returns the path from the root to node.
        path, node = self.selection()

        # Creates an aditional node from this leaf
        node = self.expansion(node)

        # Add the created node in the path
        path.append(0)

        # Simulate the game until the end
        self.simulation(node)

        # Backpropagate the result of the simulation in the three through the path
        self.backprop(path)
