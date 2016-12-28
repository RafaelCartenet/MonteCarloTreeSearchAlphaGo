import numpy as np
import tensorflow as tf
import strategy

nx = 5
ny = 5

# State class
# contains the board of the current State and a list of edges that lead to the
# next states.
class State(object):
    def __init__(self, board, state, player):
        # Board
        self.board = board
        if state == None:
            state = -np.ones([2,1])
        else:
            self.state = state

        # Player turn
        self.player = player # equals 1 if black. 2 if white

        # Actions
        self.edges = [] # List of Edge
        
        #initialise edges, otherwise no function
        free_places = nx * ny - self.state.count(1) - self.state.count(2)
        for p1 in range(free_places):
            vm1, b1, state1 = next_move.valid(self.board, self.state, next_move.xy(p1), self.player)
            n_valid_moves += vm1
            valid_moves[:, p1] = vm1
            # check only the next move
            idx = nrange(1) + p1 
            d[:, :, 0, idx] = (b1 == 1)
            d[:, :, 1, idx] = (b1 == 2)
            d[:, :, 2, idx] = (b1 == 0)
            # TODO what do we do with the boards, create Edges with them, (need the state variable for this?
        self.nbedges = 0



        # Value Network ESTIMATION
        self.ValNetOutput = None # equals to None when not estimated yet.

    # Chooses the best action according to Q value of each edge. + bonus u
    # PUCT algorithm cf lect22.
    def choose_action(self):
        PUCTs = np.zeros(self.nbedges)
        for i in range(self.nbedges):
            PUCTs[i] = self.edges[i].PUCT(self.board) # PUCT(s,a) = Q(s,a) + u(s,a). cf p10 lect22.
        return edges[np.argmax(PUCTs)]

    # Choose the best action according to Q value (Greedy)
    def choose_greedy_actionQ(self):
        Qs = np.zeros(self.nbedges)
        for i in range(self.nbedges):
            Qs[i] = self.edges[i].Q # Q(s,a)
        return edges[np.argmax(Qs)]

    # Choose the best action according to Nr value (Greedy)
    def choose_greedy_actionNr(self):
        Nrs = np.zeros(self.nbedges)
        for i in range(self.nbedges):
            Nrs[i] = self.edges[i].Nr # Nr(s,a)
        return edges[np.argmax(Nrs)]


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
    # expand a new node thanks to policy network !

        # estimates next action thanks to policy network
        sess = tf.get_default_session()
        softout = np.zeros(nx*ny)
        feed_dict = {"current board": board}
        if self.player == 1:
            softout[:] = sess.run(netP1, feed_dict = feed_dict)
            self.ValNetOutput = softout[1]
        else:
            softout[:] = sess.run(netP2, feed_dict = feed_dict)
            self.ValNetOutput = softout[1]

        # initialize new board
        new_board = np.zeros(self.board.shape)
        new_board[:, :] = board[:, :]
        new_state = np.zeros(self.state.shape)
        new_state[:] = self.state[:]

        # choose best move according to result of policy network
        temp_softout = softout[:]/np.sum(softout[:])
        n_valid_moves = 0
        while True:
            cum_softout = np.cumsum(temp_softout)
            softr = rand(1)
            softxy = np.sum(softr>cum_softout)
            rxy = np.array(self.xy(softxy))
            rj = int(rxy[0][0])
            rk = int(rxy[0][1])
            isvalid, _, _ = self.valid(b[:, :, state[:], rxy, self.player)
            if int(isvalid[0]):
                isvalid, bn, sn = self.valid(b[:, :], state[:], rxy, self.player)
                new_board[:, :] = bn
                new_state[:] = sn
                x = rj
                y = rk
                n_valid_moves += 1
                break
            else:
                temp_softout[softxy] = 0
                norm = np.sum(temp_softout)
                if norm != 0:
                    temp_softout = temp_softout/norm
                else:
                    break
        if not int(isvalid[0]):
            isvalid, bn, sn =\
                self.valid(b[:, :], state[:], -np.ones((1, 2)), self.player)
            new_state[:] = sn

        #TODO is a new class instance needed every time?
        return STATE(new_board,new_state,3-self.player)


    def rollout(self): # Rollout policy in order to determine a reward 1 0 or -1
    # according to the result of the game. policy = plays random move until the end.
        
        #b board
        # state = state? which player played where??? initialised to all -1
        # 2x1 vector where each vector is reserverd for one player and saves the position of the moves
        
        if self.player == 1:
            k = self.state.count(1)
            b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                strategy.val(self.board, self.state, 1, [], 1, self.player, k)
        else:
            k = self.state.count(2)
            b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                strategy.val(self.board, self.state, 1, [], 1, self.player, k)
        # k = number of moves, check in one of the states , how many elements are != -1

        while n_valid_moves > 0:
            np0 = n_valid_moves
            k = k + 1
            # 'm' possible board configurations
            m = np0
            d = np.zeros((nx, ny, 3, m))
            pos = np.zeros((nx,ny,m))
            
            # Check whether tie(0)/black win(1)/white win(2) in all board configurations
            w = np.zeros((m))

            # winning probability: (no work for 1st generation)       
            wp = np.zeros((m))

            # Check whether the configurations are valid for training
            valid_data = np.zeros((m))

            # Check whether the configurations are played by player 1 or player 2
            turn = np.zeros((m))

            # number of valid moves in previous time step
            vm0 = np.ones((1))

            # maximum winning probability for each game
            wp_max = np.zeros((1))

            # For each time step, check whether game is in progress or not.
            game_in_progress = np.ones((1))

            # First player: player 1 (black)
            p = 1

            for k in range(np0):
                if p == 1:
                    b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                        strategy.val(b, state, game_in_progress, netV1, r1, p, k)
                else:
                    b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                        strategy.val(b, state, game_in_progress, netV2, r2, p, k)
                
                w0, end_game, _, _ = strategy.winner(b, state)
                idx = nrange(k , (k + 1) )
                d[:, :, 0, idx] = (b == 1)
                d[:, :, 1, idx] = (b == 2)
                d[:, :, 2, idx] = (b == 0)
                
                wp[idx] = wp_max
                valid_data[idx] = game_in_progress * (n_valid_moves > 0)
                
                # information about who's the current player
                turn[idx] = p
                if k > 0:
                    for i in range(1):
                        if x_pos[i] >= 0:
                            pos[int(x_pos[i]),int(y_pos[i]),(k-1)+i] = 1
    
                game_in_progress *=\
                        ((n_valid_moves > 0) * (end_game == 0) +\
                        ((vm0 + n_valid_moves) > 0) * (end_game == -1))


                if game_in_progress == 0:
                    break

                p = 3 - p
                vm0 = n_valid_moves[:]

        for k in range(np0):
            idx = nrange(k, (k + 1))
            w[idx] = w0[:] # final winner

        # player 1's stat
        win = np.sum(w0 == 1) / float(1)
        loss = np.sum(w0 == 2) / float(1)
        tie = np.sum(w0 == 0) / float(1)
        
        if win > loss and win > tie:
            return 1
        elif tie > win and tie > loss:
            return 0
        elif loss > win and loss > tie:
            return -1
        
        #TODO is this the correct assignment??????

    def valuenetwork(self): # uses the VAL function that uses the Value network
    # to predict win value.
    # SHOULD BE CALLED ONLY ONCE PER NODE (if not saving the result might be good)
    # use the value function of strategy.py ?
    #
    # Indeed : if never computed before, estimate it thanks to val net, otherwise
    # just returns the value.
        if self.ValNetOutput == None:
            sess = tf.get_default_session()
            softout = np.zeros(3)
            feed_dict = {"current board": board}
            if self.player == 1:
                softout[:] = sess.run(netV1, feed_dict = feed_dict)
            else:
                softout[:] = sess.run(netV2, feed_dict = feed_dict)
            self.ValNetOutput = softout[1]

        return self.ValNetOutput # return result related to value network

    def evaluate(self):
    # I guess this function is about sampling some possible actions, a finite number of them,
    # and evaluate them using rollout and value net. (That should use the function edge.evaluate())
        V = self.valuenetwork()
        R = self.rollout()
        return V, R

# Edge class : Edge from state S with action A
class Edge:
    # GLOBAL VARIABLES
    self.lbda = 0.5
    self.cpuct = 5
    self.nthr = 10 # visit threshold, AlphaGo had 40 here use lower number (10 is arbitrary)

    def __init__(self, board):
        self.board = board # Board TODO has to save a State call instance
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

    # executes a rough prediction of this action.
    # forwards the board
    def roughprediction(self):

    # -------------- #
    # MCTS FUNCTIONS
    # -------------- #

    # executes the steps needed in the backprop algorithm
    # do we reinitialise the values of Nv and Wv -> no, the leaf Node will automatically be initialised by the EVALUATE step
    def backprop_update(self, nodePrevious):

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
        self.Wv += board.valuenetwork() # ESTIMATION FROM THE VALUE NETWORK (current Node is the starting point)
        self.Wr += board.rollout() # ESTIMATION FROM THE ROLLOUT POLICY (current Node is the starting point)
        # -------------------------- #
        return 1
        #TODO update the result of the evaluation for the updating of the tree in the backprop
        # -> did the change to Wv and Wr do this?



class MCTree:
    def __init__(self, board, state):
        self.starting_Node = State(board, state)# starting_Node is the tree head


    # select one path from rootNode until a leaf node of the newly constructed monteCarloTree
    def selection(self, path = [], node = None):
        if node == None:
            node = self.starting_Node
        if not node.is_leaf():
            nextt = node.choose_action() # CHOOSE NEXT EDGE ACCORDING TO POLICY
            path.append(nextt.board) # Adds the state to the path.
            return self.selection(path, nextt.board)
        return path, node

    # Adds an aditional node C from the selected node
    def expansion(self, node):
        # add Node to MCTree if a certain number of visits to an !EDGE! is reached
        # TODO node is selected and a new State instance is created, inserted into the tree??
        result = node.expand()
        return result

    # Simulates from the Node C which was !EXPANDED! until the end of the game and computes the result
    def simulation(self, node):
        node.evaluate() # no return, since function updates the tree by itself
        return 1

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
        path.append(node)

        # Simulate the game until the end
        self.simulation(node)

        # Backpropagate the result of the simulation in the three through the path
        self.backprop(path)
