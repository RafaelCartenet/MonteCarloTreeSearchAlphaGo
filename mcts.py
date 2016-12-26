
from numpy.random import choice as rchoice
from numpy import arange as nrange
from numpy.random import rand

nID = 0

class Node(object):
    def __init__(self):
        global nID
        self.nexts = []
        self.played = 0
        self.won = 0
        self.lvl = 0
        self.ID = nID
        nID += 1

    def is_leaf(self):
        return self.nexts == []

    def expand(self):
        expansion = Node()
        expansion.lvl = self.lvl + 1
        self.nexts.append(expansion)
        return expansion

    def simulate(self):
        self.played += 1
        if rand() < 0.5:
            self.won += 1
            return 1

    def toStr(self):
        print("Nde "+str(self.ID)+" | "+str(self.lvl) + " " + str(self.won)+"/"+str(self.played)+" "+str(len(self.nexts)))
        for nextt in self.nexts:
            nextt.toStr()

class Tree:
    def __init__(self, Node):
        self.starting_Node = Node
        self.starting_Node.simulate()
        Node.lvl = 0

    # Selects a random leaf
    def selection(self, path = [], node = None):
        if node == None: node = self.starting_Node
        if rand() < 0.15*node.lvl:
            return path, node
        if not node.is_leaf():
            nextt = rchoice(nrange(len(node.nexts)))
            path.append(nextt)
            return self.selection(path, node.nexts[nextt])
        return path, node

    # Adds an aditionnal node from the selected node
    def expansion(self, node):
        return node.expand()

    # Simulates the end of the game and computes the result
    def simulation(self, node):
        return node.simulate()

    # change the played / won value for every node in the path, according to the result
    def backprop(self, path, result, node = None):
        if node == None: node = self.starting_Node
        if not node.is_leaf():
            node.played += 1
            if result == 1:
                node.won += 1
            if path != []:
                n = node.nexts[path[0]]
                del path[0]
                self.backprop(path, result, n)

    def toStr(self):
        self.starting_Node.toStr()

    # Monte Carlo Tree Search
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


a = Node()
T = Tree(a)

for _ in range(100):
    T.MCTS()

T.toStr()
