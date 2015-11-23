from random import randint, gauss

import numpy as np

class DiscreteHill(object):

    directions = [(0,1), (0,-1), (1,0), (-1,0)]

    def __init__(self, board=(10,10), variance=4.):
        self.variance = variance
        self.target = (0,0)
        while self.target == (0,0):
            self.target   = (randint(-board[0], board[0]), randint(-board[1], board[1]))
        self.position = (0,0)

        self.shortest_path = self.distance(self.position, self.target)

    @staticmethod
    def add(p, q):
        return (p[0] + q[0], p[1] + q[1])

    @staticmethod
    def distance(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    def estimate_distance(self, p):
        distance = DiscreteHill.distance(self.target, p) - DiscreteHill.distance(self.target, self.position)
        return distance + abs(gauss(0, self.variance))

    def observe(self):
        return np.array([self.estimate_distance(DiscreteHill.add(self.position, delta))
                         for delta in DiscreteHill.directions])

    def perform_action(self, action):
        self.position = DiscreteHill.add(self.position, DiscreteHill.directions[action])

    def is_over(self):
        return self.position == self.target

    def collect_reward(self, action):
        return -DiscreteHill.distance(self.target, DiscreteHill.add(self.position, DiscreteHill.directions[action])) \
            + DiscreteHill.distance(self.target, self.position) - 2
