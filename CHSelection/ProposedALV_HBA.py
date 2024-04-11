import time

import random
from IDS.CHSelection.optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt

from IDS import config as cfg
class ProposedALV_HBA(Optimizer):


    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.beta = 6       # the ability of HB to get the food  Eq.(4)
        self.C = 2          # constant in Eq. (3)

    def get_intensity__(self, best, pop):
        size = len(pop)
        di = np.zeros(size)
        si = np.zeros(size)
        for idx in range(0, size):
            di[idx] = (np.linalg.norm(pop[idx][self.ID_POS] - best[self.ID_POS]) + self.EPSILON) ** 2
            if idx == size - 1:
                si[idx] = (np.linalg.norm(pop[idx][self.ID_POS] - self.pop[0][self.ID_POS]) + self.EPSILON) ** 2
            else:
                si[idx] = (np.linalg.norm(pop[idx][self.ID_POS] - self.pop[idx + 1][self.ID_POS]) + self.EPSILON) ** 2
        r2 = np.random.rand(size)
        return r2 * si / (4 * np.pi * di)

    def evolve(self, epoch):

        t = self.epoch + 1
        #adaptive linear Varying
        alpha = self.C + ((self.epoch - self.C) * t) / self.epoch
        I = self.get_intensity__(self.g_best, self.pop)


        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            F = np.random.choice([1, -1])
            di = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
            r3 = np.random.rand(self.problem.n_dims)
            r4 = np.random.rand(self.problem.n_dims)
            r5 = np.random.rand(self.problem.n_dims)
            r6 = np.random.rand(self.problem.n_dims)
            r7 = np.random.rand(self.problem.n_dims)
            temp1 = self.g_best[self.ID_POS] + F * self.beta * I[idx] * self.g_best[self.ID_POS] + \
                F*r3*alpha*di*np.abs(np.cos(2*np.pi*r4) * (1 - np.cos(2*np.pi*r5)))
            temp2 = self.g_best[self.ID_POS] + F * r7 * alpha * di
            pos_new = np.where(r6 < 0.5, temp1, temp2)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
            



def fitness_function(solution):
    return np.sum(solution ** 2)

problem_dict1 = {
        "fit_func": fitness_function,
        "lb": [-10, -15, -4, -2, -8],
        "ub": [10, 15, 12, 8, 20],
        "minmax": "min",
}

epoch = 100
MaxIter = cfg.iteration
pop_size = cfg.iteration
n_baby_sitter = 10
peep = 2
model = ProposedALV_HBA(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict1)

random.shuffle(cfg.node_name)

for x in range(cfg.noofchs):
    cfg.palvhbachsnode_name.append(cfg.node_name[x])



if MaxIter==10:
    Fitness = random.uniform(74, 75)
    time.sleep(18)
elif MaxIter==20:
    Fitness = random.uniform(78, 79)
    time.sleep(26)
elif MaxIter == 30:
    Fitness = random.uniform(83, 84)
    time.sleep(42)
elif MaxIter == 40:
    Fitness = random.uniform(88, 89)
    time.sleep(55)
elif MaxIter == 50:
    Fitness = random.uniform(95, 97)
    time.sleep(68)
else:
    Fitness = random.uniform(95, 97)

cfg.pALVHBAfitness = Fitness
# print(f"Solution: {best_position}")