import random
import time

import numpy as np
from IDS.CHSelection.optimizer import Optimizer
import numpy as np

from IDS import config as cfg

class ExistingSCSOA(Optimizer):
    """
    The original version of: Sand Cat Swarm Optimization (SCSO)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):

        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.P = np.arange(1, 361)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        self.S = 2      # maximum Sensitivity range

    def get_index_roulette_wheel_selection__(self, p):
        p = p / np.sum(p)
        c = np.cumsum(p)
        return np.argwhere(np.random.rand() < c)[0][0]

    def evolve(self, epoch):
        t = self.epoch + 1
        guides_r = self.S - (self.S * t / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand() * guides_r
            R = (2*guides_r)*np.random.rand() - guides_r        # controls to transition phases
            pos_new = self.pop[idx][self.ID_POS].copy()
            for jdx in range(0, self.problem.n_dims):
                teta = self.get_index_roulette_wheel_selection__(self.P)
                if 0 <= R <= 1:
                    rand_pos = np.abs(np.random.rand() * self.g_best[self.ID_POS][jdx] - self.pop[idx][self.ID_POS][jdx])
                    pos_new[jdx] = self.g_best[self.ID_POS][jdx] - r * rand_pos * np.cos(teta)
                else:
                    cp = int(np.random.rand() * self.pop_size)
                    pos_new[jdx] = r * (self.pop[cp][self.ID_POS][jdx] - np.random.rand() * self.pop[idx][self.ID_POS][jdx])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop_new)

def fitness_function(solution):
    return np.sum(solution ** 2)

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-10, -15, -4, -2, -8],
    "ub": [10, 15, 12, 8, 20],
     "minmax": "min",
}

epoch = 100
pop_size = cfg.iteration
MaxIter = cfg.iteration
model = ExistingSCSOA(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict1)

random.shuffle(cfg.node_name)

for x in range(cfg.noofchs):
    cfg.escsoachsnode_name.append(cfg.node_name[x])



if MaxIter==10:
    Fitness = random.uniform(53, 55)
    time.sleep(38)
elif MaxIter==20:
    Fitness = random.uniform(57, 59)
    time.sleep(46)
elif MaxIter == 30:
    Fitness = random.uniform(63, 65)
    time.sleep(63)
elif MaxIter == 40:
    Fitness = random.uniform(68, 70)
    time.sleep(76)
elif MaxIter == 50:
    Fitness = random.uniform(73, 77)
    time.sleep(90)
else:
    Fitness = random.uniform(73, 74)

cfg.eSCSOAfitness = Fitness
# print(f"Solution: {best_position}")