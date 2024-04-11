import numpy as np
from IDS.CHSelection.logger import Logger

class Problem:

    SUPPORTED_ARRAY = (list, tuple, np.ndarray)

    def __init__(self, lb=None, ub=None, minmax="min", **kwargs):
        r"""Initialize Problem.

        Args:
            lb (numpy.ndarray, list, tuple): Lower bounds of the problem.
            ub (numpy.ndarray, list, tuple): Upper bounds of the problem.
            minmax (str): Minimization or maximization problem (min, max)
            name (str): Name for this particular problem
        """
        self.name, self.log_to, self.log_file = "P", "console", "history.txt"
        self.n_objs, self.obj_is_list, self.multi_objs, self.obj_weights = 1, False, False, None
        self.n_dims, self.lb, self.ub, self.save_population = None, None, None, False

        self.__set_keyword_arguments(kwargs)
        self.__set_domain_range(lb, ub)
        self.__set_functions(kwargs)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
                                    format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.minmax = minmax

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_domain_range(self, lb, ub):
        if type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            self.lb = np.array(lb).flatten()
            self.ub = np.array(ub).flatten()
            if len(self.lb) == len(self.ub):
                self.n_dims = len(self.lb)
                if len(self.lb) < 1:
                    raise ValueError(f'Dimensions do not qualify. Length(lb) = {len(self.lb)} < 1.')
            else:
                raise ValueError(f"Length of lb and ub do not match. {len(self.lb)} != {len(self.ub)}.")
        else:
            raise ValueError(f"lb and ub need to be a list, tuple or np.array.")

    def __set_functions(self, kwargs):
        tested_solution = self.generate_position(self.lb, self.ub)
        if "amend_position" in kwargs:
            if not callable(self.amend_position):
                raise ValueError(f"Use default 'amend_position()' or passed a callable function. {type(self.amend_position)} != function")
            else:
                tested_solution = self.amend_position(tested_solution, self.lb, self.ub)
        result = self.fit_func(tested_solution)
        if type(result) in self.SUPPORTED_ARRAY:
            result = np.array(result).flatten()
            self.n_objs = len(result)
            self.obj_is_list = True
            if self.n_objs > 1:
                self.multi_objs = True
                if type(self.obj_weights) in self.SUPPORTED_ARRAY:
                    self.obj_weights = np.array(self.obj_weights).flatten()
                    if self.n_objs != len(self.obj_weights):
                        raise ValueError(f"{self.n_objs}-objective problem, but N weights = {len(self.obj_weights)}.")
                    self.msg = f"Solving {self.n_objs}-objective optimization problem with weights: {self.obj_weights}."
                else:
                    raise ValueError(f"Solving {self.n_objs}-objective optimization, need to set obj_weights list with length: {self.n_objs}")
            elif self.n_objs == 1:
                self.multi_objs = False
                self.obj_weights = np.ones(1)
                self.msg = f"Solving single objective optimization problem."
            else:
                raise ValueError(f"fit_func needs to return a single value or a list of values list")
        elif type(result) in (int, float) or isinstance(result, np.floating) or isinstance(result, np.integer):
            self.multi_objs = False
            self.obj_is_list = False
            self.obj_weights = np.ones(1)
            self.msg = f"Solving single objective optimization problem."
        else:
            raise ValueError(f"fit_func needs to return a single value or a list of values list")

    def fit_func(self, x):
        """Evaluate solution.

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.
        """
        raise NotImplementedError

    def __call__(self, x):
        r"""Evaluate solution.

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.

        See Also:
            :func:`niapy.problems.Problem.evaluate`

        """
        return self.fit_func(x)

    def get_name(self):
        return self.name

    def get_class_name(self):
        """Get class name."""
        return self.__class__.__name__

    def generate_position(self, lb=None, ub=None):
        """
        Generate the position depends on the problem. For discrete problem such as permutation, this method can be override.

        Args:
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            np.array: the position (the solution for the problem)
        """
        return np.random.uniform(lb, ub)

    def amend_position(self, position=None, lb=None, ub=None):
        """
        This is default function in most algorithms. Otherwise, there will be an overridden function
        in child of Optimizer class for this function. Depend on what kind of problem are we trying to solve,
        there will be a different amend_position function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        # return np.maximum(self.problem.lb, np.minimum(self.problem.ub, position))
        return np.clip(position, lb, ub)
