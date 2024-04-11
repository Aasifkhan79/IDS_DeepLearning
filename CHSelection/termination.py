import time
from IDS.CHSelection import validator
from IDS.CHSelection.logger import Logger

class Termination:

    SUPPORTED_TERMINATIONS = {
        "FE": ["Function Evaluation", [10, 1000000000]],
        "ES": ["Early Stopping", [1, 1000000]],
        "TB": ["Time Bound", [1, 1000000]],
        "MG": ["Maximum Generation", [1, 1000000]],
    }

    def __init__(self, mode="FE", quantity=10000, **kwargs):
        self.mode, self.quantity, self.name = None, None, None
        self.exit_flag, self.message, self.log_to, self.log_file = False, "", None, None
        self.__set_keyword_arguments(kwargs)
        self.__set_termination(mode, quantity)
        self.logger = Logger(self.log_to, log_file=self.log_file).create_logger(name=f"{__name__}.{__class__.__name__}",
            format_str='%(asctime)s, %(levelname)s, %(name)s [line: %(lineno)d]: %(message)s')
        self.logger.propagate = False

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set_termination(self, mode, quantity):
        if validator.is_str_in_list(mode, list(Termination.SUPPORTED_TERMINATIONS.keys())):
            self.mode = mode
            self.name = Termination.SUPPORTED_TERMINATIONS[mode][0]
            if type(quantity) in (int, float):
                qt = int(quantity)
                if validator.is_in_bound(qt, Termination.SUPPORTED_TERMINATIONS[mode][1]):
                    self.quantity = qt
                else:
                    raise ValueError(f"Mode: {mode}, 'quantity' is an integer and should be in range: {Termination.SUPPORTED_TERMINATIONS[mode][1]}.")
            else:
                raise ValueError(f"Mode: {mode}, 'quantity' is an integer and should be in range: {Termination.SUPPORTED_TERMINATIONS[mode][1]}.")
        else:
            raise ValueError("Supported termination mode: FE (function evaluation), TB (time bound), ES (early stopping), MG (maximum generation).")

    def get_name(self):
        return self.name

    def get_default_counter(self, epoch):
        if self.mode in ["ES", "FE"]:
            return 0
        elif self.mode == "TB":
            return time.perf_counter()
        else:
            return epoch

    def is_finished(self, counter):
        if counter >= self.quantity:
            return True
        return False
