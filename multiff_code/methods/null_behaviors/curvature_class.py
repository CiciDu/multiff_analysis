

import math
import numpy as np


class CurvatureOfPath():
    def __init__(self, ff_dataframe, monkey_information, ff_caught_T_new, ff_real_position_sorted, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.ff_dataframe = ff_dataframe[ff_dataframe.ff_angle_boundary.between(
            -45*math.pi/180, 45*math.pi/180)]
        self.monkey_information = monkey_information
        self.ff_caught_T_new = ff_caught_T_new
        self.ff_real_position_sorted = ff_real_position_sorted
        self.seed = seed
