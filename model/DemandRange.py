# 11 Mar, 2018

import random

class DemandRange(object):
    """ DemandRange -
    Member Variables -
        LowerBound - in kW
        UpperBound - in kW
    Member Functions -
        GenerateDemand() - generates demand according to gaussian or uniform distribution (All demands in terms of power)
        Getters and Setters and Constructor
    """

    default_lower_bound = 0.5
    default_upper_bound = 1.5

    def __init__(self, lower_bound=None, upper_bound=None):
        #super(DemandRange, self).__init__()
        if lower_bound is None:
            lower_bound = DemandRange.default_lower_bound
        if upper_bound is None:
            upper_bound = DemandRange.default_upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate_demand(self):
        return random.random()*(self.upper_bound - self.lower_bound) + self.lower_bound

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def set_lower_bound(self, lower_bound):
        self.lower_bound = lower_bound

    def set_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound
