from Utils import get_last_k
from Bounds import Bounds

class DemandManager(object):

    def __init__(self, loads = None):
        if loads is None:
            loads = []
        self.loads = loads
        self.num_loads = 0
        self.current_demand = 0.0
        self.demands = []
        # self.min_demand = 999999.0
        # self.max_demand = 0.0
        self.demand_bounds = Bounds()

    def remove_load(self, loadID):
        try:
            self.loads.remove(loadID)
            self.num_loads-=1
            return True
        except:
            return False

    def add_loads(self, loadIDs):
        for loadID in loadIDs:
            self.add_load(loadID)

    def add_load(self,loadID):
        if loadID not in self.loads:
            self.loads.append(loadID)
            self.num_loads+=1
            return True
        else:
            return False

    def update_current_demand(self, demand):
        self.current_demand = demand
        self.demands.append(demand)

    def reset_day(self, look_ahead):
        if len(self.demands)>0:
            self.current_demand = self.demands[-1]

        self.demands = get_last_k(self.demands, look_ahead)

    def get_current_demand(self):
        return self.current_demand

    def get_previous_demand(self):
        if len(self.demands)<2:
            return 0.0
        else:
            return self.demands[-2]

    def get_demands(self):
        return self.demands

    def get_loads(self):
        return self.loads

    # def get_demand_bounds(self):
    #     return [self.min_demand, self.max_demand]
    #
    # def update_demand_bounds(self,demand):
    #     if not demand <= 0:
    #         self.min_demand = min(self.min_demand, demand)
    #     self.max_demand = max(self.max_demand, demand)