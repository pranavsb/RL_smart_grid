from DemandManager import DemandManager
from DemandRange import DemandRange
from Utils import get_last_k
from random import randint
from Bounds import Bounds

class Source(DemandManager):


    num_sources = 0
    action_space = [0,1,2,3,4]
    base_pricing_constant = 10.0
    action_price_map = {0:1.0, 1:0.7, 2:0.3, 3:1.5, 4: 2.5}
    no_agent_action = 0

    def __init__(self, sourceID = None, capacity = 10000.0, current_price = 0.0, with_agent = False, look_ahead = 1):

        super().__init__()

        if sourceID is None:
            sourceID = Source.num_sources

        self.sourceID = sourceID
        self.supply_capacity = capacity        # MW
        self.current_price = current_price
        self.prices = []
        self.with_agent = with_agent
        self.look_ahead = look_ahead
        self.price_bounds = Bounds()
        # self.min_price = 100000.0
        # self.future_max = 0.0
        # self.future_min = 999999.9
        # self.update_count = 0
        # self.max_price = 0.0
        self.dumb_load_range = DemandRange(0.0,0.0)
        Source.num_sources +=1

    def reset_day(self):
        super().reset_day(self.look_ahead)
        if len(self.prices)>0:
            self.current_price = self.prices[-1]

        self.prices = get_last_k(self.prices, self.look_ahead)

    def step(self, action=None):

        if action is None:
            action = Source.no_agent_action

        # if not self.with_agent:
        #     action = Source.no_agent_action

        if action not in Source.action_space:
            raise AssertionError("Not a valid action")

        self.prices.append(self.calculate_base_price() * 1.0 * Source.action_price_map.get(action))
        # print('priced %f times original price with action %d' % (Source.action_price_map[action], action))
        self.current_price = self.prices[-1]
        self.price_bounds.update_bounds(self.current_price)
        self.demand_bounds.update_bounds(self.current_demand)
        return self.get_previous_price() * super().get_current_demand()


    def calculate_base_price(self):

        if self.get_current_demand()< 0.9*self.supply_capacity:
            return Source.base_pricing_constant * super().get_previous_demand() / self.num_loads
        else:
            return 1 * Source.base_pricing_constant * super().get_previous_demand() / self.num_loads

    def set_look_ahead(self, look_ahead):
        self.look_ahead = look_ahead

    def get_look_ahead(self):
        return self.look_ahead

    def get_prices(self):
        return self.prices

    def get_current_price(self):
        return self.current_price

    def get_previous_price(self):
        if len(self.prices)<2:
            return 0.0
        else:
            return self.prices[-2]

    def is_with_agent(self):
        return self.with_agent

    def get_supply_capacity(self):
        return self.supply_capacity

    def add_load(self,loadID):
        flag = super().add_load(loadID)
        if flag:
            print('Load %d successfully linked to source %d' % (loadID, self.sourceID))
        else:
            print('Load %d already handled by source %d' % (loadID, self.sourceID))

    def remove_load(self, loadID):
        flag = super().remove_load(loadID)
        if flag:
            print('Load %d successfully removed from source %d' % (loadID, self.sourceID))
        else:
            print('Load %d is not handled by source %d' % (loadID, self.sourceID))

    def sample_action(self):
        return Source.action_space[randint(0,len(Source.action_space))]

    # def update_price_bounds(self,price):
    #     if not price <= 0:
    #         self.future_min = min(self.future_min, price)
    #     self.future_max = max(self.future_max, price)
    #     self.update_count+=1
    #     if self.update_count+1 % 1000 == 0:
    #         self.update_count = 0
    #         self.min_price = self.future_min
    #         self.max_price = self.future_max
    #         self.future_max = 0.0
    #         self.future_min = 99999.9
    #
    # def get_price_bounds(self):
    #     return [self.min_price, self.max_price]

    def add_dumb_load_range(self, lowerbound, upperbound):
        self.dumb_load_range.set_lower_bound(self.dumb_load_range.get_lower_bound()+lowerbound)
        self.dumb_load_range.set_upper_bound(self.dumb_load_range.get_upper_bound()+upperbound)

    def remove_dumb_load_range(self, lowerbound, upperbound):
        self.dumb_load_range.set_lower_bound(self.dumb_load_range.get_lower_bound()-lowerbound)
        self.dumb_load_range.set_upper_bound(self.dumb_load_range.get_upper_bound()-upperbound)

    def add_dumb_loads(self,n=None, ranges = None):
        if n is None:
            for demandrange in ranges:
                self.add_dumb_load_range(demandrange[0], demandrange[1])
            self.num_loads+=len(ranges)
            return

        if ranges is None:
            ranges = [[DemandRange.default_lower_bound, DemandRange.default_upper_bound]]

        numranges = len(ranges)
        for i in range(n):
            if i < numranges:
                self.add_dumb_load_range(ranges[i][0], ranges[i][1])
            else:
                self.add_dumb_load_range(ranges[-1][0], ranges[-1][1])
        self.num_loads += n
