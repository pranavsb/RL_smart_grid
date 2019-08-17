# 11 Mar, 2018
import math
from Bounds import Bounds

class Battery(object):
    """
    Member Variables
        CurrentBatteryPercentage - FLOAT value from 0 to 100
        Battery Capacity - Total energy it can store - FLOAT value in kWh
        Update Battery Percentage - takes current energy demand and suitably updates battery percentage
        Charging rate kW
    Member functions
        Getters and Setters and Constructor

    """
    HISTORY_FACTOR = .999 # as ln(.9)/ln(.01) ~= 44

    def __init__(self, current_battery_percentage=100.0, battery_capacity=5.0, charging_rate=1.0):
        #super(Battery, self).__init__()
        self.current_battery_percentage = current_battery_percentage
        self.battery_capacity = battery_capacity
        self.charging_rate = charging_rate
        self.mean_battery = 50
        self.variance_battery = 0
        # self.min_mean_battery = 999999.9
        # self.max_mean_battery = -999999.9
        # self.min_variance_battery = 999999.9
        # self.max_variance_battery = -999999.9
        self.mean_battery_bounds = Bounds()
        self.variance_battery_bounds = Bounds(reset_count=300)

    def get_battery_bounds(self):
        return [self.mean_battery_bounds.get_bounds(), self.variance_battery_bounds.get_bounds()]

    def update_battery_bounds(self, mean, var):
        # self.min_mean_battery = min(mean, self.min_mean_battery)
        # self.max_mean_battery = max(mean, self.max_mean_battery)
        # self.min_variance_battery = min(var, self.min_variance_battery)
        # self.max_variance_battery = max(var, self.max_variance_battery)
        self.mean_battery_bounds.update_bounds(mean)
        self.variance_battery_bounds.update_bounds(math.pow(var,0.5))

    def get_current_battery_percentage(self):
        return self.current_battery_percentage

    def set_current_battery_percentage(self, current_battery_percentage):
        self.current_battery_percentage = min(100,max(0,current_battery_percentage))
        self.mean_battery += (1 - self.HISTORY_FACTOR) * (current_battery_percentage - self.mean_battery)
        self.variance_battery += (1 - self.HISTORY_FACTOR) * ((current_battery_percentage - self.mean_battery)**2 - self.variance_battery)
        self.update_battery_bounds(self.mean_battery, self.variance_battery)

    def update_battery_percentage(self, energy_demand):
        self.current_battery_percentage -= ((energy_demand / self.battery_capacity) * 100)

    def get_battery_capacity(self):
        return self.battery_capacity

    def get_charging_rate(self):
        return self.charging_rate

    def set_charging_rate(self, charging_rate):
        self.charging_rate = charging_rate

    def get_battery_reward_factor(self):
        return 0.5/(1+math.exp((-self.current_battery_percentage+40)/10))+1

    def get_mean_battery(self):
        return self.mean_battery

    def get_variance_battery(self):
        return math.pow(self.variance_battery,0.5)