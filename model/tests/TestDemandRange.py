# 11 Mar, 2018

from DemandRange import DemandRange

dr = DemandRange()
print(dr.get_lower_bound())
print(dr.get_upper_bound())
dr.set_lower_bound(10)
dr.set_upper_bound(11)
print(dr.get_lower_bound())
print(dr.get_upper_bound())
print(dr.generate_demand())
