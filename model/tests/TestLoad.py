from Load import Load
from DemandRange import DemandRange
l = Load()
dr = DemandRange()
l.set_demand_ranges(130)
dr.set_lower_bound(50)
dr.set_upper_bound(60)

